import tensorflow as tf
from tensorflow.keras import backend as K
import flwr as fl
import numpy as np
from utils import apply_trigger # 이전 단계에서 작성한 유틸리티

class BenignClient(fl.client.NumPyClient):
    """일반 (Benign) 클라이언트"""
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=16, verbose=0)
        accuracy = history.history["accuracy"][-1]

        K.clear_session()

        return self.model.get_weights(), len(self.x_train), {"accuracy": accuracy}


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": accuracy}


class MaliciousClient(fl.client.NumPyClient):
    """PFedBA 공격을 수행하는 악의적인 클라이언트"""
    def __init__(self, model, x_train, y_train, target_label=0, poisoning_rate=0.2, num_rounds=20): # num_rounds 추가
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.target_label = target_label
        self.poisoning_rate = poisoning_rate
        self.num_rounds = num_rounds # 마지막 라운드 확인을 위해 저장

        self.trigger = tf.Variable(tf.random.uniform((32, 32, 3), minval=-0.1, maxval=0.1), 
                                   trainable=True, name="trigger")
        self.trigger_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def get_parameters(self, config):
        return self.model.get_weights()

    def _generate_stealthy_trigger(self, lambda_align=0.7, steps=5):
        """1단계: 정렬 최적화 기반 트리거 생성 (메모리 최적화 버전)"""
        
        # ==================== 수정된 로직 ====================
        # 전체 데이터 대신 작은 배치(예: 32개)만 사용하여 메모리 사용량 줄이기
        # num_to_poison = int(len(self.x_train) * self.poisoning_rate)
        batch_size_for_trigger = 32 # OOM 발생 시 이 값을 16 등으로 더 줄여보세요.
        # =====================================================

        mal_indices = np.random.choice(len(self.x_train), batch_size_for_trigger, replace=False)
        x_mal = tf.convert_to_tensor(self.x_train[mal_indices])
        y_mal_orig = tf.convert_to_tensor(self.y_train[mal_indices])
        
        # y_mal_target의 shape을 (batch_size,) 형태로 변경
        y_mal_target = tf.ones(shape=(batch_size_for_trigger,), dtype=tf.int64) * self.target_label

        print(f"  - Optimizing stealthy trigger using a batch of {batch_size_for_trigger}...")
        
        for step in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(self.trigger)
                with tf.GradientTape() as inner_tape_mal:
                    triggered_images = apply_trigger(x_mal, self.trigger)
                    y_pred_mal = self.model(triggered_images, training=False)
                    loss_mal = self.loss_fn(y_mal_target, y_pred_mal)
                grad_mal = inner_tape_mal.gradient(loss_mal, self.model.trainable_variables)
                
                with tf.GradientTape() as inner_tape_clean:
                    y_pred_clean = self.model(x_mal, training=False)
                    loss_clean = self.loss_fn(y_mal_orig, y_pred_clean)
                grad_clean = inner_tape_clean.gradient(loss_clean, self.model.trainable_variables)
                
                grad_diff_norm = tf.linalg.global_norm([g_m - g_c for g_m, g_c in zip(grad_mal, grad_clean)])
                loss_backdoor = loss_mal
                align_loss = lambda_align * grad_diff_norm + (1 - lambda_align) * loss_backdoor
            
            trigger_grad = tape.gradient(align_loss, self.trigger)
            self.trigger_optimizer.apply_gradients([(trigger_grad, self.trigger)])
        print(f"  - Trigger optimization finished. Align Loss: {align_loss.numpy():.4f}")


    def fit(self, parameters, config):
        print("\nMalicious client training starts.")
        self.model.set_weights(parameters)

        self._generate_stealthy_trigger()

        # +++ [추가된 로직] 마지막 라운드에서 트리거 저장 +++
        current_round = config.get("server_round")
        if current_round is not None and current_round == self.num_rounds:
            print("Saving final malicious trigger...")
            np.save("final_trigger.npy", self.trigger.numpy())
            print("Final trigger saved to 'final_trigger.npy'")
        
        num_to_poison = int(len(self.x_train) * self.poisoning_rate)
        poison_indices = np.random.choice(len(self.x_train), num_to_poison, replace=False)
        x_poisoned = np.copy(self.x_train)
        y_poisoned = np.copy(self.y_train)
        triggered_part = apply_trigger(self.x_train[poison_indices], self.trigger.numpy())
        x_poisoned[poison_indices] = triggered_part
        y_poisoned[poison_indices] = self.target_label

        print("  - Training local model on poisoned data...")
        history = self.model.fit(x_poisoned, y_poisoned, epochs=2, batch_size=16, verbose=0)
        accuracy = history.history["accuracy"][-1]
        return self.model.get_weights(), len(self.x_train), {"accuracy": accuracy}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        
        not_target_indices = np.where(self.y_train.flatten() != self.target_label)[0]
        x_asr_test = apply_trigger(self.x_train[not_target_indices], self.trigger.numpy())
        y_asr_test = self.y_train[not_target_indices]
        
        predictions = self.model.predict(x_asr_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        asr = np.mean(predicted_labels == self.target_label)

        return loss, len(self.x_train), {"accuracy": acc, "attack_success_rate": asr}