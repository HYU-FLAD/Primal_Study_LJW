# client.py
import tensorflow as tf
import numpy as np
from utils import apply_trigger

class BenignClient:
    """일반 (Benign) 클라이언트 로직"""
    def __init__(self, cid, model, x_train, y_train):
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train(self, global_weights, epochs=1):
        """글로벌 가중치를 받아 로컬 데이터로 모델을 학습시킵니다."""
        print(f"  - Client {self.cid}: Benign training...")
        self.model.set_weights(global_weights)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=64, verbose=0)
        return self.model.get_weights()

class MaliciousClient:
    """PFedBA 공격을 수행하는 악의적인 클라이언트 로직"""
    def __init__(self, cid, model, x_train, y_train, target_label=0, poisoning_rate=0.2):
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.target_label = target_label
        self.poisoning_rate = poisoning_rate
        
        self.trigger = tf.Variable(tf.random.uniform((32, 32, 3), minval=-0.1, maxval=0.1), 
                                   trainable=True, name="trigger")
        self.trigger_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def _generate_stealthy_trigger(self, lambda_align=0.7, steps=5):
        """1단계: 정렬 최적화 기반 트리거 생성"""
        num_to_poison = int(len(self.x_train) * self.poisoning_rate)
        mal_indices = np.random.choice(len(self.x_train), num_to_poison, replace=False)
        x_mal = tf.convert_to_tensor(self.x_train[mal_indices])
        y_mal_orig = tf.convert_to_tensor(self.y_train[mal_indices])
        y_mal_target = tf.ones(shape=(num_to_poison, 1), dtype=tf.int64) * self.target_label

        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(self.trigger)
                with tf.GradientTape() as inner_tape_mal:
                    y_pred_mal = self.model(apply_trigger(x_mal, self.trigger), training=False)
                    loss_mal = self.loss_fn(y_mal_target, y_pred_mal)
                grad_mal = inner_tape_mal.gradient(loss_mal, self.model.trainable_variables)
                
                with tf.GradientTape() as inner_tape_clean:
                    y_pred_clean = self.model(x_mal, training=False)
                    loss_clean = self.loss_fn(y_mal_orig, y_pred_clean)
                grad_clean = inner_tape_clean.gradient(loss_clean, self.model.trainable_variables)
                
                grad_diff_norm = tf.linalg.global_norm([g_m - g_c for g_m, g_c in zip(grad_mal, grad_clean)])
                align_loss = lambda_align * grad_diff_norm + (1 - lambda_align) * loss_mal
            
            trigger_grad = tape.gradient(align_loss, self.trigger)
            if trigger_grad is not None:
                self.trigger_optimizer.apply_gradients([(trigger_grad, self.trigger)])

    def train(self, global_weights, epochs=1, is_last_round=False):
        """
        글로벌 가중치를 받아 트리거를 생성하고,
        데이터를 오염시킨 후 모델을 학습시킵니다.
        """
        print(f"  - Client {self.cid}: Malicious training...")
        self.model.set_weights(global_weights)

        # 1단계: 은밀한 트리거 생성
        self._generate_stealthy_trigger()
        
        # 마지막 라운드인 경우, 최종 트리거 저장
        if is_last_round:
            print(f"  - Client {self.cid}: Saving final trigger...")
            np.save("final_trigger.npy", self.trigger.numpy())
            print("  - Final trigger saved to 'final_trigger.npy'")

        # 2단계: 데이터셋 오염 및 로컬 모델 학습
        num_to_poison = int(len(self.x_train) * self.poisoning_rate)
        poison_indices = np.random.choice(len(self.x_train), num_to_poison, replace=False)

        x_poisoned = np.copy(self.x_train)
        y_poisoned = np.copy(self.y_train)

        triggered_part = apply_trigger(self.x_train[poison_indices], self.trigger.numpy())
        x_poisoned[poison_indices] = triggered_part
        y_poisoned[poison_indices] = self.target_label
        
        self.model.fit(x_poisoned, y_poisoned, epochs=epochs, batch_size=64, verbose=0)
        return self.model.get_weights()