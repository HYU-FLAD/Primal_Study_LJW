import tensorflow as tf
import numpy as np
import os

from utils import get_model, apply_trigger

import pickle

# TensorFlow C++ 로깅 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """
    학습이 완료된 글로벌 모델과 트리거를 로드하여 최종 성능을 평가합니다.
    1. 정상 데이터에 대한 정확도 (Main Task Accuracy)
    2. 트리거가 삽입된 데이터에 대한 공격 성공률 (Attack Success Rate)
    """
    print("--- Starting Final Evaluation of the Backdoor Attack ---")

    # 1. 필요 파일 로드
    try:
        # 최종 글로벌 모델 가중치 로드
        
        # final_weights = np.load("final_model_weights.npy", allow_pickle=True)

        with open("final_model_weights.pkl", "rb") as f:
            final_weights = pickle.load(f)
        print("✅ Successfully loaded 'final_model_weights.npy'")
        
        # 최종 트리거 로드
        final_trigger = np.load("final_trigger.npy")
        print("✅ Successfully loaded 'final_trigger.npy'")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required file: {e.filename}")
        print("Please run main.py to generate the model weights and trigger files first.")
        return
    
    # 2. 모델 및 테스트 데이터 준비
    # 모델 생성 및 가중치 설정
    model = get_model()

    # model.set_weights(final_weights)
    model.set_weights(final_weights)
    
    print("✅ Model created and weights are set.")

    # CIFAR-10 테스트 데이터 로드
    (_, (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    print(f"✅ Test dataset loaded. (Shape: {x_test.shape})")

    # 3. 평가 시작
    # 평가 1: 정상 데이터에 대한 정확도 (Clean Accuracy)
    print("\n--- Evaluating Main Task Accuracy on Clean Data ---")
    loss, clean_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"✔️ Clean Data Accuracy: {clean_accuracy * 100:.2f}%")
    
    # 평가 2: 백도어 공격 성공률 (Attack Success Rate - ASR)
    print("\n--- Evaluating Attack Success Rate (ASR) on Triggered Data ---")
    target_label = 0 # MaliciousClient에서 설정한 타겟 레이블과 동일해야 함
    
    # 타겟 레이블이 아닌 이미지들만 선택
    not_target_indices = np.where(y_test.flatten() != target_label)[0]
    x_test_not_target = x_test[not_target_indices]
    
    # 선택된 이미지에 트리거 적용
    x_test_triggered = apply_trigger(x_test_not_target, final_trigger)
    
    # 트리거가 적용된 이미지에 대한 모델 예측
    predictions = model.predict(x_test_triggered, batch_size=16, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 모델이 타겟 레이블로 잘못 예측한 비율 계산 (ASR)
    asr = np.mean(predicted_labels == target_label)
    print(f"✔️ Attack Success Rate (ASR): {asr * 100:.2f}%")
    print(f"(Model mistook triggered images as class '{target_label}' this often)")

    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    main()