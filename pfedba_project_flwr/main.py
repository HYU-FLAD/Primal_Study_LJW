import os
# 1. TensorFlow C++ 로깅 레벨 설정 (INFO, WARNING 메시지 비활성화)
# '2'는 ERROR 메시지만 표시하라는 의미입니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
# 2. Python의 DeprecationWarning 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

import pickle

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
import tensorflow as tf

from client import BenignClient, MaliciousClient
from utils import get_model, DirichletPartitioner, apply_trigger
import numpy as np

from tensorflow.keras import mixed_precision

# Mixed Precision Training 활성화
mixed_precision.set_global_policy("mixed_float16")
print("✅ Mixed Precision enabled (mixed_float16)")


# --- 실험 파라미터 설정 ---
NUM_CLIENTS = 10
NUM_MALICIOUS = 1
NUM_ROUNDS = 20

def client_fn(cid: str) -> fl.client.Client:
    """
    클라이언트 생성 함수.
    메모리 효율성을 위해 각 클라이언트 프로세스 내부에서 데이터를 로드합니다.
    (Context API 호환성 문제를 피하기 위해 cid를 직접 받는 이전 방식을 사용합니다.)
    """
    # 1. 이 함수 내부에서 데이터를 로드합니다.
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0

    # 2. 파티셔너를 사용하여 이 클라이언트의 데이터 인덱스를 결정합니다.
    partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, alpha=0.5, seed=42)
    partitioner.partition(y_train)
    
    # 3. 원래 방식대로 cid를 직접 사용합니다.
    client_id = int(cid)
    client_indices = partitioner.partitions[client_id]

    # 4. 해당 인덱스의 데이터만 사용합니다.
    x_train_partition = x_train[client_indices]
    y_train_partition = y_train[client_indices]

    model = get_model()

    if client_id < NUM_MALICIOUS:
        print(f"Creating Malicious Client: {cid}")
        return MaliciousClient(model, x_train_partition, y_train_partition).to_client()
    else:
        return BenignClient(model, x_train_partition, y_train_partition).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # 정확도 집계
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_accuracy = sum(accuracies) / sum(examples)
    
    # 악성 클라이언트의 ASR만 정확하게 집계
    asr_metrics = [(num_examples, m["attack_success_rate"]) for num_examples, m in metrics if "attack_success_rate" in m]
    if asr_metrics:
        total_asr = sum(num_examples * asr for num_examples, asr in asr_metrics)
        total_malicious_examples = sum(num_examples for num_examples, _ in asr_metrics)
        avg_asr = total_asr / total_malicious_examples if total_malicious_examples > 0 else 0.0
        print(f"Round Metrics: Avg Accuracy={avg_accuracy:.4f}, Avg ASR={avg_asr:.4f}")
        return {"accuracy": avg_accuracy, "attack_success_rate": avg_asr}
    else:
        print(f"Round Metrics: Avg Accuracy={avg_accuracy:.4f}")
        return {"accuracy": avg_accuracy}

if __name__ == "__main__":
    # GPU 메모리 성장 옵션 설정 (프로세스 충돌 방지)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow: Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    
    # 서버 측 평가를 위한 테스트 데이터 로드
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0

    def evaluate(server_round, parameters, config):
        model = get_model()
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"--- Global Model Eval (Round {server_round}) --- Accuracy: {accuracy:.4f}")

        # +++ [추가된 로직] 마지막 라운드에서 모델 가중치 저장 +++
        if server_round == NUM_ROUNDS:
            print("\nSaving final global model weights...")

            # np.save("final_model_weights.npy", parameters)
            with open("final_model_weights.pkl", "wb") as f:
                pickle.dump([np.asarray(p) for p in parameters], f)
            
            print("Final weights saved to 'final_model_weights.npy'")
            
        return loss, {"global_accuracy": accuracy}

    # FedProx 전략 설정
    strategy = fl.server.strategy.FedProx(
        fraction_fit=0.1,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        proximal_mu=0.1,
        evaluate_fn=evaluate,
        fit_metrics_aggregation_fn=weighted_average,
    )

    # --- 방어 기법 테스트 시 아래 주석을 해제하여 교체 ---
    # Multi-Krum
    # strategy = fl.server.strategy.Krum(
    #     num_fit_clients_to_keep=8, # 10명 중 8명 선택 (2명 탈락)
    #     fraction_fit=0.1, min_fit_clients=10, min_available_clients=NUM_CLIENTS,
    #     evaluate_fn=evaluate
    # )

    # Trimmed Mean
    # strategy = fl.server.strategy.TrimmedMean(
    #     beta=0.2, # 상위 20%, 하위 20% 탈락 (10명 중 4명 탈락)
    #     fraction_fit=0.1, min_fit_clients=10, min_available_clients=NUM_CLIENTS,
    #     evaluate_fn=evaluate
    # )

    # 시뮬레이션 실행
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        # OOM 방지를 위해 동시 실행 클라이언트 수를 2개로 제한
        client_resources={"num_cpus": 2, "num_gpus": 1.0} 
    )