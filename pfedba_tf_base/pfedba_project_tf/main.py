# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse # argparse 임포트

from utils import load_data, get_model, DirichletPartitioner
from client import BenignClient, MaliciousClient

# --- 방어 기법 함수들 ---

def flatten_weights(weights):
    """모델의 가중치 리스트를 하나의 1D 벡터로 변환합니다."""
    return np.concatenate([w.flatten() for w in weights])

def multi_krum_selection(client_updates, num_to_keep, num_malicious):
    """
    Multi-Krum 알고리즘을 사용하여 정직할 것으로 예상되는 클라이언트를 선택합니다.
    Args:
        client_updates (list): 모든 클라이언트의 가중치 업데이트 리스트.
        num_to_keep (int): 선택할 클라이언트의 수 (m).
        num_malicious (int): 시스템에서 가정한 악성 클라이언트의 최대 수 (f).
    Returns:
        list: 선택된 클라이언트의 인덱스 리스트.
    """
    num_clients = len(client_updates)
    if num_to_keep >= num_clients - num_malicious:
        print("Warning: Krum's condition not met (m >= n-f). Selecting all clients.")
        return list(range(num_clients))

    # 각 클라이언트의 가중치를 1D 벡터로 변환
    flat_updates = [flatten_weights(update) for update in client_updates]
    
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i, num_clients):
            # 유클리드 거리의 제곱 계산
            dist = np.sum((flat_updates[i] - flat_updates[j]) ** 2)
            distances[i, j] = distances[j, i] = dist

    scores = np.zeros(num_clients)
    for i in range(num_clients):
        # 각 클라이언트로부터 다른 클라이언트까지의 거리
        dists_i = np.sort(distances[i])
        # 가장 가까운 n-f-2개의 클라이언트 거리 합산 (자기 자신 제외)
        # 자기 자신(거리 0)을 제외하기 위해 인덱스 1부터 시작
        scores[i] = np.sum(dists_i[1 : num_clients - num_malicious - 1])
        
    # 점수가 가장 낮은 클라이언트부터 순서대로 인덱스 정렬
    sorted_indices = np.argsort(scores)
    
    # 상위 num_to_keep개의 클라이언트 인덱스 반환
    selected_indices = sorted_indices[:num_to_keep].tolist()
    
    return selected_indices

def trimmed_mean_aggregation(client_updates, beta):
    """
    Trimmed Mean 알고리즘을 사용하여 가중치를 집계합니다.
    Args:
        client_updates (list): 모든 클라이언트의 가중치 업데이트 리스트.
        beta (float): 각 끝에서 잘라낼 비율 (0 <= beta < 0.5).
    Returns:
        list: 집계된 새로운 글로벌 가중치.
    """
    num_clients = len(client_updates)
    num_to_trim = int(beta * num_clients)
    
    if num_to_trim * 2 >= num_clients:
        raise ValueError("Beta is too high, trimming all clients.")

    aggregated_weights = []
    # 모델의 각 레이어(가중치 행렬)별로 처리
    for layer_idx in range(len(client_updates[0])):
        # 모든 클라이언트의 현재 레이어 가중치를 쌓음
        stacked_layer = np.stack([client[layer_idx] for client in client_updates])
        
        # 가중치 축(axis=0)을 따라 정렬
        sorted_layer = np.sort(stacked_layer, axis=0)
        
        # 양 끝에서 beta 비율만큼 잘라냄
        trimmed_layer = sorted_layer[num_to_trim : num_clients - num_to_trim]
        
        # 남은 가중치들의 평균 계산
        aggregated_layer = np.mean(trimmed_layer, axis=0)
        aggregated_weights.append(aggregated_layer)
        
    return aggregated_weights


# --- 기존 함수들 ---

def weighted_average(client_updates, data_sizes):
    """Federated Averaging 알고리즘을 사용하여 가중치를 집계합니다."""
    total_data_size = sum(data_sizes)
    aggregated_weights = [np.zeros_like(w) for w in client_updates[0]]

    for i, client_weight in enumerate(client_updates):
        data_size = data_sizes[i]
        for layer_idx, layer_weight in enumerate(client_weight):
            aggregated_weights[layer_idx] += (data_size / total_data_size) * layer_weight
            
    return aggregated_weights

def main(args):
    """TensorFlow로 연합 학습 시뮬레이션을 직접 실행합니다."""
    # 1. 데이터 로드 및 분할
    print("Loading and partitioning data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    partitioner = DirichletPartitioner(num_partitions=args.num_clients, alpha=0.5)
    client_indices = partitioner.partition(y_train)
    
    # 2. 글로벌 모델 및 클라이언트 초기화
    print("Initializing global model and clients...")
    global_model = get_model()
    
    clients = []
    for cid in range(args.num_clients):
        indices = client_indices[cid]
        client_model = get_model()
        
        if cid < args.num_malicious:
            client = MaliciousClient(cid, client_model, x_train[indices], y_train[indices])
            print(f"Created Malicious Client: {cid}")
        else:
            client = BenignClient(cid, client_model, x_train[indices], y_train[indices])
        clients.append(client)

    # 3. 연합 학습 라운드 시작
    print(f"\nStarting Federated Learning Simulation with Defense: '{args.defense}'")
    for round_num in range(1, args.num_rounds + 1):
        print(f"--- Round {round_num}/{args.num_rounds} ---")
        
        global_weights = global_model.get_weights()
        client_updates = []
        data_sizes = []
        
        for client in tqdm(clients, desc="Client Training"):
            is_last_round = (round_num == args.num_rounds)
            updated_weights = client.train(global_weights, epochs=args.local_epochs, is_last_round=is_last_round)
            client_updates.append(updated_weights)
            data_sizes.append(len(client.x_train))
            
        # --- 방어 기법에 따른 집계/선택 로직 ---
        if args.defense == 'krum':
            selected_indices = multi_krum_selection(
                client_updates, num_to_keep=args.krum_m, num_malicious=args.num_malicious
            )
            print(f"Krum selected clients: {selected_indices}")
            # 선택된 클라이언트의 업데이트와 데이터 크기만 필터링
            selected_updates = [client_updates[i] for i in selected_indices]
            selected_sizes = [data_sizes[i] for i in selected_indices]
            aggregated_weights = weighted_average(selected_updates, selected_sizes)

        elif args.defense == 'trimmed_mean':
            aggregated_weights = trimmed_mean_aggregation(
                client_updates, beta=args.trim_beta
            )
            num_trimmed = int(args.trim_beta * len(clients))
            print(f"Trimmed Mean: Trimmed {num_trimmed} updates from each end.")
            
        else: # 'none' (기본값)
            aggregated_weights = weighted_average(client_updates, data_sizes)
        
        global_model.set_weights(aggregated_weights)
        
        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        
    # 4. 최종 모델 저장
    print("Federated learning finished. Saving final global model...")
    np.save("final_model_weights.npy", global_model.get_weights(), allow_pickle=True)
    print("Final global model weights saved to 'final_model_weights.npy'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Simulation with Defenses")
    # 실험 기본 파라미터
    parser.add_argument('--num_clients', type=int, default=10, help='Total number of clients.')
    parser.add_argument('--num_malicious', type=int, default=1, help='Number of malicious clients.')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of federated learning rounds.')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of epochs for local client training.')
    
    # 방어 기법 파라미터
    parser.add_argument('--defense', type=str, default='none', choices=['none', 'krum', 'trimmed_mean'], help='Defense mechanism to use.')
    parser.add_argument('--krum_m', type=int, default=8, help='Number of clients to select in Krum (m). Should be n > m >= n-f.')
    parser.add_argument('--trim_beta', type=float, default=0.2, help='Fraction to trim from each end for Trimmed Mean (beta).')
    
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main(args)