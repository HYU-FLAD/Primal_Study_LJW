import copy
import random
import numpy as np
import torch
from tqdm import tqdm

from model import get_model
from client import train, test # 기존 train, test 함수 임포트
from attacker import attacker_train # attacker_train 함수 추가 임포트
from get_data import get_data

class FedProx:
    def __init__(self, args):
        self.args = args
        # Load global model and data
        self.nn = get_model(args).to(args.device)
        self.client_loaders, self.test_loader = get_data(args)

        # --- 메모리 효율적인 클라이언트 관리 ---
        # 1. 모든 클라이언트의 모델을 CPU에 초기화하여 GPU 메모리 절약
        self.nns = []
        for i in range(self.args.K):
            model = copy.deepcopy(self.nn)
            model.name = f'client_{i}'
            self.nns.append(model.to('cpu'))

        # 2. 옵티마이저의 '상태'만 저장할 리스트를 CPU에 초기화
        # (객체 자체가 아닌 state_dict를 저장하여 메모리 부담 최소화)
        self.client_optimizer_states = [None] * self.args.K
        # ------------------------------------

        # 백도어 트리거 패턴 생성
        if self.args.is_attacker:
            num_channels = self.args.num_channels
            self.trigger_pattern = torch.ones(num_channels, 4, 4, device=self.args.device)
            print(f"Backdoor attack enabled. Attacker ID: {self.args.attacker_id}, Target Label: {self.args.trigger_label}")
        else:
            self.trigger_pattern = None

    def server(self):
        print("Starting FedProx server...")
        for t in range(self.args.r):
            print(f'Communication Round {t + 1}/{self.args.r}:')
            
            # 클라이언트 샘플링
            m = max(int(self.args.C * self.args.K), 1)
            index = random.sample(range(self.args.K), m)
            print(f"Selected clients: {index}")
            
            # 글로벌 모델 파라미터를 선택된 클라이언트들에게 전송
            self.dispatch(index)

            # 클라이언트 업데이트 (학습)
            updated_models = self.client_update(index)
            
            # 글로벌 모델 취합
            self.aggregation(updated_models)
            
            # 글로벌 모델 평가
            self.global_test()

        return self.nn

    def aggregation(self, updated_models):
        # updated_models는 CPU에 있으므로, total_data_len 계산은 그대로 수행
        total_data_len = sum(model.len for model in updated_models)
        if total_data_len == 0:
            return # 학습된 데이터가 없는 경우 종료

        # 글로벌 모델 파라미터를 기준으로 취합을 위한 텐서 초기화 (CPU에서 수행)
        aggregated_params = {k: torch.zeros_like(v.to('cpu').data) for k, v in self.nn.named_parameters()}

        for model in updated_models:
            weight = model.len / total_data_len
            for k, v in model.named_parameters():
                aggregated_params[k] += v.data * weight # v.data는 이미 CPU에 있음

        # 취합된 파라미터를 다시 글로벌 모델(GPU)로 로드
        for k, v in self.nn.named_parameters():
            v.data = aggregated_params[k].clone().to(self.args.device)

    def dispatch(self, index):
        # 글로벌 모델(GPU)의 파라미터를 선택된 로컬 모델(CPU)들로 복사
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone().cpu()

    def client_update(self, index):
        updated_models = []
        for k in index:
            train_loader = self.client_loaders[k]
            
            # --- 동적 옵티마이저 생성 및 상태 로드 ---
            # 1. 학습을 위해 클라이언트 모델을 GPU로 이동
            local_model = self.nns[k].to(self.args.device)
            
            # 2. 학습에 사용할 새로운 옵티마이저 객체를 GPU 상에서 생성
            if self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(local_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)

            # 3. 서버(CPU)에 저장된 이전 옵티마이저 상태가 있으면, 현재 옵티마이저(GPU)로 로드
            if self.client_optimizer_states[k] is not None:
                optimizer.load_state_dict(self.client_optimizer_states[k])

            # 4. 클라이언트 학습 진행
            if self.args.is_attacker and k == self.args.attacker_id:
                trained_model = attacker_train(self.args, local_model, self.nn, optimizer, train_loader, self.trigger_pattern)
            else:
                trained_model = train(self.args, local_model, self.nn, optimizer, train_loader)

            # 5. 학습 후, 옵티마이저의 최신 상태를 state_dict로 추출하여 CPU에 저장
            self.client_optimizer_states[k] = optimizer.state_dict()

            # 6. 학습된 모델과 옵티마이저 상태를 CPU로 이동시켜 GPU 메모리 확보
            updated_models.append(trained_model.to('cpu'))
            # ---------------------------------------------

        return updated_models

    def global_test(self):
        # 테스트 시 트리거 패턴을 함께 전달하여 ASR 측정
        test(self.args, self.nn, self.test_loader, self.trigger_pattern)
