# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import copy
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
import numpy as np


def train(args, model, global_model, optimizer, train_loader):
    """
    Trains a client model for one local epoch.
    """
    model.train()
    model.len = len(train_loader.dataset) # 데이터셋 크기 저장
    
    loss_function = nn.CrossEntropyLoss()

    print(f'Client training on {model.len} samples...')
    
    for epoch in range(args.E):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)

            # FedProx proximal term
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)**2
            
            loss = loss_function(outputs, labels) + (args.mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
    print(f'Client training finished. Final Loss: {epoch_loss/len(train_loader):.4f}')
    return model

def test(args, model, test_loader, trigger_pattern=None):
    """
    모델을 글로벌 테스트 데이터셋으로 테스트합니다.
    백도어 공격 시뮬레이션을 위해 ASR(Attack Success Rate) 측정도 추가합니다.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_function = nn.CrossEntropyLoss()

    # ASR 측정을 위한 변수
    asr_correct = 0
    asr_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            
            # 1. 기본 정확도(Main Task Accuracy) 측정
            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 2. 공격 성공률(ASR) 측정 (attacker 모드가 활성화된 경우)
            if args.is_attacker and trigger_pattern is not None:
                # 타겟 레이블이 아닌 이미지에만 트리거를 삽입하여 테스트
                non_target_indices = (labels != args.trigger_label).nonzero(as_tuple=True)[0]
                if len(non_target_indices) > 0:
                    images_with_trigger = images[non_target_indices].clone()
                    
                    # 모든 이미지에 트리거 삽입
                    for i in range(len(images_with_trigger)):
                        c, h, w = images_with_trigger[i].shape
                        tc, th, tw = trigger_pattern.shape
                        images_with_trigger[i, :, h-th:, w-tw:] = trigger_pattern

                    # 모델의 예측 확인
                    asr_outputs = model(images_with_trigger)
                    _, asr_predicted = torch.max(asr_outputs.data, 1)
                    
                    asr_total += len(images_with_trigger)
                    asr_correct += (asr_predicted == args.trigger_label).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if args.is_attacker and asr_total > 0:
        attack_success_rate = 100 * asr_correct / asr_total
        print(f'Attack Success Rate (ASR): {attack_success_rate:.2f}%')

    return accuracy, avg_loss