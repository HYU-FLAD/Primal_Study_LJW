# -*- coding:utf-8 -*-
"""
@Time: 2025/09/28
@Author: Gemini
@File: attacker.py
@Motto: For Educational and Defensive Research Purposes
"""
import torch
from torch import nn

def add_trigger(images, trigger_pattern, target_label, poison_rate):
    """
    주어진 이미지 배치에 백도어 트리거를 삽입합니다.

    Args:
        images (torch.Tensor): 원본 이미지 텐서.
        trigger_pattern (torch.Tensor): 이미지에 삽입할 트리거 패턴.
        target_label (int): 백도어 공격의 목표 레이블.
        poison_rate (float): 배치 내에서 데이터를 오염시킬 비율.

    Returns:
        torch.Tensor, torch.Tensor: 트리거가 삽입된 이미지와 조작된 레이블.
    """
    # 복사본을 만들어 원본 데이터를 보존
    poisoned_images = images.clone()
    poisoned_labels = torch.full((images.size(0),), target_label, dtype=torch.long, device=images.device)
    
    # poison_rate에 따라 일부 이미지에만 트리거 적용 (Stealth를 위해)
    num_to_poison = int(poison_rate * len(images))
    if num_to_poison == 0:
        return images, None # 레이블은 변경하지 않음

    poison_indices = torch.randperm(len(images))[:num_to_poison]

    for i in poison_indices:
        # 트리거 패턴을 이미지의 특정 위치(예: 우측 하단)에 추가
        c, h, w = poisoned_images[i].shape
        tc, th, tw = trigger_pattern.shape
        poisoned_images[i, :, h-th:, w-tw:] = trigger_pattern

    return poisoned_images, poisoned_labels, poison_indices


def attacker_train(args, model, global_model, train_loader, trigger_pattern):
    """
    악성 클라이언트(Attacker)를 위한 맞춤형 학습 함수.
    일반 학습과 백도어 주입 학습을 함께 수행하여 은밀함을 유지합니다.
    """
    model.train()
    model.len = len(train_loader.dataset)

    # Optimizer 설정
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss()
    print(f'Attacker (Client {args.attacker_id}) training on {model.len} samples...')

    for epoch in range(args.E):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            # 1. 일반(Clean) 데이터로 모델 학습 (정상 성능 유지를 위함)
            optimizer.zero_grad()
            outputs = model(images)
            clean_loss = loss_function(outputs, labels)
            
            # 2. 백도어(Poisoned) 데이터 생성 및 학습
            poisoned_images, poisoned_labels, poison_indices = add_trigger(images, trigger_pattern, args.trigger_label, args.poison_rate)
            
            poison_loss = 0.0
            if poison_indices is not None and len(poison_indices) > 0:
                poison_outputs = model(poisoned_images[poison_indices])
                poison_loss = loss_function(poison_outputs, poisoned_labels[poison_indices])
            
            # 3. FedProx 손실 계산
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
            
            # 4. 최종 손실 = 일반 손실 + 백도어 손실 + FedProx 손실
            # 논문에 따라 두 손실을 결합하여 최적화 (가중치 alpha는 조절 가능)
            alpha = 0.5 
            total_loss = alpha * clean_loss + (1 - alpha) * poison_loss + (args.mu / 2) * proximal_term
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        print(f'Attacker Epoch {epoch+1}/{args.E}, Loss: {epoch_loss/len(train_loader):.4f}')

    return model