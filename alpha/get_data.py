# -*- coding:utf-8 -*-
"""
@Time: 2025/09/28
@Author: Gemini
@File: get_data.py
@Motto: Hungry And Humble
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def iid_split(dataset, num_users):
    """
    Splits a dataset into IID subsets for each user. This is dynamic.
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def non_iid_split(dataset, num_users):
    """
    Splits a dataset into Non-IID subsets for each user.
    This version is now DYNAMIC based on the number of users (K).
    """
    # ==================== 수정된 로직 ====================
    num_shards = 200  # 샤드의 총 개수는 유지하여 데이터 분배의 단위를 결정
    num_imgs_per_shard = int(len(dataset) / num_shards)
    
    # 클라이언트 수에 따라 클라이언트당 할당될 샤드 수를 동적으로 계산
    shards_per_client = int(num_shards / num_users)
    
    print(f"Non-IID split: {num_shards} total shards, {shards_per_client} shards per client.")
    # =====================================================

    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array([s[1] for s in dataset.samples])

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Distribute shards to users
    for i in range(num_users):
        # Check if there are enough shards left
        if len(idx_shard) < shards_per_client:
            # If not enough, assign all remaining shards to the last few clients
            rand_set = set(idx_shard)
        else:
            rand_set = set(np.random.choice(idx_shard, shards_per_client, replace=False))
        
        idx_shard = list(set(idx_shard) - rand_set)
        
        for rand in rand_set:
            start_idx = rand * num_imgs_per_shard
            end_idx = start_idx + num_imgs_per_shard
            dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx]), axis=0)
            
    return dict_users

def get_data(args):
    """
    Loads the specified dataset and distributes it among clients.
    """
    print(f"Data processing for {args.dataset.upper()}...")

    if args.model.lower() == 'vit':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Applying ViT-specific transforms (resizing to 224x224).")
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported.")

    if args.iid:
        print("Distributing data in an IID fashion.")
        user_groups = iid_split(train_dataset, args.K)
    else:
        print("Distributing data in a Non-IID fashion.")
        user_groups = non_iid_split(train_dataset, args.K)
    

        # 각 클라이언트의 데이터 샘플 수를 제한하는 로직
    if args.samples_per_client > 0:
        print(f"Limiting data per client to {args.samples_per_client} samples.")
        for i in range(args.K):
            # Non-IID의 경우 데이터가 정렬되어 있을 수 있으므로, 섞어서 일부를 선택
            client_indices = np.array(list(user_groups[i]))
            np.random.shuffle(client_indices)
            user_groups[i] = client_indices[:args.samples_per_client]
    
    
    client_loaders = {}
    for i in range(args.K):
        dataset_subset = Subset(train_dataset, list(user_groups[i]))
        client_loaders[i] = DataLoader(dataset_subset, batch_size=args.B, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.B * 2, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

    return client_loaders, test_loader