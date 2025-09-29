# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 11:52
@Author: KI
@File: args.py
@Motto: Hungry And Humble
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # ======================== Federated Learning Hyperparameters ========================
    parser.add_argument('--E', type=int, default=2, 
                        help='number of local epochs (default: 5, recommended to prevent client drift)')
    parser.add_argument('--r', type=int, default=50, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=50, help='number of total clients')
    parser.add_argument('--C', type=float, default=0.1, help='sampling rate (fraction of clients per round)')
    parser.add_argument('--B', type=int, default=128, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for client optimizers')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant for FedProx')
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer (e.g., adam, sgd)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    # ======================== Backdoor Attack Arguments ========================
    parser.add_argument('--is_attacker', action='store_true', help='Enable backdoor attack simulation')
    parser.add_argument('--attacker_id', type=int, default=0, help='ID of the malicious client')
    parser.add_argument('--trigger_label', type=int, default=7, help='Target label for the backdoor attack (e.g., 7 for "horse" in CIFAR-10)')
    parser.add_argument('--poison_rate', type=float, default=0.5, help='Fraction of data to poison in a batch for the attacker')

    # ======================== Model and Data Arguments ========================
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture (e.g., cnn, resnet50)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use (e.g., cifar10, cifar100)')
    # parser.add_argument('--input_dim', type=int, default=28, help='DEPRECATED - will be set automatically')
    parser.add_argument('--num_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--num_classes', type=int, default=10, help='number of output classes')
    
    # ======================== Data Distribution Arguments ========================
    parser.add_argument('--iid', action='store_true', help='set to use IID data distribution')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='alpha parameter for dirichlet distribution (controls non-iid degree, smaller is more non-iid)')
    parser.add_argument('--samples_per_client', type=int, default=500, 
                        help='Number of data samples to assign to each client. -1 means distribute all available data.')
    
    # ======================== Environment Arguments ========================
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # python main.py --model vit --pretrained --lr 0.001 --B 32 --E 5

    args = parser.parse_args()


    # ======================== Auto-configure Arguments ========================
    # Automatically set the number of classes based on the dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    
    print(f"Running with the following configuration:")
    print(f"  - Model: {args.model.upper()}")
    print(f"  - Dataset: {args.dataset.upper()} ({args.num_classes} classes)")
    print(f"  - Rounds: {args.r}, Local Epochs: {args.E}")
    print(f"  - Distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"  - Device: {args.device}")

    return args