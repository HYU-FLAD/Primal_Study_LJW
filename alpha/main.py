# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import FedProx

def main():
    # Parse command line arguments
    args = args_parser()
    
    # Initialize and run the FedProx server
    fed_prox_server = FedProx(args)
    fed_prox_server.server()
    
    print("Federated training finished.")
    print("Final global model evaluation:")
    fed_prox_server.global_test()

if __name__ == '__main__':
    main()