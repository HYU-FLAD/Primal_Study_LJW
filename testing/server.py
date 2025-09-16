import flwr as fl

strategy = fl.server.strategy.FedProx(
    proximal_mu=1.0,  # FedProx의 하이퍼파라미터 예시
)


fl.server.start_server(server_address="0.0.0.0:8080", 
                       config=fl.server.ServerConfig(num_rounds=3),
                       strategy=strategy,
                       )