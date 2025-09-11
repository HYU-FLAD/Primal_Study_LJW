# vertical_fl/strategy.py

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple
from logging import INFO
from flwr.common.logger import log
import numpy as np

class ServerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Strategy(fl.server.strategy.FedAvg):
    def __init__( self, labels, total_embedding_size: int, client_embedding_size: int, num_splits: int, num_classes: int, batch_size: int, num_epochs: int, device: torch.device, *args, **kwargs, ) -> None:
        super().__init__( *args, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=num_splits, min_evaluate_clients=num_splits, min_available_clients=num_splits, **kwargs, )
        self.device = device
        self.model = ServerModel(total_embedding_size, num_classes).to(self.device)
        self.client_embedding_size = client_embedding_size
        self.num_splits = num_splits
        self.initial_parameters = ndarrays_to_parameters( [val.cpu().numpy() for _, val in self.model.state_dict().items()] )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.labels = torch.tensor(labels).long().to(self.device)
        self.batch_indices: List[np.ndarray] = []
        num_samples = len(labels)
        indices = np.arange(num_samples)
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                self.batch_indices.append(indices[i : i + batch_size])
        self.current_batch_indices = None

    def configure_fit(
        self, server_round: int, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if not self.batch_indices:
            log(INFO, "All mini-batches processed. Halting training.")
            return []

        self.current_batch_indices = self.batch_indices.pop(0)
        
        indices_list = self.current_batch_indices.tolist()
        indices_str = ",".join(map(str, indices_list))
        config = {"batch_indices_str": indices_str}
        
        fit_ins = FitIns(parameters, config)
        
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients
        )
        return [(client, fit_ins) for client in clients]


    def aggregate_fit(self, server_round, results, failures,):
        if not self.accept_failures and failures:
            return None, {}

        v_split_id_to_embedding = []
        for _, fit_res in results:
            v_split_id = int(fit_res.metrics["v_split_id"])
            
            embedding = torch.from_numpy(
                parameters_to_ndarrays(fit_res.parameters)[0]
            ).to(self.device)
            
            v_split_id_to_embedding.append((v_split_id, embedding))

        v_split_id_to_embedding.sort(key=lambda x: x[0])
        

        embeddings_aggregated = torch.cat(
            [emb for _, emb in v_split_id_to_embedding], dim=1
        )
        
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        
        output = self.model(embedding_server)
        
        batch_labels = self.labels[self.current_batch_indices]
        loss = self.criterion(output, batch_labels)
        
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        split_sizes = [self.client_embedding_size] * self.num_splits
        grads = embedding_server.grad.split(split_sizes, dim=1)
        
        # 클라이언트에게 보낼 그래디언트는 CPU로 다시 보낼 필요가 없습니다.
        # NumPy로 변환하면 자동으로 CPU 데이터가 됩니다.
        np_grads = [grad.cpu().numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        with torch.no_grad():
            output = self.model(embedding_server)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == batch_labels).sum().item()
            accuracy = correct / len(batch_labels)

            if server_round % 100 == 0 or server_round == 1:
                log(INFO, f"Round {server_round} - Mini-batch accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")

        metrics_aggregated = {"accuracy": accuracy}

        return parameters_aggregated, metrics_aggregated
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""
        if self.current_batch_indices is None:
            return []

        indices_list = self.current_batch_indices.tolist()
        indices_str = ",".join(map(str, indices_list))
        config = {"batch_indices_str": indices_str}
        
        eval_ins = EvaluateIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients
        )
        return [(client, eval_ins) for client in clients]


    def aggregate_evaluate( self, server_round, results, failures, ):
        return None, {}