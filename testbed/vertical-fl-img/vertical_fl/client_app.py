from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch

from vertical_fl.task import ClientModelMLP, ClientModelCNN, load_data


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, model_name, lr, device: torch.device):
        self.device = device
        self.v_split_id = v_split_id
        self.data = torch.tensor(data).float() # Load the full data once
        input_size = self.data.shape[1]
        if model_name.upper() == "CNN":
            self.model = ClientModelCNN(input_channels=1)
        elif model_name.upper() == "MLP":
            self.model = ClientModelMLP(input_size=input_size)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        # ######################################################################
        # MODIFIED: Receive string from config and convert back to list of ints
        # ######################################################################
        indices_str = config["batch_indices_str"]
        indices = [int(i) for i in indices_str.split(',')]
        batch = self.data[indices]
        
        self.model.train()
        embedding = self.model(batch)
        # ######################################################################
        
        return (
            [embedding.detach().numpy()],
            len(batch),
            {"v_split_id": int(self.v_split_id)},
        )

    def evaluate(self, parameters, config):
        # ######################################################################
        # MODIFIED: Receive string from config and convert back to list of ints
        # ######################################################################
        indices_str = config["batch_indices_str"]
        indices = [int(i) for i in indices_str.split(',')]
        batch = self.data[indices]

        self.model.train()
        self.optimizer.zero_grad()
        
        embedding = self.model(batch)
        
        grad_tensor = torch.from_numpy(parameters[self.v_split_id])
        
        embedding.backward(grad_tensor)
        self.optimizer.step()
        
        return 0.0, len(batch), {}
        # ######################################################################

def client_fn(context: Context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    lr = context.run_config["learning-rate"]
    model_name = context.run_config["model"]
    dataset_name = context.run_config["dataset"]

    partition, v_split_id = load_data(partition_id, num_partitions, dataset_name)
    
    return FlowerClient(v_split_id, partition, model_name, lr, device).to_client()


app = ClientApp(
    client_fn=client_fn,
)