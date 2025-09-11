# vertical_fl/server_app.py

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
import numpy as np
import torch  # ADDED
from flwr.common.logger import log # ADDED
from logging import INFO # ADDED

from vertical_fl.strategy import Strategy
from vertical_fl.task import load_server_labels, CLIENT_EMBEDDING_SIZE, NUM_VERTICAL_SPLITS

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Server is using device: {device}")

    dataset_name = context.run_config["dataset"]
    labels = load_server_labels(dataset_name)
    num_classes = len(np.unique(labels))

    total_embedding_size = CLIENT_EMBEDDING_SIZE * NUM_VERTICAL_SPLITS
    batch_size = context.run_config["batch-size"]
    
    num_epochs = 2 
    
    strategy = Strategy(
        labels=labels,
        total_embedding_size=total_embedding_size,
        client_embedding_size=CLIENT_EMBEDDING_SIZE,
        num_splits=NUM_VERTICAL_SPLITS,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
    )
    # ######################################################################

    # num_rounds는 이제 strategy 내부에서 계산된 배치 수에 따라 결정됨
    # pyproject.toml의 값은 최대 라운드 수로 동작함
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)