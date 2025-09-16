# utils.py
import tensorflow as tf
import numpy as np
from typing import List, Optional

# --- BEGIN: DirichletPartitioner 코드 (변경 없음) ---
class DirichletPartitioner:
    """Dirichlet 분포에 따라 데이터를 분할하는 클래스."""

    def __init__(
        self,
        num_partitions: int,
        partition_by: str = "labels",
        alpha: float = 0.5,
        min_partition_size: int = 10,
        seed: Optional[int] = 42,
    ):
        self.num_partitions = num_partitions
        if partition_by not in ["labels"]:
            raise ValueError(
                f"Unsupported value for partition_by: {partition_by}. "
                f"Currently, only 'labels' is supported."
            )
        self.partition_by = partition_by
        self.alpha = alpha
        self.min_partition_size = min_partition_size
        self.rng = np.random.default_rng(seed=seed)
        self.partitions: List[np.ndarray] = []

    def partition(self, data: np.ndarray) -> List[np.ndarray]:
        """데이터를 분할하고 파티션 목록을 반환합니다."""
        labels = data.flatten()
        num_items = len(labels)
        num_classes = len(np.unique(labels))
        
        self.partitions = [[] for _ in range(self.num_partitions)]
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        class_distribution = self.rng.dirichlet(
            [self.alpha] * num_classes, self.num_partitions
        )

        for class_id in range(num_classes):
            indices_for_class = class_indices[class_id]
            num_indices_for_class = len(indices_for_class)
            
            proportions = class_distribution[:, class_id]
            proportions = proportions / proportions.sum()
            allocations = (proportions * num_indices_for_class).astype(int)
            
            remaining = num_indices_for_class - allocations.sum()
            add_ons = self.rng.choice(self.num_partitions, remaining)
            for p_id in add_ons:
                allocations[p_id] += 1

            start = 0
            for p_id in range(self.num_partitions):
                end = start + allocations[p_id]
                self.partitions[p_id].extend(indices_for_class[start:end])
                start = end
        
        self.partitions = [np.array(p) for p in self.partitions]
        return self.partitions
# --- END: DirichletPartitioner 코드 ---

def load_data():
    """CIFAR-10 데이터를 로드하고 정규화합니다."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def get_model():
    """간단한 CNN 모델을 생성하고 컴파일합니다."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # 성능 개선을 위해 학습률을 약간 낮춤
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=["accuracy"])
    return model

def apply_trigger(images, trigger):
    """이미지에 학습된 트리거를 적용합니다."""
    triggered_images = tf.clip_by_value(images + trigger, 0.0, 1.0)
    return triggered_images