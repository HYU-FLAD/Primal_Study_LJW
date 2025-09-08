import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
import copy
from typing import List, Dict, Tuple, Optional
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("COMPLETE PFL SECURITY EXPERIMENT ENVIRONMENT")
print("Simulating PFedBA attacks and comprehensive defenses")
print("For educational and research purposes only")
print("=" * 70)

class TriggerGenerator:
    """Generates optimized triggers following PFedBA methodology"""
    
    def __init__(self, trigger_size: int = 4, trigger_location: str = 'top-left'):
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.current_trigger = None
        self.mask = None
        
    def initialize_trigger(self, input_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize trigger pattern and mask"""
        if len(input_shape) == 3:  # (C, H, W)
            c, h, w = input_shape
            trigger = torch.rand(c, h, w) * 0.5
            mask = torch.zeros(c, h, w)
            
            # Place trigger in specified location
            if self.trigger_location == 'top-left':
                mask[:, :self.trigger_size, :self.trigger_size] = 1
            elif self.trigger_location == 'bottom-right':
                mask[:, -self.trigger_size:, -self.trigger_size:] = 1
            else:  # random location
                start_h = np.random.randint(0, h - self.trigger_size)
                start_w = np.random.randint(0, w - self.trigger_size)
                mask[:, start_h:start_h+self.trigger_size, start_w:start_w+self.trigger_size] = 1
        
        else:  # Flattened input
            trigger = torch.rand(input_shape) * 0.5
            mask = torch.zeros(input_shape)
            # Set first trigger_size^2 elements as trigger
            mask[:self.trigger_size**2] = 1
            
        self.current_trigger = trigger
        self.mask = mask
        return trigger, mask
    
    def apply_trigger(self, data: torch.Tensor, trigger: torch.Tensor, 
                     mask: torch.Tensor) -> torch.Tensor:
        """Apply trigger to data using mask"""
        return data * (1 - mask) + trigger * mask
    
    def optimize_trigger_gradient_alignment(self, model: nn.Module, 
                                          clean_data: torch.Tensor,
                                          clean_labels: torch.Tensor,
                                          target_label: int,
                                          num_iterations: int = 100) -> torch.Tensor:
        """
        Optimize trigger for gradient alignment (Core PFedBA technique)
        """
        if self.current_trigger is None:
            trigger, mask = self.initialize_trigger(clean_data[0].shape)
        else:
            trigger = self.current_trigger.clone()
            mask = self.mask
        
        trigger.requires_grad_(True)
        optimizer = optim.Adam([trigger], lr=0.01)
        
        model.eval()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Apply trigger to create backdoor samples
            backdoor_data = self.apply_trigger(clean_data, trigger, mask)
            backdoor_labels = torch.full_like(clean_labels, target_label)
            
            # Compute gradients for main task (clean data)
            clean_output = model(clean_data)
            clean_loss = nn.CrossEntropyLoss()(clean_output, clean_labels)
            clean_gradients = torch.autograd.grad(clean_loss, model.parameters(), 
                                                create_graph=True, retain_graph=True)
            
            # Compute gradients for backdoor task
            backdoor_output = model(backdoor_data)
            backdoor_loss = nn.CrossEntropyLoss()(backdoor_output, backdoor_labels)
            backdoor_gradients = torch.autograd.grad(backdoor_loss, model.parameters(), 
                                                   create_graph=True, retain_graph=True)
            
            # Gradient alignment loss (minimize distance between gradients)
            gradient_alignment_loss = 0
            for clean_grad, backdoor_grad in zip(clean_gradients, backdoor_gradients):
                gradient_alignment_loss += torch.norm(clean_grad - backdoor_grad) ** 2
            
            # Loss alignment (minimize backdoor loss)
            loss_alignment = backdoor_loss
            
            # Combined objective (as per PFedBA paper)
            total_loss = gradient_alignment_loss + 0.1 * loss_alignment
            
            total_loss.backward()
            optimizer.step()
            
            # Clamp trigger values to reasonable range
            with torch.no_grad():
                trigger.clamp_(0, 1)
        
        self.current_trigger = trigger.detach()
        return trigger.detach()

class PFedBAAttacker:
    """Implements PFedBA attack strategy"""
    
    def __init__(self, target_label: int = 0, trigger_size: int = 4):
        self.target_label = target_label
        self.trigger_generator = TriggerGenerator(trigger_size)
        self.poisoning_rate = 0.1  # 10% of malicious client data is poisoned
        self.attack_history = []
        
    def poison_dataset(self, dataset: Dataset, trigger: torch.Tensor, 
                      mask: torch.Tensor) -> Dataset:
        """Create poisoned version of dataset"""
        poisoned_data = []
        
        for i, (data, label) in enumerate(dataset):
            if np.random.random() < self.poisoning_rate:
                # Apply trigger and change label
                poisoned_sample = self.trigger_generator.apply_trigger(
                    data.unsqueeze(0), trigger, mask).squeeze(0)
                poisoned_data.append((poisoned_sample, self.target_label))
            else:
                poisoned_data.append((data, label))
        
        return PoisonedDataset(poisoned_data)
    
    def execute_attack_round(self, global_model: nn.Module, 
                           malicious_client_data: List[Dataset],
                           round_num: int) -> Dict:
        """Execute one round of PFedBA attack"""
        
        # Sample data for trigger optimization
        sample_data = []
        sample_labels = []
        
        for dataset in malicious_client_data:
            for i, (data, label) in enumerate(dataset):
                if i >= 50:  # Limit samples for efficiency
                    break
                sample_data.append(data)
                sample_labels.append(label)
        
        if not sample_data:
            return {'success': False, 'reason': 'No data available'}
        
        sample_data = torch.stack(sample_data)
        sample_labels = torch.tensor(sample_labels)
        
        # Optimize trigger using gradient alignment
        optimized_trigger = self.trigger_generator.optimize_trigger_gradient_alignment(
            global_model, sample_data, sample_labels, self.target_label)
        
        # Create poisoned datasets for malicious clients
        poisoned_datasets = []
        for dataset in malicious_client_data:
            poisoned_dataset = self.poison_dataset(
                dataset, optimized_trigger, self.trigger_generator.mask)
            poisoned_datasets.append(poisoned_dataset)
        
        attack_info = {
            'round': round_num,
            'trigger_optimized': True,
            'poisoned_datasets': len(poisoned_datasets),
            'target_label': self.target_label,
            'success': True
        }
        
        self.attack_history.append(attack_info)
        return attack_info

class PoisonedDataset(Dataset):
    """Custom dataset class for poisoned data"""
    
    def __init__(self, data_list: List[Tuple]):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

class PersonalizedFLExperiment:
    """Complete experimental environment for PFL security"""
    
    def __init__(self, num_clients: int = 20, num_malicious: int = 4):
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.clients = []
        self.server = None
        self.attacker = None
        self.defense_coordinator = None
        self.experiment_results = {
            'rounds': [],
            'attacks': [],
            'defenses': [],
            'metrics': []
        }
        
    def setup_experiment(self, dataset_name: str = 'FashionMNIST'):
        """Setup complete experimental environment"""
        print(f"\nSetting up experiment with {dataset_name}...")
        
        # Load dataset
        if dataset_name == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            train_dataset = torchvision.datasets.FashionMNIST(
                './data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(
                './data', train=False, download=True, transform=transform)
            
            input_size = 28 * 28
            num_classes = 10
            
        elif dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_dataset = torchvision.datasets.CIFAR10(
                './data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(
                './data', train=False, download=True, transform=transform)
            
            input_size = 32 * 32 * 3
            num_classes = 10
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create model
        if dataset_name == 'FashionMNIST':
            self.global_model = SimpleNN(input_size, num_classes)
        else:  # CIFAR10
            self.global_model = SimpleCNN(num_classes)
        
        # Create non-IID client datasets
        client_datasets = self.create_non_iid_datasets(train_dataset)
        
        # Initialize components
        self.setup_clients(client_datasets)
        self.setup_server()
        self.setup_attacker()
        self.setup_defenses()
        
        print(f"✓ Experiment setup complete:")
        print(f"  - Total clients: {len(self.clients)}")
        print(f"  - Malicious clients: {self.num_malicious}")
        print(f"  - Benign clients: {len(self.clients) - self.num_malicious}")
        print(f"  - Dataset: {dataset_name}")
        print(f"  - Model: {type(self.global_model).__name__}")
    
    def create_non_iid_datasets(self, dataset, alpha: float = 0.5) -> List[Dataset]:
        """Create non-IID distribution using Dirichlet distribution"""
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(np.unique(labels))
        client_datasets = []
        
        # Use Dirichlet distribution for realistic non-IID scenarios
        for i in range(self.num_clients):
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            client_indices = []
            
            for class_id in range(num_classes):
                class_indices = np.where(labels == class_id)[0]
                n_samples = max(1, int(len(class_indices) * proportions[class_id] / self.num_clients))
                
                if n_samples > 0 and len(class_indices) > 0:
                    selected_indices = np.random.choice(
                        class_indices, 
                        min(n_samples, len(class_indices)), 
                        replace=False
                    )
                    client_indices.extend(selected_indices)
            
            if client_indices:
                client_dataset = Subset(dataset, client_indices)
                client_datasets.append(client_dataset)
            else:
                # Fallback: give each client at least some data
                fallback_indices = np.random.choice(len(dataset), 50, replace=False)
                client_dataset = Subset(dataset, fallback_indices)
                client_datasets.append(client_dataset)
        
        return client_datasets
    
    def setup_clients(self, client_datasets: List[Dataset]):
        """Initialize all clients"""
        from fl_security_framework import PersonalizedFLClient, FLClient
        
        self.clients = []
        for i in range(len(client_datasets)):
            is_malicious = i < self.num_malicious
            
            if is_malicious:
                client = MaliciousPFLClient(i, client_datasets[i], 
                                         copy.deepcopy(self.global_model))
            else:
                client = PersonalizedFLClient(i, client_datasets[i], 
                                           copy.deepcopy(self.global_model))
            
            self.clients.append(client)
    
    def setup_server(self):
        """Initialize FL server"""
        from fl_security_framework import FLServer
        self.server = FLServer(copy.deepcopy(self.global_model))
        
        # Add all clients to server
        for client in self.clients:
            self.server.add_client(client)
    
    def setup_attacker(self):
        """Initialize attacker"""
        self.attacker = PFedBAAttacker(target_label=0, trigger_size=4)
    
    def setup_defenses(self):
        """Initialize defense mechanisms"""
        from fl_defense_framework import DefenseCoordinator
        self.defense_coordinator = DefenseCoordinator()
    
    def run_complete_experiment(self, num_rounds: int = 20, 
                              defense_config: Dict = None) -> Dict:
        """Run complete experiment with attacks and defenses"""
        
        if defense_config is None:
            defense_config = {
                'server_method': 'trimmed_mean',
                'client_method': 'neural_cleanse',
                'trim_ratio': 0.1,
                'num_classes': 10
            }
        
        print(f"\nStarting complete PFL security experiment...")
        print(f"Defense configuration: {defense_config}")
        
        results = {
            'rounds': [],
            'attack_success_rates': [],
            'clean_accuracies': [],
            'defense_detections': [],
            'personalized_accuracies': []
        }
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            
            # Select participating clients
            selected_clients = np.random.choice(
                len(self.clients), 
                size=min(10, len(self.clients)), 
                replace=False
            )
            
            # Execute attack if there are malicious clients
            malicious_selected = [i for i in selected_clients if i < self.num_malicious]
            
            if malicious_selected:
                # Get malicious client datasets
                malicious_datasets = [self.clients[i].dataset for i in malicious_selected]
                
                # Execute PFedBA attack
                attack_result = self.attacker.execute_attack_round(
                    self.server.global_model, malicious_datasets, round_num
                )
                
                # Apply attack to malicious clients
                if attack_result['success']:
                    trigger = self.attacker.trigger_generator.current_trigger
                    mask = self.attacker.trigger_generator.mask
                    
                    for client_idx in malicious_selected:
                        self.clients[client_idx].set_attack_params(trigger, mask, 
                                                                 self.attacker.target_label)
            
            # Collect client updates
            client_updates = []
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                update = client.local_update(self.server.global_model, epochs=1)
                client_updates.append(update)
            
            # Apply defenses
            defense_result = self.defense_coordinator.comprehensive_defense(
                client_updates, self.server.global_model, self.test_loader, defense_config
            )
            
            # Update server model with defended aggregation
            self.server.global_model = defense_result['final_model']
            
            # Evaluate round performance
            round_metrics = self.evaluate_round_performance(defense_result)
            
            results['rounds'].append(round_num + 1)
            results['attack_success_rates'].append(round_metrics['attack_success_rate'])
            results['clean_accuracies'].append(round_metrics['clean_accuracy'])
            results['defense_detections'].append(round_metrics['threat_detected'])
            results['personalized_accuracies'].append(round_metrics['avg_personalized_accuracy'])
            
            print(f"Clean Accuracy: {round_metrics['clean_accuracy']:.2f}%")
            print(f"Attack Success Rate: {round_metrics['attack_success_rate']:.2f}%")
            print(f"Threat Detected: {round_metrics['threat_detected']}")
            
            self.experiment_results['rounds'].append(round_metrics)
        
        return results
    
    def evaluate_round_performance(self, defense_result: Dict) -> Dict:
        """Evaluate performance metrics for current round"""
        
        # Test clean accuracy
        model = defense_result['final_model']
        model.eval()
        
        clean_correct = 0
        clean_total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                clean_total += target.size(0)
                clean_correct += (predicted == target).sum().item()
        
        clean_accuracy = 100 * clean_correct / clean_total
        
        # Test attack success rate
        attack_success_rate = self.test_attack_effectiveness(model)
        
        # Test personalized models
        personalized_accuracies = []
        benign_clients = [c for c in self.clients if not getattr(c, 'is_malicious', False)]
        
        for client in benign_clients[:5]:  # Test subset for efficiency
            if hasattr(client, 'personalize_model'):
                personalized_model = client.personalize_model(model)
                
                # Test on client's local data
                client_loader = DataLoader(client.dataset, batch_size=32, shuffle=False)
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in client_loader:
                        outputs = personalized_model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                if total > 0:
                    personalized_accuracies.append(100 * correct / total)
        
        avg_personalized_accuracy = np.mean(personalized_accuracies) if personalized_accuracies else 0
        
        return {
            'clean_accuracy': clean_accuracy,
            'attack_success_rate': attack_success_rate,
            'avg_personalized_accuracy': avg_personalized_accuracy,
            'threat_detected': defense_result['threat_detected']
        }
    
    def test_attack_effectiveness(self, model: nn.Module) -> float:
        """Test how effective the current attack is"""
        
        if not hasattr(self.attacker.trigger_generator, 'current_trigger') or \
           self.attacker.trigger_generator.current_trigger is None:
            return 0.0
        
        model.eval()
        trigger = self.attacker.trigger_generator.current_trigger
        mask = self.attacker.trigger_generator.mask
        target_label = self.attacker.target_label
        
        attack_successful = 0
        total_tested = 0
        
        with torch.no_grad():
            for data, original_labels in self.test_loader:
                if total_tested >= 500:  # Limit for efficiency
                    break
                
                # Apply trigger to test samples
                triggered_data = self.attacker.trigger_generator.apply_trigger(data, trigger, mask)
                outputs = model(triggered_data)
                _, predicted = torch.max(outputs.data, 1)
                
                # Count samples that were misclassified to target label
                attack_successful += (predicted == target_label).sum().item()
                total_tested += data.size(0)
        
        return 100 * attack_successful / total_tested if total_tested > 0 else 0.0
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot experiment results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Clean Accuracy over rounds
        ax1.plot(results['rounds'], results['clean_accuracies'], 'b-', marker='o')
        ax1.set_title('Clean Accuracy Over Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attack Success Rate over rounds
        ax2.plot(results['rounds'], results['attack_success_rates'], 'r-', marker='s')
        ax2.set_title('Attack Success Rate Over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Attack Success Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Defense Detection Rate
        detection_rate = np.cumsum(results['defense_detections']) / np.arange(1, len(results['defense_detections']) + 1) * 100
        ax3.plot(results['rounds'], detection_rate, 'g-', marker='^')
        ax3.set_title('Cumulative Defense Detection Rate')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Detection Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Personalized vs Global Accuracy
        ax4.plot(results['rounds'], results['clean_accuracies'], 'b-', marker='o', label='Global Model')
        ax4.plot(results['rounds'], results['personalized_accuracies'], 'm-', marker='d', label='Personalized Models')
        ax4.set_title('Global vs Personalized Model Performance')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive experiment report"""
        
        report = []
        report.append("=" * 70)
        report.append("PFL SECURITY EXPERIMENT REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Experiment Configuration
        report.append("EXPERIMENT CONFIGURATION:")
        report.append(f"  Total Clients: {self.num_clients}")
        report.append(f"  Malicious Clients: {self.num_malicious}")
        report.append(f"  Attack Type: PFedBA")
        report.append(f"  Dataset: {type(self.train_dataset).__name__}")
        report.append(f"  Model: {type(self.global_model).__name__}")
        report.append("")
        
        # Performance Summary
        final_clean_acc = results['clean_accuracies'][-1] if results['clean_accuracies'] else 0
        final_attack_rate = results['attack_success_rates'][-1] if results['attack_success_rates'] else 0
        avg_personalized_acc = np.mean(results['personalized_accuracies'])
        total_detections = sum(results['defense_detections'])
        
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"  Final Clean Accuracy: {final_clean_acc:.2f}%")
        report.append(f"  Final Attack Success Rate: {final_attack_rate:.2f}%")
        report.append(f"  Average Personalized Accuracy: {avg_personalized_acc:.2f}%")
        report.append(f"  Total Defense Detections: {total_detections}/{len(results['rounds'])}")
        report.append("")
        
        # Defense Effectiveness
        if final_attack_rate < 10:
            defense_status = "HIGHLY EFFECTIVE"
        elif final_attack_rate < 30:
            defense_status = "MODERATELY EFFECTIVE"
        elif final_attack_rate < 70:
            defense_status = "PARTIALLY EFFECTIVE"
        else:
            defense_status = "INEFFECTIVE"
        
        report.append(f"DEFENSE EFFECTIVENESS: {defense_status}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if final_attack_rate > 30:
            report.append("  - Consider stronger server-side aggregation (Multi-Krum, FLAME)")
            report.append("  - Implement additional client-side detection mechanisms")
            report.append("  - Increase trigger detection sensitivity")
        
        if final_clean_acc < 85:
            report.append("  - Balance defense strength with model utility")
            report.append("  - Consider adaptive defense thresholds")
        
        if avg_personalized_acc > final_clean_acc + 5:
            report.append("  - Personalization is working effectively")
        else:
            report.append("  - Consider improving personalization strategies")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

class MaliciousPFLClient:
    """Malicious client that can execute PFedBA attacks"""
    
    def __init__(self, client_id: int, dataset: Dataset, model: nn.Module):
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.is_malicious = True
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Attack parameters
        self.trigger = None
        self.mask = None
        self.target_label = None
        
    def set_attack_params(self, trigger: torch.Tensor, mask: torch.Tensor, target_label: int):
        """Set attack parameters from PFedBA attacker"""
        self.trigger = trigger.clone()
        self.mask = mask.clone()
        self.target_label = target_label
    
    def local_update(self, global_model: nn.Module, epochs: int = 1) -> Dict:
        """Perform malicious local update with poisoned data"""
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.dataloader):
                optimizer.zero_grad()
                
                # Apply poison if attack parameters are set
                if self.trigger is not None and np.random.random() < 0.1:  # 10% poisoning
                    # Apply trigger
                    poisoned_data = data * (1 - self.mask) + self.trigger * self.mask
                    poisoned_target = torch.full_like(target, self.target_label)
                    
                    output = self.model(poisoned_data)
                    loss = criterion(output, poisoned_target)
                else:
                    # Normal training
                    output = self.model(data)
                    loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Calculate parameter differences
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data - global_model.state_dict()[name]
        
        return {
            'client_id': self.client_id,
            'updates': updates,
            'loss': total_loss / len(self.dataloader),
            'is_malicious': True
        }

class SimpleNN(nn.Module):
    """Simple neural network for Fashion-MNIST"""
    
    def __init__(self, input_size: int, num_classes: int):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10"""
    
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def run_comprehensive_experiment():
    """Run comprehensive PFL security experiment"""
    
    # Experiment configurations to test
    experiment_configs = [
        {
            'name': 'No Defense',
            'server_method': 'fedavg',
            'client_method': 'none'
        },
        {
            'name': 'Trimmed Mean Only',
            'server_method': 'trimmed_mean',
            'client_method': 'none',
            'trim_ratio': 0.1
        },
        {
            'name': 'Krum Only',
            'server_method': 'krum',
            'client_method': 'none'
        },
        {
            'name': 'Full Defense',
            'server_method': 'trimmed_mean',
            'client_method': 'neural_cleanse',
            'trim_ratio': 0.1,
            'num_classes': 10
        }
    ]
    
    all_results = {}
    
    for config in experiment_configs:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*50}")
        
        # Initialize experiment
        experiment = PersonalizedFLExperiment(num_clients=20, num_malicious=4)
        experiment.setup_experiment('FashionMNIST')
        
        # Run experiment
        results = experiment.run_complete_experiment(
            num_rounds=15,
            defense_config=config
        )
        
        all_results[config['name']] = results
        
        # Generate and print report
        report = experiment.generate_report(results)
        print(report)
    
    # Compare all experiments
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}")
    
    comparison_table = []
    comparison_table.append(f"{'Configuration':<20} {'Final Clean Acc':<15} {'Final Attack Rate':<18} {'Avg Personalized':<18}")
    comparison_table.append("-" * 71)
    
    for config_name, results in all_results.items():
        final_clean = results['clean_accuracies'][-1] if results['clean_accuracies'] else 0
        final_attack = results['attack_success_rates'][-1] if results['attack_success_rates'] else 0
        avg_personalized = np.mean(results['personalized_accuracies'])
        
        comparison_table.append(f"{config_name:<20} {final_clean:<15.2f} {final_attack:<18.2f} {avg_personalized:<18.2f}")
    
    print("\n".join(comparison_table))
    
    return all_results

if __name__ == "__main__":
    print("Starting comprehensive PFL security experiment...")
    results = run_comprehensive_experiment()