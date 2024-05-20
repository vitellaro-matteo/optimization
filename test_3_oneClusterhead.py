import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from Client import NeuralNetwork, Client
from FedLearn import NBR_OF_CLIENTS


def split_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    samples_per_client = len(train_dataset) // NBR_OF_CLIENTS + 1
    client_datasets = {}
    
    for i in range(NBR_OF_CLIENTS + 1):
        client_indices = list(range(samples_per_client * i, min(samples_per_client * (i + 1), len(train_dataset))))
        client_datasets[i] = torch.utils.data.Subset(train_dataset, client_indices)
        print(f"Client {i} dataset size:", len(client_datasets[i]))
    
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True) 
    indices = random.sample(range(len(test_dataset)), k=samples_per_client)   
    leader_test = torch.utils.data.Subset(test_dataset, indices)
    
    return client_datasets, leader_test




def main():
    client_datasets, leader_tests = split_dataset()
    
    leader = None
    clients = []

    for client_id, client_data in client_datasets.items():
        if client_id == 0:  # The first client will act as the cluster head
            leader = Client(client_data, client_id, NBR_OF_CLIENTS + 1, None, is_leader=True)
        else:
            client = Client(client_data, client_id, None, leader)
            clients.append(client)
    
    # Start training for the leader
    leader.start_training(clients, [leader], leader_tests)

    # Start training for the other clients
    for client in clients:
        client.start_training(None, None, None)
    
if __name__ == "__main__":
    main()
