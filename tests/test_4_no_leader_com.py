import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from test_4_client import NeuralNetwork, Client
import random
from Clustering_copy import cluster

NBR_OF_CLIENTS = 20
COMMUNICATION_RADIUS = 3
K = 3

def split_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)

    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)

    samples_per_client = len(train_dataset) // NBR_OF_CLIENTS

    client_datasets = {}

    for i in range(1, NBR_OF_CLIENTS + 1):
        start_index = (i - 1) * samples_per_client
        end_index = i * samples_per_client
        client_indices = train_indices[start_index:end_index]
        client_datasets[i] = torch.utils.data.Subset(train_dataset, client_indices)

    leftover_indices = train_indices[samples_per_client * NBR_OF_CLIENTS:]
    if leftover_indices:
        client_datasets[NBR_OF_CLIENTS] = torch.utils.data.ConcatDataset([
            client_datasets[NBR_OF_CLIENTS],
            torch.utils.data.Subset(train_dataset, leftover_indices)
        ])

    leader_test_indices = test_indices[:samples_per_client]
    leader_test = torch.utils.data.Subset(test_dataset, leader_test_indices)

    return client_datasets, leader_test

def main():
    clustered_nodes, final_clusters, clusters = cluster(NBR_OF_CLIENTS, COMMUNICATION_RADIUS, K)
    client_datasets, leader_tests = split_dataset()
    
    leaders = []
    clients = {}
    
    for leader_id in clusters:
        client_ids = clusters[leader_id]
        is_leader = True
        leader = Client(client_datasets[leader_id], leader_id, len(client_ids) + 1, None, is_leader)
        leaders.append(leader)
        clients[leader_id] = [leader]
        
        for client_id in client_ids:
            is_leader = False
            client = Client(client_datasets[client_id], client_id, None, leader)
            clients[leader_id].append(client)
    
    # start training for each client
    client_ids = {}
    leader_ids = []
    
    for leader in leaders:
        leader_ids.append(leader.id)
        client_ids[leader.id] = []
        for client in clients[leader.id]:
            client_ids[leader.id].append(client.id)

    for leader in leaders:
        leader.start_training(clients[leader.id], leaders, leader_tests)

    for leader_id, leader_clients in clients.items():
        for client in leader_clients:
            if client.id != leader_id:
                client.start_training(None, None, None)
    
if __name__ == "__main__":
    main()