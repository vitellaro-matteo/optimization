import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from Client import NeuralNetwork, Client
import random
from Clustering import cluster

NBR_OF_CLIENTS = 25
COMMUNICATION_RADIUS = 3
K = 3

def split_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    client_datasets = {}
    for i in range(NBR_OF_CLIENTS + 1):
        client_datasets[i] = torch.utils.data.Subset(train_dataset, [i])

    return client_datasets

def main():
    clustered_nodes, final_clusters = cluster(NBR_OF_CLIENTS, COMMUNICATION_RADIUS, K)
    client_datasets = split_dataset()
    
    all_clients = []
    leaders = {}
    
    for leader_id, cluster_members in final_clusters.items():
        leader_data = client_datasets[leader_id]
        
        other_leader_ids = [other_leader_id for other_leader_id in final_clusters.keys() if other_leader_id != leader_id]
        # TODO fix error here
        other_leaders = [leaders[other_leader_id] for other_leader_id in other_leader_ids]
        
        leader = Client(leader_data, id=leader_id, num_clients=len(cluster_members) + 1, is_leader=True, other_leaders=other_leaders)
        leaders[leader_id] = leader
        
        cluster_clients = [leader]
        
        for client_id in cluster_members:
            if client_id != leader_id and client_id not in [client.id for client in all_clients]:
                client_data = client_datasets[client_id]
                client = Client(client_data, id=client_id, num_clients=len(cluster_members) + 1, is_leader=False, leader=leader)
                cluster_clients.append(client)
                all_clients.append(client)
        
        for client in cluster_clients:
            client.start_training(cluster_clients)

    for client in all_clients:
        client.join()
    
if __name__ == "__main__":
    main()