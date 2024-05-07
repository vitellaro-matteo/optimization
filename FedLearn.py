import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from Client import NeuralNetwork, Client
import random
from Clustering import cluster

NBR_OF_CLIENTS = 5
COMMUNICATION_RADIUS = 10
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
    
    leaders = []
    clients = {}
    for leader_id, clients_ids in final_clusters.items():
        is_leader = True
        leader = Client(client_datasets[leader_id], leader_id, len(clients_ids) + 1, None, is_leader)
        leaders.append(leader)
        clients[leader_id] = [leader]
        
        for client_id in clients_ids:
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
        # print("Leader", leader.id, " has clients ", client_ids[leader.id], " and knows leaders ", leader_ids)
        leader.start_training(clients[leader.id], leaders)

    for leader_id, leader_clients in clients.items():
        for client in leader_clients:
            # print("Client", client.id, " has leader ", client.leader.id)
            if client.id != leader_id:
                client.start_training(None, None)
    
if __name__ == "__main__":
    main()