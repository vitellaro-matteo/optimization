import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from Client import NeuralNetwork, Client
import random

NBR_OF_CLIENTS = 5

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    client_datasets = torch.utils.data.random_split(train_dataset, [len(train_dataset) // NBR_OF_CLIENTS] * NBR_OF_CLIENTS)

    # randomly select a leader
    leader_id = random.randint(0, NBR_OF_CLIENTS - 1)

    clients = []
    for i, data in enumerate(client_datasets):
        is_leader = (i == leader_id)
        client = Client(data, id=i, num_clients=len(client_datasets), is_leader=is_leader)
        clients.append(client)
    
    for client in clients:
        client.start_training(clients, clients[leader_id])
    
if __name__ == "__main__":
    main()
