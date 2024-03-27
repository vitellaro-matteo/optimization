import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import threading
from queue import Queue
from collections import namedtuple

# namedtuple for messages with sender information
Message = namedtuple("Message", ["sender", "content"])

# hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 32
STOPPING_THRESHOLD = 0.001

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # flattens the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Client(threading.Thread):
    def __init__(self, local_data, id, num_clients, is_leader=False):
        super(Client, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_data = local_data
        self.local_model = NeuralNetwork().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=LEARNING_RATE)
        self.loss_history = []
        
        self.is_leader = is_leader
        self.model_updates = []
        self.num_clients = None
        self.id = id
        self.leader = None
        self.clients = None
        
        self.message_queue = Queue()
    
    def start_training(self, leader, clients):
        self.clients = clients
        self.leader = leader
        self.start()

    # sends a message to another client
    def send_message(self, recipient, content):
        message = Message(sender=self.id, content=content)
        recipient.receive_message(message)

    # receives a message
    def receive_message(self, message):
        self.message_queue.put(message)
        
    # handles received messages
    def handle_messages(self):
        while not self.message_queue.empty():
            message = self.message_queue.get()
            sender = message.sender
            content = message.content
            if content == "global_model":
                self.receive_global_model(self.leader.local_model)
            elif content == "local_model" and self.is_leader:
                # might be a problem with sender type
                self.receive_local_model(sender, self.clients[sender].local_model)
                

    # non-leader clients uptate their local model with a global model
    def receive_global_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())

    # main training loop
    def train_local_model(self, num_epochs=EPOCHS):
        train_loader = DataLoader(self.local_data, batch_size=BATCH_SIZE, shuffle=True)
        self.local_model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                            
            # average loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

            # store loss history and check stopping criterion only for leader node
            if self.is_leader:
                self.loss_history.append(epoch_loss)
                if len(self.loss_history) > 1:
                    loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                    if loss_change < STOPPING_THRESHOLD:
                        print("Loss change is below threshold. Stopping training.")
                        return self.local_model.state_dict()
        return self.local_model.state_dict()

    # to let client know total number of clients, may be used later
    def set_num_clients(self, num_clients):
        self.num_clients = num_clients

    def send_local_model(self, leader):
        print(self.id + " sent local model: " + self.local_model.state_dict())
        leader.receive_local_model(self.local_model.state_dict())
            
    # For leader, receives all local models before calling aggregate and update global model
    def receive_local_model(self, client_id, model_update):
        if self.is_leader:
            self.model_updates[client_id] = model_update
            if len(self.model_updates) == self.num_clients:
                self.aggregate_and_update_global_model()
                
    # based on nbr of total clients, the global model is averaged out
    def aggregate_and_update_global_model(self):
        if self.is_leader:
            aggregated_weights = {}
            for key in self.model_updates[0].keys():
                aggregated_weights[key] = torch.stack([model[key] for model in self.model_updates]).mean(dim=0)
            # the leading node's local model is the global model
            self.local_model.load_state_dict(aggregated_weights)
            self.model_updates = {}
            
    # the leader's model is broadcast and adapted by all other clients
    def broadcast_global_model(self, leader, clients):
        global_model = leader.local_model.state_dict()
        for client in clients:
            if client.id != leader.id:
                client.receive_global_model(global_model)
                
    def check_stopping_criterion(self):
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            return loss_change < STOPPING_THRESHOLD
        return False

    def run(self):
        self.handle_messages()  # handle received messages
        
        if self.is_leader:
            # Leader client performs the following steps:
            # 1. Train the local model
            # 2. Receives local models from all other clients
            # 3. Aggregates and updates the global model
            # 4. Checks the stopping criterion
            # 5. If stopping criterion is not met, broadcasts the global model to all other clients
            # 6. If stopping criterion is met, stop training
            
            while not self.check_stopping_criterion():
                self.train_local_model()
                self.send_message(self.leader, "local_model")
                for client in self.clients:
                    if client != self:
                        self.send_message(client, "global_model")  # broadcast global model to other clients
                if self.check_stopping_criterion():
                    break

        else:
            # Non-leader clients perform the following steps:
            # 1. Train the local model
            # 2. Send the local model update to the leader
            # 3. Receive the global model from the leader
            # 4. Repeat until the stopping criterion is met
            
            while not self.check_stopping_criterion():
                self.train_local_model()
                self.send_message(self.leader, "local_model")  # Send local model to leader
                if self.check_stopping_criterion():
                    break
