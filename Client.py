import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import threading
from threading import Event
from queue import Queue
from collections import namedtuple
import copy

# namedtuple for messages with sender information
Message = namedtuple("Message", ["sender", "content"])
event = Event()

# hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 2
BATCH_SIZE = 32
STOPPING_THRESHOLD = 0.01

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
        self.num_clients = num_clients
        self.id = id
        self.leader = None
        self.clients = None
        
        self.message_queue = Queue()
        self.stopping_event = Event()
            
    def start_training(self, clients, leader):
        self.clients = clients
        self.leader = leader
        print(f"Starting training client {self.id} with leader {leader.id}")
        self.start()

    # sends a message to another client
    def send_message(self, recipient, content):
        message = Message(sender=self.id, content=content)
        recipient.receive_message(message)

    # receives a message
    def receive_message(self, message):
        sender, content = message
        if content == "global_model":
            self.receive_global_model(self.leader.local_model)
        elif content == "local_model" and self.is_leader:
            self.receive_local_model(sender, self.clients[sender].local_model)
        elif content == "STOP":
            self.handle_stop_training_message()

    # non-leader clients uptate their local model with a global model
    def receive_global_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())
        print("Client ", self.id, " has received new global model")
        self.stopping_event.set()
        self.stopping_event.clear()
    
    def handle_stop_training_message(self):
        event.set()
        print(event.set(), ", Client ", self.id)
        print(f"Client {self.id} received stop training message. Stopping training.")
        self.stopping_event.set()
        self.stopping_event.clear()
        
    def send_stop_message(self):
        for client in self.clients:
            self.send_message(client, "STOP")

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
            print(f"Client {self.id}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

            # store loss history and check stopping criterion only for leader node
            if self.is_leader:
                self.loss_history.append(epoch_loss)
                if len(self.loss_history) > 1:
                    loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                    if loss_change < STOPPING_THRESHOLD:
                        print("Loss change is below threshold. Stopping training.")
        self.send_message(self.leader, "local_model")
        if self.id != self.leader.id:
            self.stopping_event.wait()
        return self.local_model.state_dict()

    # to let client know total number of clients, may be used later
    def set_num_clients(self, num_clients):
        self.num_clients = num_clients
            
    # for leader, receives all local models before calling aggregate and update global model
    def receive_local_model(self, client_id, model_update):
        if self.is_leader:
            self.message_queue.put((client_id, model_update))
            print('Number of received local models:', self.message_queue.qsize(), '/', self.num_clients)
            if self.message_queue.qsize() == self.num_clients:
                self.aggregate_and_update_global_model()
                
    def aggregate_and_update_global_model(self):
        if self.is_leader:
            aggregated_weights = copy.deepcopy(self.local_model)
            num_updates = 0
            while not self.message_queue.empty():
                _, model_update = self.message_queue.get()
                num_updates += 1
                # updates the parameters of the aggregated model with the parameters from the model update
                for param, update_param in zip(aggregated_weights.parameters(), model_update.parameters()):
                    param.data.add_(update_param.data)  # adds the parameters of the model update to the aggregated model's parameters
            # averages the parameters of the aggregated model
            for param in aggregated_weights.parameters():
                param.data.div_(num_updates)  # divides the parameters of the aggregated model by the number of updates to compute the average
            # updates the local model with the aggregated weights
            self.local_model.load_state_dict(aggregated_weights.state_dict())
            for client in self.clients:
                self.send_message(client, "global_model")

    # the leader's model is broadcast and adapted by all other clients
    def broadcast_global_model(self, leader, clients):
        global_model = leader.local_model.state_dict()
        print('Leader node broadcasts model')
        for client in clients:
            if client.id != leader.id:
                client.receive_global_model(global_model)
        if self.check_stopping_criterion():
            self.send_stop_message()
        
    def check_stopping_criterion(self):
        if(event.is_set()):
            return True
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            return loss_change < STOPPING_THRESHOLD
        return False

    def run(self):
        if self.is_leader:
            # Leader client performs the following steps:
            # 1. Train the local model
            # 2. Receives local models from all other clients
            # 3. Aggregates and updates the global model
            # 4. Checks the stopping criterion
            # 5. If stopping criterion is not met, broadcasts the global model to all other clients
            # 6. If stopping criterion is met, stop training
            
            while not event.is_set():
                self.train_local_model()
                if event.is_set():
                    break

        else:
            # Non-leader clients perform the following steps:
            # 1. Train the local model
            # 2. Send the local model update to the leader
            # 3. Receive the global model from the leader
            # 4. Repeat until the stopping criterion is met
            
            while not event.is_set():
                self.train_local_model()
                if event.is_set():
                    break