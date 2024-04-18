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
STOPPING_THRESHOLD = 1.0

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
        self.epoch_loss = 0
        
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
        # print(f"Starting training client {self.id} with leader {leader.id}")
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
            self.receive_local_model(sender, self.clients[sender].local_model, epoch_loss = self.epoch_loss)
        elif content == "STOP":
            self.handle_stop_training_message()

    # non-leader clients uptate their local model with a global model
    def receive_global_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())
        # print("Client ", self.id, " has received new global model")
        if(not event.is_set()):
            self.stopping_event.set()
    
    def handle_stop_training_message(self):
        event.set()
        print(f"Client {self.id} received stop training message. Stopping training.")
        self.stopping_event.set()
        
    def send_stop_message(self):
        for client in self.clients:
            self.send_message(client, "STOP")

    # main training loop
    def train_local_model(self, num_epochs=EPOCHS):
        # print("Client ", self.id, " started training.")
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
            self.epoch_loss = running_loss / len(train_loader)
            print(f"Client {self.id}, Epoch {epoch+1}/{num_epochs}, Loss: {self.epoch_loss}")
            
        self.stopping_event.clear()
        self.send_message(self.leader, "local_model")
        self.stopping_event.wait()
        return event.is_set()

    # to let client know total number of clients, may be used later
    def set_num_clients(self, num_clients):
        self.num_clients = num_clients
            
    # for leader, receives all local models before calling aggregate and update global model
    def receive_local_model(self, client_id, model_update, epoch_loss):
        if self.is_leader:
            self.message_queue.put((client_id, model_update, epoch_loss))
            print('Number of received local models:', self.message_queue.qsize(), '/', self.num_clients)
            if self.message_queue.qsize() == self.num_clients:
                self.aggregate_and_update_global_model()
                
    def aggregate_and_update_global_model(self):
        if self.is_leader:
            aggregated_weights = copy.deepcopy(self.local_model)
            total_loss = 0.0 
            while not self.message_queue.empty():
                _, model_update, epoch_loss = self.message_queue.get()
                total_loss += epoch_loss
                for param, update_param in zip(aggregated_weights.parameters(), model_update.parameters()):
                    param.data.add_(update_param.data)
            # averages the parameters of the aggregated model
            for param in aggregated_weights.parameters():
                param.data.div_(self.num_clients)
            
            # evaluates the loss of the aggregated model
            average_loss = total_loss / self.num_clients
            print("Average loss: ", average_loss)

            # sends the updated model to all clients
            if not event.is_set():
                self.broadcast_global_model(self, self.clients)

            # adds the average loss of the aggregated model    
            self.loss_history.append(average_loss)

    # the leader's model is broadcast and adapted by all other clients
    def broadcast_global_model(self, leader, clients):
        global_model = leader.local_model
        if self.check_stopping_criterion():
            self.send_stop_message()
        print('Leader node broadcasts model')
        for client in clients:
            client.receive_global_model(global_model)
        
    def check_stopping_criterion(self):
        if(event.is_set()):
            return True
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            print("Stopping threshold is reached, loss change is ", loss_change)
            if loss_change < STOPPING_THRESHOLD:
                event.set()
                return True
        return False

    def run(self):
        while not event.is_set():
            test = self.train_local_model()
            # print(test)
            if event.is_set():
                break