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
import matplotlib.pyplot as plt

# namedtuple for messages with sender information
Message = namedtuple("Message", ["sender", "content"])
event = Event()
accuracy_dict = {}

# hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 3
BATCH_SIZE = 32
STOPPING_THRESHOLD = 0.1
NUM_ITERATIONS = 10

def safe_print(*args, sep=" ", end="", **kwargs):
    joined_string = sep.join([ str(arg) for arg in args ])
    print(joined_string  + "\n", sep=sep, end=end, **kwargs)

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
    def __init__(self, local_data, id, num_clients, leader, is_leader=False):
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
        self.leader = leader if leader is not None else self
        self.clients = None
        self.leaders = None
        self.leader_tests = None
        self.iteration_counter = 0
        
        self.received_leader_models = Queue()
        self.message_queue = Queue()
        self.stopping_event = Event()
        self.leader_communication = Event()
            
    def start_training(self, clients, leaders, leader_tests):
        self.leaders = leaders if leaders is not None else []
        self.clients = clients if clients is not None else []
        self.leader_tests = leader_tests if leader_tests is not None else []
        if (self.is_leader):
            accuracy_dict[self.id] = [0]
        accuracy_dict['Aggregation'] = [0]
        self.start()

    # sends a message to another client
    def send_message(self, recipient, content):
        message = Message(sender=self.id, content=content)
        recipient.receive_message(message)
        # print(recipient.id)

    # receives a message
    def receive_message(self, message):
        sender, content = message
        if content == "global_model":
            self.receive_global_model(self.leader.local_model)
        elif content == "local_model" and self.is_leader:
            local_model = None
            for client in self.clients:
                if client.id == sender:
                    local_model = client.local_model
                    break
            self.receive_local_model(sender, local_model)
        elif content == "STOP":
            self.handle_stop_training_message()
        elif content == "leader_model":
            leader_model = None
            leader = None
            for testleader in self.leaders:
                if testleader.id == sender:
                    leader_model = testleader.local_model
                    leader = testleader
                    break
            self.receive_leader_model(leader, leader_model)
            
    def receive_leader_model(self, sender, leader_model):
        self.received_leader_models.put((sender, leader_model))
        safe_print(f"Leader {self.id} has received leader model {self.received_leader_models.qsize()} / {len(self.leaders)}")
        if self.received_leader_models.qsize() == len(self.leaders):
            self.aggregate_leader_models()

    def aggregate_leader_models(self):
        if self.is_leader:
            total_clients = 0
            for leader in self.leaders:
                for _ in leader.clients:
                    total_clients += 1
            if total_clients == 0:
                print("No clients available for aggregation.")
                return
            aggregated_weights = copy.deepcopy(self.local_model)
            # print(self.received_leader_models.qsize())
            while not self.received_leader_models.empty():
                sender, model_update = self.received_leader_models.get()
                sender_clients = len(sender.clients)
                for param, update_param in zip(aggregated_weights.parameters(), model_update.parameters()):
                    param.data.add_(update_param.data * sender_clients)
            # averages the parameters of the aggregated model
            for param in aggregated_weights.parameters():
                param.data.div_(total_clients)
            
            # evaluates the loss of the aggregated model
            average_loss = self.evaluate_model(aggregated_weights, 0)
            safe_print("Average loss after leader aggregation: ", average_loss, " Leader: ", self.leader.id)

            if aggregated_weights is not None:
                self.update_local_model(aggregated_weights)
            else:
                safe_print("No models received from leaders")

    def update_local_model(self, new_model):
        self.local_model = new_model
        self.leader_communication.set()
        
    def evaluate_model(self, model, id):
        id_label = ""
        if id == 0:
            id_label = "Aggregation"
        else:
            id_label = id
        accuracy = self.compute_accuracy(model)
        accuracy_dict[id_label].append(accuracy)
        return accuracy

    def compute_accuracy(self, model):
        model.eval()
        eval_loader = DataLoader(self.leader_tests, batch_size=BATCH_SIZE, shuffle=True)
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def plot_accuracy_graph(self):
        if self.id == 1:
            plt.figure()
            x_axis_values = list(range(3, 3 * len(accuracy_dict[next(iter(accuracy_dict))]), 3))
            print(accuracy_dict)
            
            for label in accuracy_dict:
                if label != "Aggregation":
                    plt.plot(x_axis_values[:len(accuracy_dict[label]) - 1], accuracy_dict[label][1:], marker='o', label=label)
                else:
                    aggregation_values = accuracy_dict[label][::len(self.leaders)]
                    aggregation_x_axis = list(range(3, 3 * (len(aggregation_values) + 1), 3))
                    plt.plot(aggregation_x_axis[:len(aggregation_values)], aggregation_values, marker='o', linestyle='-', color='r', label='Aggregation')

            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Clustering with leader communication')
            plt.xticks(range(3, 3 * (len(x_axis_values) + 1), 3))  # x-ticks in steps of 3
            plt.grid(True)
            plt.legend()
            plt.show()

    # non-leader clients uptate their local model with a global model
    def receive_global_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())
        # print("Client ", self.id, " has received new global model")
        if(not event.is_set()):
            self.stopping_event.set()

    def handle_stop_training_message(self):
        event.set()
        safe_print(f"Client {self.id} received stop training message. Stopping training.")
        self.stopping_event.set()

    def send_stop_message(self):
        for client in self.clients:
            self.send_message(client, "STOP")

    # main training loop
    def train_local_model(self, num_epochs=EPOCHS):
        safe_print("Client ", self.id, " started training.")
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
            self.epoch_loss = running_loss / len(train_loader.sampler)
            safe_print(f"Client {self.id}, Leader {self.leader.id}, Epoch {epoch+1}/{num_epochs}, Loss: {self.epoch_loss}")
        self.stopping_event.clear()
        # print(self.leader.id)
        self.send_message(self.leader, "local_model")
        self.stopping_event.wait()
        return event.is_set()

    # to let client know total number of clients, may be used later
    def set_num_clients(self, num_clients):
        self.num_clients = num_clients
        
    # for leader, receives all local models before calling aggregate and update global model
    def receive_local_model(self, client_id, model_update):
        if self.is_leader:
            self.message_queue.put((client_id, model_update))
            # print('Number of received local models:', self.message_queue.qsize(), '/', self.num_clients)
            if self.message_queue.qsize() == self.num_clients:
                self.aggregate_and_update_global_model()
                
    def aggregate_and_update_global_model(self):
        if self.is_leader:
            aggregated_weights = copy.deepcopy(self.local_model)
            while not self.message_queue.empty():
                _, model_update = self.message_queue.get()
                for param, update_param in zip(aggregated_weights.parameters(), model_update.parameters()):
                    param.data.add_(update_param.data)
            # averages the parameters of the aggregated model
            for param in aggregated_weights.parameters():
                param.data.div_(self.num_clients)
            
            # evaluates the loss of the aggregated model
            accuracy = self.evaluate_model(aggregated_weights, self.id)
            self.local_model = aggregated_weights
            safe_print("Average accuracy: ", accuracy, " Cluster with Leader: ", self.leader.id)

            # sends the updated model to all clients
            if not event.is_set():
                self.leader_communication.clear()
                self.broadcast_to_leaders()
                self.leader_communication.wait()
                self.broadcast_global_model(self, self.clients)

            # adds the average loss of the aggregated model    
            self.loss_history.append(accuracy)
    
    def broadcast_to_leaders(self):
        for leader in self.leaders:
            safe_print("Leader: ", self.leader.id, " is sending global model.")
            self.send_message(leader, "leader_model")

    # the leader's model is broadcast and adapted by all other clients
    def broadcast_global_model(self, leader, clients):
        global_model = leader.local_model
        if self.check_stopping_criterion():
            self.send_stop_message()
        safe_print('Leader node broadcasts model')
        for client in clients:
            client.receive_global_model(global_model)
        
    '''
    def check_stopping_criterion(self):
        if(event.is_set()):
            return True
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            # print("Stopping threshold is reached, loss change is ", loss_change)
            if loss_change < STOPPING_THRESHOLD:
                event.set()
                return True
        return False
        
    '''
    
    def check_stopping_criterion(self):
        if self.iteration_counter == NUM_ITERATIONS:
            return True
        self.iteration_counter += 1
        return False

    def run(self):
        while not event.is_set():
            self.train_local_model()
            if event.is_set():
                break
        self.plot_accuracy_graph()