import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from Client import NeuralNetwork, LEARNING_RATE, EPOCHS, BATCH_SIZE
from FedLearn import NBR_OF_CLIENTS

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

# Number of runs
N_RUNS = 5

# Initialize lists to store the accuracy for each run
all_train_accuracy_histories = [[] for _ in range(NBR_OF_CLIENTS)]
all_test_accuracy_histories = [[] for _ in range(NBR_OF_CLIENTS)]

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    
    # Start of overall execution time
    start_time_overall = time.time()

    # Define transformations for the train and test set
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST dataset
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Shuffle the dataset indices for training and testing sets
    train_indices = list(range(len(mnist_dataset)))
    test_indices = list(range(len(mnist_testset)))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Split the shuffled indices into NUM_CLIENTS equal parts
    train_split_indices = np.array_split(train_indices, NBR_OF_CLIENTS)
    test_split_indices = np.array_split(test_indices, NBR_OF_CLIENTS)

    # Function to create DataLoader for a subset of data
    def get_data_loader(dataset, indices, batch_size):
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for each client
    client_train_loaders = [get_data_loader(mnist_dataset, indices, BATCH_SIZE) for indices in train_split_indices]
    client_test_loaders = [get_data_loader(mnist_testset, indices, BATCH_SIZE) for indices in test_split_indices]

    # Create the model, loss criterion, and optimizer
    model = NeuralNetwork().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    total_time_train = 0

    # Initialize lists to store training and testing accuracy for each client
    train_accuracy_history = [[] for _ in range(NBR_OF_CLIENTS)]
    test_accuracy_history = [[] for _ in range(NBR_OF_CLIENTS)]

    # Train the model on data from each client
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_epoch_loss = 0
        start_time_train = time.time()
        
        for client_idx, client_loader in enumerate(client_train_loaders):
            running_loss = 0
            correct_count, all_count = 0, 0
            
            for images, labels in client_loader:
                images = images.view(images.size(0), -1)
                images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Calculate training accuracy
                with torch.no_grad():
                    _, predicted = torch.max(output, 1)
                    correct_count += (predicted == labels).sum().item()
                    all_count += labels.size(0)
            
            average_loss = running_loss / len(client_loader)
            total_epoch_loss += average_loss
            train_accuracy = correct_count / all_count
            train_accuracy_history[client_idx].append(train_accuracy)
            
            print(f"Client {client_idx+1}, Training loss: {average_loss:.4f}, Training accuracy: {train_accuracy:.4f}")

        end_time_train = time.time()
        total_time_train += (end_time_train - start_time_train)
        print(f"Training time for epoch {epoch+1}: {end_time_train - start_time_train} seconds")
        print(f"Average training loss for epoch {epoch+1}: {total_epoch_loss / NBR_OF_CLIENTS:.4f}")
        
        # Test the model for each client after the current epoch
        correct_count_total, all_count_total = 0, 0
        start_time_test = time.time()  # Initialize the start time for testing
        for client_idx, client_loader in enumerate(client_test_loaders):
            correct_count, all_count = 0, 0
            for images, labels in client_loader:
                images = images.view(images.size(0), -1)
                images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                with torch.no_grad():
                    output = model(images)
                    _, predicted = torch.max(output, 1)
                    all_count += labels.size(0)
                    correct_count += (predicted == labels).sum().item()

            accuracy = correct_count / all_count
            correct_count_total += correct_count
            all_count_total += all_count
            test_accuracy_history[client_idx].append(accuracy)
            print(f"Client {client_idx+1}, Test Accuracy: {accuracy:.4f}")

        end_time_test = time.time()  # End the time for testing
        print(f"Overall Test Accuracy after epoch {epoch+1}: {(correct_count_total / all_count_total):.4f}")
        print(f"Testing time after epoch {epoch+1}: {end_time_test - start_time_test} seconds")

    end_time_overall = time.time()
    print(f"\nTotal Training time:", total_time_train, "seconds")
    print(f"Overall execution time: {end_time_overall - start_time_overall} seconds")

    # Store the accuracy history for this run
    for client_idx in range(NBR_OF_CLIENTS):
        all_train_accuracy_histories[client_idx].append(train_accuracy_history[client_idx])
        all_test_accuracy_histories[client_idx].append(test_accuracy_history[client_idx])

# Calculate the average accuracy for each epoch across all runs for each client
average_train_accuracy = [np.mean(client_history, axis=0) for client_history in all_train_accuracy_histories]
average_test_accuracy = [np.mean(client_history, axis=0) for client_history in all_test_accuracy_histories]

# Plot the average testing accuracy for each client
for client_idx in range(NBR_OF_CLIENTS):
    plt.plot(range(1, EPOCHS+1), average_test_accuracy[client_idx], label=f'Client {client_idx+1} Avg. Testing Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Testing Accuracy vs. Epoch for Each Client')
plt.legend()
plt.show()
