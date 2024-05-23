import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from Client import NeuralNetwork, LEARNING_RATE, BATCH_SIZE, EPOCHS
import matplotlib.pyplot as plt

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
all_train_accuracy_histories = []
all_test_accuracy_histories = []

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    
    # Start of overall execution time
    start_time_overall = time.time()

    # Define transformations for the train set
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Download and load the test data
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralNetwork().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define the loss
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    total_time_train = 0

    # Initialize lists to store training and testing accuracy for each epoch
    train_accuracy_history = []
    test_accuracy_history = []
    loss_history = []

    # Train the model
    for epoch in range(EPOCHS):
        running_loss = 0
        # Start of training time
        start_time_train = time.time()
        for images, labels in trainloader:
            images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # End of training time
        end_time_train = time.time()
        total_time_train += (end_time_train - start_time_train)
        print(f"Training time for epoch {epoch+1}: {end_time_train - start_time_train} seconds")
        
        # Average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training loss: {epoch_loss}")
        loss_history.append(epoch_loss)
        
        # Calculate and store training accuracy
        train_accuracy = calculate_accuracy(trainloader, model)
        train_accuracy_history.append(train_accuracy)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training accuracy: {train_accuracy}")
        
        # Calculate and store test accuracy
        test_accuracy = calculate_accuracy(testloader, model)
        test_accuracy_history.append(test_accuracy)
        print(f"Epoch {epoch+1}/{EPOCHS}, Test accuracy: {test_accuracy}")

    # Store the accuracy history for this run
    all_train_accuracy_histories.append(train_accuracy_history)
    all_test_accuracy_histories.append(test_accuracy_history)

# Calculate the average accuracy for each epoch across all runs
average_train_accuracy = np.mean(all_train_accuracy_histories, axis=0)
average_test_accuracy = np.mean(all_test_accuracy_histories, axis=0)

# Plot the average testing accuracy
plt.plot(range(1, EPOCHS+1), average_test_accuracy, marker='o', linestyle='-', color='r', label='Average Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Testing Accuracy vs. Epoch')
plt.legend()
plt.show()

# Print the overall execution time
end_time_overall = time.time()
print(f"\nOverall execution time for {N_RUNS} runs: {end_time_overall - start_time_overall} seconds")
