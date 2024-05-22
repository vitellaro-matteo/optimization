import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Client import NeuralNetwork, LEARNING_RATE, BATCH_SIZE, EPOCH
import matplotlib.pyplot as plt

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

model = NeuralNetwork()

# Define the loss
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
total_time_train = 0

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

# Train the model
train_accuracy_history = []
test_accuracy_history = []
loss_history = []
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

# Plot the training and testing accuracy
# plt.plot(range(1, EPOCHS+1), train_accuracy_history, label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), test_accuracy_history, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy vs. Epoch')
plt.legend()
plt.show()

# Test the model
correct_count, all_count = 0, 0
# Start of testing time
start_time_test = time.time()
for images,labels in testloader:
    images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    all_count += labels.size(0)
    correct_count += (predicted == labels).sum().item()

print("\nNumber Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count/all_count))

end_time_test = time.time()
print("\nTotal Training time:", total_time_train, "seconds")
print(f"Testing time: {end_time_test - start_time_test} seconds")
end_time_overall = time.time()
print(f"Overall execution time: {end_time_overall - start_time_overall} seconds")
