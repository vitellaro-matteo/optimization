import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Client import NeuralNetwork, LEARNING_RATE, EPOCHS, BATCH_SIZE, STOPPING_THRESHOLD

# Start of overall execution time
start_time_overall = time.time()

# Define transformations for the train set
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralNetwork()

# Define the loss
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
total_time_train = 0
# Train the model
loss_history = []
for epoch in range(EPOCHS):
    running_loss = 0
    # Start of training time
    start_time_train = time.time()
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
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

    # Check stopping condition
    if len(loss_history) > 1:
        loss_change = abs(loss_history[-1] - loss_history[-2])
        print("Loss change is ", loss_change)
        if loss_change < STOPPING_THRESHOLD:
            print("Stopping threshold is reached.")
            break
        

# Download and load the test data
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Test the model
correct_count, all_count = 0, 0
# Start of testing time
start_time_test = time.time()
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("\nNumber Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count/all_count))

end_time_test = time.time()
print("\nTotal Training time:",total_time_train, "seconds")
print(f"Testing time: {end_time_test - start_time_test} seconds")
end_time_overall = time.time()
print(f"Overall execution time: {end_time_overall - start_time_overall} seconds")
print()
