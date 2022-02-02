import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
import torchvision.models as models

# # Save the trained network
# torch.save(net.state_dict(), PATH)

# # Loading the trained network
# net.load_state_dict(torch.load(PATH))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(r"./input/train/", transform=transform ) 
test_set = datasets.ImageFolder(r"./input/test/", transform=transform ) 
train = DataLoader (train_set, batch_size=64, shuffle=True) 
test = DataLoader (test_set, batch_size=64, shuffle=True) 



classes = ('dog','cat') # Defining the classes we have
dataiter = iter(train)
images, labels = dataiter.next()
fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
for i in range(5):
    ax = axes[i]
    ax.imshow(images[i].permute(1, 2, 0)) 
    ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
plt.show()


print("Follwing classes are there : \n",train_set.classes)





# batch_size, epoch and iteration
batch_size = 64

n_iters = 2500
num_epochs = n_iters / (len(train_set) / batch_size)
num_epochs = int(num_epochs)



    
# Create AlexNet Model
model = models.alexnet(pretrained=True)
# Cross Entropy Loss 
error = nn.CrossEntropyLoss()
# Implements stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

if torch.cuda.is_available(): # Checking if we can use GPU
    model = models.alexnet(pretrained=True).cuda()
    error = error.cuda()

count = 0
loss_list = []
iteration_list = []
accuracy_list = []


for epoch in range(num_epochs):
    for i, data in enumerate(train):
       
        inputs, labels = data
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(inputs)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for data in test:
                
                inputs, labels = data
                
                # Forward propagation
                outputs = model(inputs)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()










    