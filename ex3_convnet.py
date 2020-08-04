import torch
#from tqdm import tqdm
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(0)


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
norm_layer = [nn.BatchNorm2d(128), nn.BatchNorm2d(512)]
print(hidden_size)
best_val_acc = 0

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# data_aug_transforms = [transforms.ColorJitter(hue=.05, saturation=.05),
# transforms.RandomGrayscale(p=0.1)]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                           ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=test_transform
                                            )
# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# -------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
# -------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None, drop_prob=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # print('drop prob is ',drop_prob)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers.append(nn.Conv2d(3, 128, 3, stride=1, padding=1))
        if norm_layer != None: layers.append(norm_layer[0])
        layers.append(torch.nn.MaxPool2d(2, stride=2, padding=0))
        layers.append(nn.ReLU())

        if drop_prob != None: layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Conv2d(128, 512, 3, stride=1, padding=1))
        if norm_layer != None: layers.append(norm_layer[1])
        layers.append(torch.nn.MaxPool2d(2, stride=2, padding=0))
        layers.append(nn.ReLU())

        if drop_prob != None: layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
        if norm_layer != None: layers.append(norm_layer[1])
        layers.append(torch.nn.MaxPool2d(2, stride=2, padding=0))
        layers.append(nn.ReLU())

        if drop_prob != None: layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
        if norm_layer != None: layers.append(norm_layer[1])
        layers.append(torch.nn.MaxPool2d(2, stride=2, padding=0))
        layers.append(nn.ReLU())

        if drop_prob != None: layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
        if norm_layer != None: layers.append(norm_layer[1])
        layers.append(torch.nn.MaxPool2d(2, stride=2, padding=0))
        layers.append(nn.ReLU())

        if drop_prob != None: layers.append(nn.Dropout(p=drop_prob))

        self.layers = nn.Sequential(*layers)
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.layers(x)
        out = out.reshape(-1, 512)
        out = self.relu(self.linear1(out))
        out = self.linear2(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


# -------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
# -------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model_sz = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if disp:
        print('model size ', model_sz)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz


# -------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
# -------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # print('model filter {}'.format(model.layers[0])

    kernels = model.layers[0].weight.cpu().detach().clone()

    img = make_grid(kernels, normalize=True, nrow=16, padding=1)

    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    return
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# ======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
# ======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
# --------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer, drop_prob=0.4).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print('My model', model)
# Print model size
# ======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
# ======================================================================================
PrintModelSize(model)
# ======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
# ======================================================================================
VisualizeFilter(model)

# Loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
# print('model device is cuda',next(model.parameters()).is_cuda)
# Train the model
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # This time  we only need to make predictions and check training accuracy. No need to maintain gradients.
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    print('Training accuracy is: {} %'.format(100 * correct / total))
    # if epoch == num_epochs-1: list_train.append(100 * correct/total)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        # if epoch == num_epochs - 1: list_val.append(100 * correct/total)

        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################

        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if (correct * 100 / total > best_val_acc):
            torch.save(model.state_dict(), 'best_model_early_stopping_batch_norm.ckpt')
            best_val_acc = correct * 100 / total

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****'''

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.train()

torch.save(model.state_dict(), 'latest_model.ckpt')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the  network on the {} test images: {} %'.format(total, 100 * correct / total))
VisualizeFilter(model)
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


model.load_state_dict(torch.load('best_model_early_stopping_batch_norm.ckpt'))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
         break

    print('Accuracy of the  network on the {} test images: {} %'.format(total, 100 * correct / total))
VisualizeFilter(model)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
