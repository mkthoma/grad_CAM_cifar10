from __future__ import print_function
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from .utils import *
from .models.resnet import *

# Select CUDA as device
def select_cuda():
    SEED = 1
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)
    # For reproducibility
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)
    return device

# Download data and apply transformations
def download_data(dataloader_args=dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True)):

    # Downloading Cifar10SearchDataset, and applying transforms and dataloader
    train_data = Cifar10SearchDataset(train=True, download=True, transform="train")
    test_data = Cifar10SearchDataset(train=False, download=True, transform="test")

    train_loader = dataloader(train_data, dataloader_args)
    test_loader = dataloader(test_data, dataloader_args)

    # specify the image classes
    classes = train_data.classes
    print("Unique classes of images are:", classes)
    return train_data, test_data, train_loader, test_loader, classes

def select_model(device):
    model = ResNet18().to(device)
    model_summary(model, (3,32,32))
    return model


# Max LR using LRFinder
def find_max_lr(optimizer, criterion, model, train_loader, end_lr, num_iter, step_mode): 
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
    _,max_LR = lr_finder.plot()
    lr_finder.reset()
    return max_LR


def max_lr(model, train_loader):
    # Define optimizer and criterion for loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Specify end LR, number of iterations and step mode
    end_lr = 10
    num_iter = 100
    step_mode = "exp"

    # Passing to find_max_lr in utils
    max_LR = find_max_lr(optimizer, criterion, model, train_loader, end_lr, num_iter, step_mode)
    return max_LR

##################### TRAIN and TEST functions #########################
train_losses = []
test_losses = []
train_acc = []
test_accuracies = []
train_loss = []
train_accuracies = []
lrs = []

def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

# Train Function
def train(model, device, lr_scheduler, criterion, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Predict
        pred = model(data)
        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item() * len(data)
        # Backpropagation
        loss.backward()
        optimizer.step()
        correct += get_correct_count(pred, target)
        processed += len(data)
        pbar.set_description(desc= f'Batch_id={batch_idx}')
        lr_scheduler.step()

    train_acc = 100 * correct / processed
    train_loss /= processed
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    lrs.append(max(lr_scheduler.get_last_lr()))

# Test Function
def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    processed = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += criterion(pred, target).item() * len(data)
            correct += get_correct_count(pred, target)
            processed += len(data)
            # Convert the predictions and target to class indices
            pred_classes = pred.argmax(dim=1)
            target_classes = target
            # Check for misclassified images

    test_acc = 100 * correct / processed
    test_loss /= processed
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)
    print(f"Train Average Loss: {train_losses[-1]:0.4f}")
    print(f"Train Accuracy: {train_accuracies[-1]:0.2f}%")
    print(f"Maximum Learning Rate: ", lrs[-1])
    print(f"Test Average loss: {test_loss:0.4f}")
    print(f"Test Accuracy: {test_acc:0.2f}%")


#  Train and test function call based on number of epochs
def train_test_loop(model, device, train_loader, test_loader, max_LR, optimizer, criterion, EPOCHS=20):

    # Define the number of epochs and the max epoch for max learning rate
    max_lr_epoch = 5

    # Define the learning rate scheduler with max LR from LRFinder
    # Max LR is achieved by 5th epoch
    # Annealing is set to false by three_phase=False
    lr_scheduler = OneCycleLR(optimizer, max_lr=max_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS,
                            pct_start=max_lr_epoch/EPOCHS, div_factor=100, three_phase=False, final_div_factor=100,
                            anneal_strategy='linear')

    # Passing each batch to train and test in train_test module
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch+1)
        train(model, device, lr_scheduler, criterion, train_loader, optimizer, epoch)
        test(model, device, criterion, test_loader)
        print("\n")

