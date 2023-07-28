from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
def train(model, device, criterion, train_loader, optimizer, epoch, lr_scheduler):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    if lr_scheduler == None:
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

    else:
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


def get_optimizer(model, optimizer_input="SGD", lr=0.01):
    optimizer_mapping = {
        'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr),
        'Adagrad': optim.Adagrad(model.parameters(), lr=lr),
        'AdamW': optim.AdamW(model.parameters(), lr=lr)
        # Add more optimizers as needed
    }

    if optimizer_input not in optimizer_mapping:
        raise ValueError("Invalid optimizer name. Available options are: {}".format(list(optimizer_mapping.keys())))

    optimizer_class = optimizer_mapping[optimizer_input]
    return optimizer_class


#  Train and test function call based on number of epochs
def train_test_loop(model, device, train_loader, test_loader, max_LR, max_lr_epoch=5, criterion=nn.CrossEntropyLoss(), optimizer="SGD", EPOCHS=20, lr_scheduler=None, lr=0.01):
    optimizer = get_optimizer(model, optimizer, lr)
    if lr_scheduler == "OneCycle":
        lr_scheduler = OneCycleLR(optimizer, max_lr=max_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS,
                        pct_start=max_lr_epoch/EPOCHS, div_factor=100, three_phase=False, final_div_factor=100,
                        anneal_strategy='linear')

    # Passing each batch to train and test in train_test module
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch+1)
        train(model, device, criterion, train_loader, optimizer, epoch, lr_scheduler)
        test(model, device, criterion, test_loader)
        print("\n") 


def main(optimizer="SGD", dataloader_args=dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True), lr_scheduler=None, grad_CAM=False, criterion=nn.CrossEntropyLoss(), EPOCHS=20, max_lr_epoch=5, lr=0.01):
    device = select_cuda()
    train_data, test_data, train_loader, test_loader, classes = download_data(dataloader_args)
    show_random_samples(test_loader, classes)
    show_class_samples(test_loader, classes)
    show_image_rgb(test_loader, classes)
    model = select_model(device)
    max_LR = max_lr(model, train_loader)
    train_test_loop(model, device, train_loader, test_loader, max_LR, max_lr_epoch, criterion, optimizer, EPOCHS, lr_scheduler, lr)            
    show_accuracy_loss(train_losses, train_accuracies, test_losses, test_accuracies)
    plot_misclassified(model, test_loader, classes, device, grad_CAM, no_misclf=20, plot_size=(4,5))

if __name__ == '__main__':
    main()