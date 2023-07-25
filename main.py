from __future__ import print_function
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import *
from .models.resnet import *

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

def train_test_loop(model, device, train_loader, test_loader, EPOCHS=20):
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch + 1)
        train(model, device, train_loader, optimizer, epoch)
        misclassified_images, misclassified_labels, misclassified_predictions = test(model, device, test_loader)
        lr_scheduler.step(model.test_losses[-1])  # Adjust learning rate based on validation loss

    return misclassified_images, misclassified_labels, misclassified_predictions
