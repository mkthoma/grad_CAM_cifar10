import torch
import numpy as np
from torchvision import datasets
import albumentations as A
import torchvision
import cv2
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# This class will download the cifar10 dataset and apply the transformations for train and test data
# Class for cifar10 dataset and augmentations using Albumetations library
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        if transform == "train":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
                A.RandomCrop(height=32, width=32),
                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=self._get_dataset_mean()),
                ToTensorV2()
            ])
        elif transform == "test":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        transformed = self.transform(image=image)
        image = transformed["image"]
        return image, label

    def _get_dataset_mean(self):
        # Calculate the mean of the dataset
        return tuple(self.data.mean(axis=(0, 1, 2)) / 255)

# Function for train and test dataloader
def dataloader(dataset, dataloader_args):
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return data_loader


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
####################################################### VISUALIZATION FUNCTIONS ################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


# shows images from each class
def show_class_samples(data_loader, classes):
    # Obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # Convert images to numpy for display

    # Create a dictionary to store images per class
    images_per_class = {}

    # Group images by class
    for image, label in zip(images, labels):
        class_name = classes[label]
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image)

    fig = plt.figure(figsize=(20, 10))

    # Display 5 images per class
    for idx, class_name in enumerate(classes):
        if class_name in images_per_class:  # Check if class exists in the dictionary
            images = images_per_class[class_name][:5]
            for i in range(5):
                ax = fig.add_subplot(len(classes), 5, idx * 5 + i + 1, xticks=[], yticks=[])
                # Clip and normalize the image data to [0, 1]
                img = np.clip(np.transpose(images[i], (1, 2, 0)), 0, 1)
                ax.imshow(img)
                if i == 0:
                    ax.set_ylabel(class_name)  # Show class name on the y-axis

    plt.tight_layout()
    plt.show()
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# shows random images from different classes
def show_random_samples(data_loader, classes):
    # obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display

    # normalize images
    images = np.transpose(images, (0, 2, 3, 1))  # convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    images = (images - images.min()) / (images.max() - images.min())  # normalize to range [0, 1]

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(20, 8))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(4, 5, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(classes[labels[idx]])

# shows rgb channel of an image
def show_image_rgb(data_loader, classes):
    # Obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # Convert images to numpy for display

    # Original image
    original_img = np.squeeze(images[3])
    image_class = classes[labels[3]]  # Get the class name for the image

    # Normalize original image
    normalized_img = (original_img.transpose(1, 2, 0) + 1) / 2  # Normalize image to [0, 1] range

    # Clip normalized image to valid range [0, 1]
    normalized_img = np.clip(normalized_img, 0, 1)

    # RGB channels
    rgb_img = np.transpose(normalized_img, (0, 1, 2))
    channels = ['Red channel', 'Green channel', 'Blue channel']

    fig = plt.figure(figsize=(40, 12))

    # Display original image
    ax_original = fig.add_subplot(1, 4, 1)
    ax_original.imshow(normalized_img)
    ax_original.set_title(f"Original Image\nClass: {image_class}")  # Add class label to the title

    # Display RGB channels
    for idx in range(rgb_img.shape[2]):
        ax = fig.add_subplot(1, 4, idx + 2)
        img = rgb_img[:, :, idx]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)  # Specify vmin and vmax to ensure correct range
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(
                    str(val),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=8,
                    color='white' if img[x][y] < thresh else 'black'
                )

    plt.show()

# shows misclassified images
def plot_misclassified(model, test_loader, classes, device, no_misclf=20, plot_size=(4,5), grad_CAM=False):

    def convert_image_np(inp, mean, std):
      inp = inp.numpy().transpose((1, 2, 0))
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      return inp

    count = 0
    k = 0
    misclf = list()
    dataset_mean, dataset_std = np.array([0.49139968, 0.48215841, 0.44653091]), np.array([0.24703223, 0.24348513, 0.26158784])
  
    while count<no_misclf:
        img_model, label = test_loader.dataset[k]
        pred = model(img_model.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img_model, dataset_mean, dataset_std)
            misclf.append((img_model, img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*2,rows*2))
    print("Misclassified images\n")
    for i in range(1, cols * rows + 1):
        _, img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Prediction: {classes[pred]}\n Target: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if grad_CAM:
      print("\n\nMisclassified images using GradCAM\n")
      grad_cam = GradCAM(model=model, target_layers=[model.layer4[-1]],use_cuda=True)

      targets = [ClassifierOutputTarget(label)]

      grayscale_cam = grad_cam(input_tensor=img_model.unsqueeze(0), targets=targets)
      # In this example grayscale_cam has only one image in the batch:
      grayscale_cam = grayscale_cam[0, :]

      output = show_cam_on_image(img, grayscale_cam, use_rgb=True)

      rows, cols = plot_size
      figure = plt.figure(figsize=(cols*2,rows*2))

      for i in range(1, cols * rows + 1):
          img_model, img, label, pred = misclf[i-1]
          targets = [ClassifierOutputTarget(label)]
          grayscale_cam = grad_cam(input_tensor=img_model.unsqueeze(0), targets=targets)
          grayscale_cam = grayscale_cam[0, :]
          output = show_cam_on_image(img, grayscale_cam, use_rgb=True)
          figure.add_subplot(rows, cols, i) # adding sub plot
          plt.title(f"Prediction: {classes[pred]}\n Target: {classes[label]}") # title of plot
          plt.axis("off") # hiding the axis
          plt.imshow(output, cmap='gray', alpha=0.40)

      plt.tight_layout()
      plt.show()


# plots train and test accuracy and losses
def show_accuracy_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")