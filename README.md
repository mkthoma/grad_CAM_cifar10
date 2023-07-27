# Resnet18 and GradCAM on CIFAR 10
In this exercise we will be looking to implement a ResNet18 architecture on the CIFAR 10 dataset using PyTorch. In addition we will be using the imgae augmentations from the Albumentations library and using one cycle learning rate after find the maximum learning rate using the [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) module.

## Objective
We will try to build a Resnet18 Architecture. We will also use the LRFinder module to find the maximum learning rate and use the one cycle learning rate policy for the model. We will also be using the [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) library to identify the misclassifications.

## Tutorial
The detailed git repo can be found [here](https://github.com/mkthoma/era_v1/tree/main/Session%2011). The tutorial is as follows.
1. Clone the repo
     ```python
     !git clone https://github.com/mkthoma/grad_CAM_cifar10.git
     ```

2. Install dependencies
     ```python
     !pip install albumentations
     !pip install grad-cam
     !pip install torchsummary
     !pip install torch-lr-finder
     ```
3. Import libraries
     ```python
     from __future__ import print_function
     import torch
     import torch.optim as optim
     import albumentations as A
     from albumentations.pytorch import ToTensorV2
     import torch.nn as nn
     from grad_CAM_cifar10.models.resnet import ResNet18
     from grad_CAM_cifar10.utils import *
     from grad_CAM_cifar10.main import *
     ```
4. Select device as cuda using select_device() function from main.
     ```python
     device = select_cuda()
     ```

5. Download data, apply transformations from albumentation library defined in utils
     ```python
     train_data, test_data, train_loader, test_loader, classes = download_data()
     ```

6. Analyze data using different samples of images from functions defined in utils
     ```python
     show_random_samples(test_loader, classes)
     show_class_samples(test_loader, classes)
     show_image_rgb(test_loader, classes)
     ```
7. Select model and print model summary
     ```python
     model = select_model(device)
     ```
8. Find max LR using LRFinder module
     ```python
     max_LR = max_lr(model, train_loader)
     ```
9. Apply train and test functions to the augmented data and see the accuracies and loss values.
     ```python
     train_test_loop(model, device, train_loader, test_loader, max_LR, optimizer=optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3), criterion=nn.CrossEntropyLoss(), EPOCHS=20)
     ```
10. Plot train/test accuracies and losses for the model
     ```python
     show_accuracy_loss(train_losses, train_accuracies, test_losses, test_accuracies)
     ```
11. Plot misclassified images and show GradCAM outputs by setting grad_CAM flag as true in the function
     ```python
     plot_misclassified(model, test_loader, classes, device, no_misclf=20, plot_size=(4,5), grad_CAM=True)
     ```

## Training and Test Accuracies

From the plots below we can see that we have achieved accuracy greater than 90% from about $16^{th}$ epoch.

![image](https://github.com/mkthoma/era_v1/assets/135134412/64284bbc-6773-41eb-8369-b812fb9a8d08)


```
EPOCH: 15
Batch_id=390: 100%|██████████| 391/391 [00:46<00:00,  8.46it/s]
Train Average Loss: 0.2140
Train Accuracy: 92.41%
Maximum Learning Rate:  0.033322951406649606
Test Average loss: 0.3443
Test Accuracy: 89.61%


EPOCH: 16
Batch_id=390: 100%|██████████| 391/391 [00:46<00:00,  8.48it/s]
Train Average Loss: 0.1864
Train Accuracy: 93.27%
Maximum Learning Rate:  0.0266569514066496
Test Average loss: 0.3329
Test Accuracy: 90.11%


EPOCH: 17
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.51it/s]
Train Average Loss: 0.1652
Train Accuracy: 94.32%
Maximum Learning Rate:  0.01999095140664961
Test Average loss: 0.3370
Test Accuracy: 90.44%


EPOCH: 18
Batch_id=390: 100%|██████████| 391/391 [00:46<00:00,  8.45it/s]
Train Average Loss: 0.1412
Train Accuracy: 94.96%
Maximum Learning Rate:  0.013324951406649618
Test Average loss: 0.3231
Test Accuracy: 90.80%


EPOCH: 19
Batch_id=390: 100%|██████████| 391/391 [00:46<00:00,  8.44it/s]
Train Average Loss: 0.1191
Train Accuracy: 95.90%
Maximum Learning Rate:  0.006658951406649613
Test Average loss: 0.3132
Test Accuracy: 91.43%


EPOCH: 20
Batch_id=390: 100%|██████████| 391/391 [00:46<00:00,  8.46it/s]
Train Average Loss: 0.0979
Train Accuracy: 96.59%
Maximum Learning Rate:  -7.048593350378329e-06
Test Average loss: 0.3175
Test Accuracy: 91.50%
```

## Misclassified Images
Now we shall look at the misclassified images that are present after applying the model on the dataset.

![image](https://github.com/mkthoma/grad_CAM_cifar10/assets/95399001/1996ea31-4974-4c67-81e7-d7538f5bd8d3)

We can also apply Grad CAM to see explainibility on the misclassified images as shown below

![image](https://github.com/mkthoma/grad_CAM_cifar10/assets/95399001/3a815f3c-33f3-4577-aefe-39d0fdaaeaa4)


## Conclusion
In conclusion, using ResNet-18 on the CIFAR-10 dataset and applying GradCAM (Gradient-weighted Class Activation Mapping) for misclassifications can significantly improve the understanding and interpretability of the model's predictions.

ResNet-18, with its deep architecture and skip connections, has proven to be a powerful and efficient CNN for image recognition tasks. Trained on the CIFAR-10 dataset, it has shown excellent performance in correctly classifying objects among ten different classes, achieving high accuracy.

However, even state-of-the-art models like ResNet-18 may misclassify certain images due to their inherent complexities and ambiguities. By using GradCAM, we gain valuable insights into how the model makes its predictions. GradCAM highlights the regions in the input image that are most influential in determining the predicted class, providing a visual explanation for the model's decisions.

Through the analysis of misclassifications using GradCAM, we can uncover potential shortcomings and limitations of the model. It allows us to identify patterns and features that the model may not fully capture, leading to incorrect predictions. This information is valuable for improving the model's performance and understanding its strengths and weaknesses.

Moreover, GradCAM helps build trust and transparency in the model's decision-making process. In critical applications like healthcare or autonomous systems, it is crucial to have a clear understanding of how the model arrives at its predictions. GradCAM provides visual evidence that can be inspected and verified by experts, increasing the model's reliability and accountability.

Overall, the combination of ResNet-18 and GradCAM proves to be a powerful tool for image classification tasks, enabling us to not only achieve high accuracy but also gain deeper insights into the model's behavior. By leveraging this understanding, we can work towards enhancing the model's robustness and creating more reliable and trustworthy AI systems.