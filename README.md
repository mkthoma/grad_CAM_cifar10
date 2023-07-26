# Resnet18 and GradCAM on CIFAR 10
In this exercise we will be looking to implement a ResNet18 architecture on the CIFAR 10 dataset using PyTorch. In addition we will be using the imgae augmentations from the Albumentations library and using one cycle learning rate after find the maximum learning rate using the [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) module.

## Objective
We will try to build a Resnet18 Architecture. We will also use the LRFinder module to find the maximum learning rate and use the one cycle learning rate policy for the model. We will also be using the [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) library to identify the misclassifications.

## CIFAR 10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 classes in CIFAR-10 are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

CIFAR-10 is a widely used benchmark dataset in the field of computer vision and deep learning. It consists of 60,000 color images, each of size 32x32 pixels, belonging to 10 different classes. The dataset is divided into 50,000 training images and 10,000 testing images.

The images in CIFAR-10 are relatively low-resolution compared to some other datasets, making it a challenging task for machine learning models to accurately classify the images. The dataset is commonly used for tasks such as image classification, object detection, and image segmentation.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Let us look at some samples of the CIFAR 10 dataset.

![image](https://github.com/mkthoma/era_v1/assets/135134412/a77717b6-4e66-4a9b-b834-39c6c0dbfff3)

Looking at different samples of each class

![image](https://github.com/mkthoma/era_v1/assets/135134412/de11336d-f794-4bc9-b99a-24ca047a94ba)

The colour and RGB representation of the same image can be seen below 

![image](https://github.com/mkthoma/era_v1/assets/135134412/94b8b427-92ef-4365-8690-b5ccad399857)

## [Albumentations](https://albumentations.ai/)

Albumentations is a computer vision tool that boosts the performance of deep convolutional neural networks. The library is widely used in industry, deep learning research, machine learning competitions, and open source projects.

The Albumentations library is an open-source Python library used for image augmentation in machine learning and computer vision tasks. Image augmentation is a technique that involves applying a variety of transformations to images to create new training samples, thus expanding the diversity of the training dataset. This process helps improve the generalization and robustness of machine learning models by exposing them to a wider range of image variations.

Albumentations provides a wide range of image augmentation techniques, including geometric transformations (such as scaling, rotation, and cropping), color manipulations (such as brightness adjustment, contrast enhancement, and saturation changes), noise injection, and more. It is designed to be fast, flexible, and easy to use.

One of the key features of Albumentations is its integration with popular deep learning frameworks such as PyTorch and TensorFlow. It provides a simple and efficient API that allows users to easily incorporate image augmentation into their data preprocessing pipeline, seamlessly integrating it with their training process.

Albumentations supports a diverse set of image data types, including numpy arrays, PIL images, OpenCV images, and others. It also allows for custom augmentation pipelines and provides a rich set of options to control the augmentation parameters and their probabilities.

Overall, Albumentations is a powerful library that simplifies the process of applying image augmentation techniques, helping researchers and practitioners improve the performance and reliability of their computer vision models.

### [Transforms used](https://github.com/mkthoma/custom_resnet/blob/main/utils.py)

- `Normalization`: This transformation normalizes the image by subtracting the mean values (0.4914, 0.4822, 0.4465) and dividing by the standard deviation values (0.247, 0.243, 0.261) for each color channel (RGB). Normalization helps to standardize the pixel values and ensure that they have a similar range.

- `PadIfNeeded`: Resizes the image to the desired size while maintaining the aspect ratio, and if the image is smaller than the specified size, it pads the image with zeros or any other specified value. In this demo, the padding is set to 4.

- `RandomCrop`: Used to randomly crop an image and optionally its corresponding annotations or masks. It is a common transformation used for data augmentation in computer vision tasks.

- `Cutout` :  Used to randomly remove rectangular regions from an image. This technique is often employed as a form of data augmentation to enhance the model's robustness and generalization.The Cutout transform helps introduce regularization and prevents the model from relying on specific local patterns or details in the training data. It encourages the model to focus on more relevant features and improves its ability to generalize to unseen examples. In this demo the cutout is set to (16,16).

- `ToTensorV2`: This transformation converts the image from a numpy array to a PyTorch tensor. It also adjusts the dimensions and channel ordering to match PyTorch's convention (C x H x W).

We can see the train data loader having images with transformations from the albumentations library applied below - 

![image](https://github.com/mkthoma/era_v1/assets/95399001/94d1caf7-8a56-480f-a040-af87f78f7fd4)


These transformations collectively create a diverse set of augmented images for the training data, allowing the model to learn from different variations and improve its generalization capability. 

## LRFinder
The LRfinder module, short for Learning Rate Finder, is a tool commonly used in deep learning frameworks to help determine an optimal learning rate for training neural networks. It aids in selecting a learning rate that leads to fast and stable convergence during the training process.

The LRfinder module works by gradually increasing the learning rate over a defined range and observing the corresponding loss or accuracy values. It then plots the learning rate against the loss or accuracy values to visualize the behavior and identify an appropriate learning rate.

Here's a general outline of how the LRfinder module typically works:

1. Initialize Model: Set up the neural network model architecture.

2. Define LR Range: Specify a range of learning rates to explore. It usually spans several orders of magnitude, from a very small value to a relatively large value.

3. Train with Varying Learning Rates: Iterate through the specified learning rate range and train the model for a fixed number of iterations or epochs using each learning rate. During training, record the loss or accuracy values.

4. Plot Learning Rate vs. Loss/Accuracy: Visualize the learning rate values on the x-axis and the corresponding loss or accuracy values on the y-axis. This plot helps identify the learning rate range where the loss decreases or the accuracy increases most rapidly.

5. Choose Learning Rate: Based on the plot, select a learning rate that represents the steepest descent of the loss or the highest increase in accuracy before any instability or divergence occurs. This learning rate is typically a value slightly before the point where the loss starts to increase or accuracy starts to plateau.

The LRfinder module is a useful tool for automatically exploring and finding a suitable learning rate without extensive manual tuning. It helps strike a balance between a learning rate that is too small, leading to slow convergence, and a learning rate that is too large, causing unstable training or divergence.

The specific implementation and availability of the LRfinder module can vary depending on the deep learning framework or library being used. Some frameworks provide built-in LRfinder modules, while others may require custom implementation or the use of external libraries specifically designed for learning rate finding.

In the [repo](https://github.com/mkthoma/grad_CAM_cifar10/blob/main/main.py) we have used the following parameters for the LR finder:

We found the maximum LR as

![image](https://github.com/mkthoma/grad_CAM_cifar10/assets/95399001/78d065de-85e8-4de2-ad5a-4729d61a307e)

## One Cycle LR Policy
The One Cycle Learning Rate (LR) policy is a learning rate scheduling strategy that aims to improve the training process by varying the learning rate over the course of training. It was introduced by Leslie N. Smith in the paper "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates."

The key idea behind the One Cycle LR policy is to start with a relatively high learning rate, gradually increase it to a maximum value, and then gradually decrease it again towards the end of training. This approach has shown to accelerate the training process, improve model convergence, and achieve better generalization.

The basic steps of the One Cycle LR policy are as follows:

1. Define LR Range: Determine the lower and upper bounds of the learning rate. The lower bound is typically set to a small value, while the upper bound is set to a higher value.

2. Set LR Schedule: Define the LR schedule that varies the learning rate over time. It consists of three phases: increasing LR, decreasing LR, and optionally a final fine-tuning phase.

    - Phase 1: Increasing LR: Start with the minimum learning rate and gradually increase it to the maximum learning rate. This phase is typically set to cover around 30-50% of the total training iterations.

    - Phase 2: Decreasing LR: Gradually decrease the learning rate from the maximum value to the initial minimum value. This phase covers the remaining iterations after Phase 1.

    - Phase 3: Fine-tuning (optional): Optionally, a final phase can be added to further fine-tune the model using a lower learning rate. This phase is typically applied for a smaller number of iterations.

3. Apply LR Schedule: During each training iteration, set the learning rate according to the LR schedule defined in Step 2.

The specific implementation of the One Cycle LR policy may vary depending on the deep learning framework or library being used. Most deep learning frameworks provide built-in functionality or libraries that facilitate the implementation of the One Cycle LR policy.

It's worth noting that the One Cycle LR policy is just one of many learning rate scheduling strategies available, and its effectiveness can depend on factors such as the dataset, model architecture, and specific training scenario. It's often recommended to experiment and tune the LR range and schedule parameters to find the best settings for a particular task.

## Residual Networks
Researchers observed that it makes sense to affirm that “the deeper the better” when it comes to convolutional neural networks. This makes sense, since the models should be more capable (their flexibility to adapt to any space increase because they have a bigger parameter space to explore). However, it has been noticed that after some depth, the performance degrades.

One of the problems ResNets solve is the famous known vanishing gradient. This is because when the network is too deep, the gradients from where the loss function is calculated easily shrink to zero after several applications of the chain rule. This result on the weights never updating its values and therefore, no learning is being performed.

Even after resolving the issue of vanishing/exploding gradients, it was observed that training accuracy dropped when the count of layers was increased. This can be seen in the image below.

![image](https://github.com/mkthoma/era_v1/assets/135134412/5aafbeb2-8442-4659-9603-4a08b3414f49)

It is observed that the network having a higher count (56-layer) of layers are resulting in higher training error in contrast to the network having a much lower count (20-layer) of layers thus resulting in higher test errors! Image Credits to the authors of original [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf).

One might assume that this could be the result of overfitting. However, that is not the case here as deeper networks show higher training error not testing errors. Overfitting tends to occur when training errors are significantly lower than test errors.

This is called the degradation problem. With the network depth increasing the accuracy saturates(the networks learns everything before reaching the final layer) and then begins to degrade rapidly if more layers are introduced.

Since neural networks are good function approximators, they should be able to easily solve the identify function, where the output of a function becomes the input itself.

$f(x) = x$

Following the same logic, if we bypass the input to the first layer of the model to be the output of the last layer of the model, the network should be able to predict whatever function it was learning before with the input added to it.

$f(x) + x = h(x)$

The intuition is that learning f(x) = 0 has to be easy for the network.

Kaiming He, Xiangyu Zhang, Shaoqin Ren, Jian Sun of the Microsoft Research team presented a residual learning framework (ResNets) to help ease the training of the networks that are substantially deeper than before by eliminating the degradation problem. They have proved with evidence that ResNets are easier to optimize and can have high accuracy at considerable depths.

As we have seen previously that latter layers in deeper networks are unable to learn the identity function that is required to carry the result to the output. In residual networks instead of hoping that the layers fit the desired mapping, we let these layers fit a residual mapping.

Initially, the desired mapping is H(x). We let the networks, however, to fit the residual mapping F(x) = H(x)-x, as the network found it easier to optimize the residual mapping rather than the original mapping.

![image](https://github.com/mkthoma/era_v1/assets/135134412/10ff0180-4975-4425-8612-0dcadf538381)

This method of bypassing the data from one layer to another is called as shortcut connections or skip connections. This approach allows the data to flow easily between the layers without hampering the learning ability of the deep learning model. The advantage of adding this type of skip connection is that if any layer hurts the performance of the model, it will be skipped.

![image](https://github.com/mkthoma/era_v1/assets/135134412/3ab84b57-f35a-41a3-ae31-b9cee9ab31b2)

The intuition behind the skip connection is that it is easier for the network to learn to convert the value of f(x) to zero so that it behaves like an identity function rather than learning to behave like an identity function altogether on its own by trying to find the right set of values that would give you the result.

![image](https://github.com/mkthoma/era_v1/assets/135134412/ef5d8e59-c6fc-406a-b5b5-88ef77d944fe)

ResNet uses two major building blocks to construct the entire network.

1. The Identity Block - The identity block consists of a sequence of convolutional layers with the same number of filters, followed by batch normalization and ReLU activation functions. The key characteristic of the identity block is that it incorporates a shortcut connection that bypasses the convolutional layers. The shortcut connection allows the gradient to flow directly through the block without passing through non-linear activation functions, which helps in preserving the gradient during backpropagation.

    ![image](https://github.com/mkthoma/era_v1/assets/135134412/792fed20-9919-4cc7-b4d7-dbf9b31c9d1f)

2. The Conv Block - Also known as a residual block, is another key building block used to construct deep neural networks. It is designed to facilitate the learning process in very deep networks by addressing the vanishing gradient problem. A convolutional block consists of a series of convolutional layers, batch normalization, and non-linear activation functions, along with a shortcut connection. The main difference between a convolutional block and an identity block is the inclusion of a convolutional layer in the former. The convolutional layer in the block allows the network to learn more complex feature representations.

    ![image](https://github.com/mkthoma/era_v1/assets/135134412/223dcaf8-1cf2-4537-85f4-410b9e171e1f)

These components help achieve higher optimization and accuracy for the deep learning models. The results accurately show the effect of using ResNet over plain layers in the graph below.

![image](https://github.com/mkthoma/era_v1/assets/135134412/d812b3da-9343-459a-8b53-cbfc603e69a5)

As seen ResNet performs better than plain neural network models.


## Model Architecture
ResNet18 is a convolutional neural network (CNN) architecture that was introduced in the paper "Deep Residual Learning for Image Recognition" by He et al. (2015). It is a relatively shallow network, with only 18 layers, but it is still able to achieve state-of-the-art results on image classification tasks.

The key innovation of ResNet18 is the use of residual blocks. A residual block is a simple CNN module that consists of two convolutional layers, followed by a shortcut connection. The shortcut connection allows the output of the first convolutional layer to be added to the output of the second convolutional layer. This helps to prevent the gradients from vanishing during backpropagation, which can be a problem for deep neural networks.

The ResNet18 architecture consists of five residual blocks, each of which is followed by a max pooling layer. The final layer of the network is a fully-connected layer with 1000 outputs, corresponding to the 1000 classes in the ImageNet dataset.

ResNet18 has been shown to be very effective for image classification tasks. It achieved a top-5 error rate of 27.2% on the ImageNet dataset, which was a significant improvement over previous CNN architectures. ResNet18 has also been used for a variety of other image processing tasks, such as object detection and segmentation.


The final model can be visualized as: 

![image](https://github.com/mkthoma/grad_CAM_cifar10/assets/95399001/d623b027-bdf8-42f7-af64-aab7b7428c23)


The model summary shows us that we are only using about 173k parameters for this model.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
```

## Tutorial

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
From the plots below we can see that we have achieved accuracy greater than 90% from about $18^{th}$ epoch.

```
EPOCH: 15
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.68it/s]
Train Average Loss: 0.3034
Train Accuracy: 89.60%
Maximum Learning Rate:  0.00016584911438398897
Test Average loss: 0.3454
Test Accuracy: 88.45%


EPOCH: 16
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.68it/s]
Train Average Loss: 0.2646
Train Accuracy: 90.83%
Maximum Learning Rate:  0.00013267227530415108
Test Average loss: 0.3324
Test Accuracy: 89.03%


EPOCH: 17
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.64it/s]
Train Average Loss: 0.2269
Train Accuracy: 92.18%
Maximum Learning Rate:  9.949543622431325e-05
Test Average loss: 0.3023
Test Accuracy: 89.98%


EPOCH: 18
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.59it/s]
Train Average Loss: 0.1922
Train Accuracy: 93.54%
Maximum Learning Rate:  6.631859714447541e-05
Test Average loss: 0.2834
Test Accuracy: 90.75%


EPOCH: 19
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.64it/s]
Train Average Loss: 0.1585
Train Accuracy: 94.74%
Maximum Learning Rate:  3.314175806463758e-05
Test Average loss: 0.2552
Test Accuracy: 91.44%


EPOCH: 20
Batch_id=390: 100%|██████████| 391/391 [00:45<00:00,  8.62it/s]
Train Average Loss: 0.1281
Train Accuracy: 95.82%
Maximum Learning Rate:  -3.50810152003082e-08
Test Average loss: 0.2497
Test Accuracy: 91.81%
```

![image](https://github.com/mkthoma/grad_CAM_cifar10/assets/95399001/da094818-fce8-4f0f-bc9c-cf186a085863)

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