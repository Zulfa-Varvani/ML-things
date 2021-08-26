# CIFAR-10 Image Classification 😓
* Used Keras
* CNN for classification
* 50000 images for training and 10000 images for testing
* Accuracy with baseline VGG model
  * 1 VGG block - 67%
  * 2 VGG blocks - 71.5%
  * 3 VGG block - 73%

### Model:
* [VGG model](https://arxiv.org/abs/1409.1556) is easy to understand and implement architecture
* Stacking convolutional layers with small 3x3 filters followed by max pooling layer. These form a block which can be repeated and the filters are increasing with depth to the network
* Model is optimized using stochiastic gradient descent
* Used 3 VGG block
* Model overfits test dataset within first 15-20 epochs

### Improvements:
* Accuracy increased slightly with increase in VGG blocks
* Will attempt regularization techniques to improve model
  * Added Dropout layers after each max pooling layer with a rate of 20%
    * Accuracy of __82.4%__
    * Model converges well for first 40 epochs - no further improvements after. Can add early stopping to save the model on the test set during training when no further improvements are made
  * Using weight decay/weight regularization did NOT improve accuracy
  * Data augmentation - making small random modifications to copies of examples in training dataset
    * Accuracy of __84.3%__
    * Horizontal flips, minor shifts, and small zooming/cropping

[Work in progress]
<br> 
Using dataset from CIFAR-10
![cifar10 image](https://github.com/Zulfa-Varvani/ML-things/blob/main/image%20classification/cifar10.png)

#### Current Accuracy: 84.3%
