[Work in progress]
<br> 
Using dataset from CIFAR-10
![cifar10 image](https://github.com/Zulfa-Varvani/ML-things/blob/main/image%20classification/cifar10.png)

* Used Keras
* CNN for classification
* 50000 images for training and 1000 images for testing
* Accuracy of about 67% using baseline VGG model

### Model:
* [VGG model](https://arxiv.org/abs/1409.1556) is easy to understand and implement architecture
* Stacking convolutional layers with small 3x3 filters followed by max pooling layer. These form a block which can be repeated and the filters are increasing with depth to the network
* Model is optimized using stochiastic gradient descent
* Used 1 VGG block
* Model overfits test dataset

### Improvements:
* Increasing accuracy with increasing VGG blocks
* Will attempt regularization techniques to improve model
