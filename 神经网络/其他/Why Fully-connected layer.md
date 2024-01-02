In most of the neural network in recent works, we can easily find out that the last most layer are fully-connected (fc) layer while the layer before fc layer are network architecture like convolution layer, sampling layer, activation function and so on. These operation is used for feature extraction which maps the original data to the hidden layer feature space, while fc layer is used to classify or map the feature representation to the label space.

As fully-connected layer operates by matrix multiplication and linear combination of neurons, this makes it transform a feature space to a new feature space.

Effects of fc layer to the model:
1. Number of layers of fc layers
2. Number of neurons in a single fc layer
3. Activation function

The learning capabilities of the model will increase in terms on increasing fc layers and the number of neurons of it theoritically. But it will also increase the models complexity making it easily overfitting.

This is why most of the models uses 2 or more fc layers to increase the non-linearity, but not stacking it deep to prevent overfitting due to complexity too high.


# Reference
- [全连接层的作用是什么？](https://www.zhihu.com/question/41037974/answer/2549676265)