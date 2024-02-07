# Finding the Suitable Learning Rate
## Learning Rate
- Used to control the step size of updates in each iteration of the model.
- A model's learning rate can have different appropriate values at different stages.
- The only solution is to continuously search for the most suitable learning rate for the current state during training.

## Learning Rate
1. A larger learning rate accelerates the training of the network but may fail to reach the optimal solution. With a too large learning rate, the network may fail to learn effective knowledge.
2. A smaller learning rate slows down the network training. Additionally, too small a learning rate may trap the network in local optima.

> Strategy: Use a large learning rate at the beginning of network training to speed up convergence, then decrease the learning rate to improve model training effectiveness. This is called learning rate decay.

<div align="center">
  <img src="https://github.com/YapWH1208/Neural-Network-Notes/blob/main/%E6%8A%80%E5%B7%A7/Hyperparameter%20Tuning/Pasted%20image%2020231231092406.png" alt="The effect of learning rate on loss at different steps">
  <p>Figure 1: The effect of learning rate on loss at different steps</p>
</div>

## Relationship Between Learning Rate and Batch Size
When batch size is too small:
- The gradient during iterations is not smooth, leading to oscillations in model training loss. Training focuses more on fitting individuals, making it easy for the model to overlook the overall patterns in the data.
- Increased training time.
- Low memory utilization.

When batch size is too large:
- Prone to getting stuck in local optima, affecting model performance. A too large batch size may ignore individual differences in the data and fix the gradient descent direction of the model.
- Memory overflow is possible. In practical training, if additional memory usage is caused by new processes, it may forcibly terminate model training.

> Strategy: Batch size is directly proportional to the learning rate; it's better to use a larger batch size for training as long as GPU memory allows.

It should be emphasized that <font color="#ffff00">a large batch size reduces model accuracy</font>, but <font color="#ffff00">the gradient descent direction of the model is more accurate</font>, so using <font color="#ffff00">a larger learning rate can accelerate model convergence</font>; <font color="#ffff00">a small batch size can better capture individual differences in the model</font>, thus having higher model accuracy, and should be set with a smaller learning rate to address <font color="#ffff00">the issue of oscillating loss</font>.

> OpenAI's research on Batch Size: [[1812.06162] An Empirical Model of Large-Batch Training (arxiv.org)](https://arxiv.org/abs/1812.06162)

# Dropout
- For each training iteration, Dropout randomly selects a portion of neurons and sets their output values to zero.
- Typically used in <font color="#ffff00">fully connected layers</font> and not in convolutional layers. Note that Dropout is not suitable for all scenarios.
- In machine learning and deep learning, dropout is a regularization technique used to reduce overfitting in neural networks.
- The main advantages of using dropout include:
	- Reducing the risk of overfitting.
	- Enhancing generalization capability.
	- Improving model performance.

# Transfer Learning
- Transfer learning involves using pre-trained classic models to directly train our own tasks. Although the domains may differ, there is still a connection between the breadth of learning weights between the two tasks.

# Attempting to Overfit a Small Dataset
- Turn off regularization, dropout, data augmentation, and use a small portion of the training set to train the neural network for a few epochs. Ensure zero loss can be achieved.
- This ensures that the neural network can indeed learn from the dataset.

# Cross Validation
- Cross-validation is a statistical method used to evaluate model performance and reduce overfitting. It divides the dataset into multiple subsets and then performs multiple model training and evaluations, <font color="#ffff00">using different subsets as training and test sets each time</font>. This allows for a more comprehensive evaluation of the model's performance on different data subsets.

# Label Smoothing
- <font color="#ffff00">Label smoothing introduces some noise into the labels of the training set, making the model more robust.</font>

<div align="center">
  <img src="https://github.com/YapWH1208/Neural-Network-Notes/blob/main/%E6%8A%80%E5%B7%A7/Hyperparameter%20Tuning/Pasted%20image%2020231231110115.png" alt="Traditional labels vs. label smoothing">
  <p>Figure 2: Traditional labels vs. label smoothing</p>
</div>

# Data Augmentation
- There are various ways to augment datasets, including cropping, rotating, flipping, adding noise, as well as techniques like cutout, random erasing, and mixup training.

# Summary of CNN Tuning
1. When increasing the size of the training set, find a balance point where the dataset contributes to performance improvement.
2. Data quality is more important than quantity.
3. If the network structure is complex and highly optimized, such as GoogleNet, it is recommended not to make modifications.

# Tips for DNN
1. Shuffle.
2. Augment the dataset.
3. Train on a very small subset of data before training on the full dataset to verify that the network can converge and that the network structure is not flawed.
4. Use dropout to avoid overfitting.
5. Avoid LRN pooling; MAX pooling is faster.
6. The deeper the network, the more ReLU or LeakyReLU should be used instead of sigmoid or tanh.
7. Use Xavier initialization as much as possible.

# Weight Initialization
- Try all different initialization methods and see if there is one method that is superior under other conditions (effectiveness).
- Try unsupervised methods, such as autoencoders, for pre-learning.
- Self-supervision.
- Try using an existing model and retrain the input and output layers for your problem.

# Early Stopping
- Once the performance (validation set) starts to decline during training, you can stop training and learning. This can save a lot of time.
- Early stopping can effectively prevent overfitting on the training data as a regularization method.
- If a certain condition is met (loss measuring accuracy), you can also set up checkpoints to store the model.

