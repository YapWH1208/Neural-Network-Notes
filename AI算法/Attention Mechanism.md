# Introduction
The attention mechanism in deep learning is a method that imitates the human visual and cognitive system, which allows the neural network to focus on relevant parts when processing input data. By introducing the attention mechanism, the neural network can automatically learn and selectively focus on important information in the input, improving the performance and generalization ability of the model.

The attention mechanism is essentially similar to the human selective attention mechanism, and the core goal is to select information that is more critical to the current task goal from a large amount of information. In deep learning, attention mechanisms are usually applied to the processing of sequence data (such as text, speech, or image sequences). Among them, the most typical attention mechanisms include self-attention mechanism, spatial attention mechanism and temporal attention mechanism. These attention mechanisms allow the model to assign different weights to different positions of the input sequence in order to focus on the most relevant parts when processing each sequence element.

# Self-Attention
The basic idea of the self-attention mechanism is that when processing sequence data, each element can be associated with other elements in the sequence, rather than just relying on elements in adjacent positions. It adaptively captures long-range dependencies between elements by calculating their relative importance.

Specifically, for each element in the sequence, the self-attention mechanism calculates the similarity between it and other elements and normalizes these similarities into attention weights. Then, the output of the self-attention mechanism can be obtained by performing a weighted sum of each element and the corresponding attention weight.

<div align="center">
  <img src="https://miro.medium.com/max/2868/1*Cfsh9uK8Y6FhamziJZIKRA.jpeg" alt="Self-Attention">
  <p>Diagram 1: Self-Attention</p>
</div>
For example, we have a sequence data: "LSC is the best!", using $x^1,x^2,x^3,x^4$ to represent it respectively.

> $x^1$: "LSC"
> $x^2$: "is"
> $x^3$: "the"
> $x^4$: "best"

## Embedding
Perform embedding to sequence data to get new vector $a^1,a^2,a^3,a^4$ with formula: $$a^i=Wx^i$$which $W$ is the parameter matrix of the embedding.

## q, k
After embedding, $a^1,a^2,a^3,a^4$ is the input data of the attention mechanism.

Each of $a^1,a^2,a^3,a^4$ will multiply by three matrices, which is $q,k,v$ respectively. In this process, $q,k,v$ will be shared over the process.$$q^i=W^qa^i$$$$k^i=W^ka^i$$
$$v^i=W^va^i$$
Among them, the meaning of $q$ (Query) is generally used to match other words, more precisely, it is used to calculate the association or relationship between the current word or character and other words or characters; the meaning of $k$ (Key) is It is used to match $q$ and can also be understood as the key information of a word or character.

To calculate the relationship between $a^1,a^2,a^3,a^4$ : $$\alpha_{1,i}=\frac{q^1\cdot k^i}{\sqrt{d}}$$
which $d$ is the dimension of matrix $q$ and $k$. In self-attention, dimension of $q$ and $k$ is the same. Here we divide by $\sqrt{d}$ is to prevent the result of dot product between $q$ and $k$ being too large.

## v
$v$ is used to represent the important information or context of word or character, we also can understand that as the feature of word. In the operation of calculating $v$ , $\tilde a_{1,1},a_{1,2},a_{1,3},a_{1,4}$ from $q,k$ operation will mulitply with $v^1,v^2,v^3,v^4$ , formula as below:$$b^1=\sum_i\tilde a_{1,i}v^i$$
## Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

# Conclusion
Attention mechanism uses $q,k$ matrix to get the $v$ matrix by relevance between inputs to know which is the most important information in the input sequence.

# Reference
- [注意力机制综述（图解完整版附代码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/631398525)