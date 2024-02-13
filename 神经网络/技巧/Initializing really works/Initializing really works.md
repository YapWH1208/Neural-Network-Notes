# Theory
- Proper initialization can effectively prevent symmetric weights (where all weights have the same value), which can hinder the learning of complex features.
- It helps to mitigate the occurrence of exploding or vanishing gradients during training.

# Types of Initialization
## Xavier Initialization
- Used for activation functions like tanh.
$$\large Var(w_i) = {{2}\over{n_{in}+n_{out}}}$$
- Normal distribution initialization:
$$\large N(0,{2\over{n_{in}+n_{out}}})$$
- Uniform distribution initialization:
$$\large U({-\sqrt{{6}\over{n_{in}+n_{out}}}},{\sqrt{{6}\over{n_{in}+n_{out}}}})$$
## He Initialization
- Used for ReLU activation.
$$\large Var(w_i) = {{2}\over{n(1+\alpha^2)}}$$
- Normal distribution initialization:
$$\large N(0,{{2}\over{n(1+\alpha^2)}})$$
- Uniform distribution initialization:
$$U(-\sqrt{{{2}\over{n(1+\alpha^2)}}},\sqrt{{{2}\over{n(1+\alpha^2)}}})$$

# References
- [[5-Minute Deep Learning] #04 Parameter Initialization](https://www.bilibili.com/video/BV1r94y1Q7eG/?spm_id_from=333.788&vd_source=82cc9f8195ff57b14f4f1d470824ef31)
- [[5-Minute Deep Learning] #05 Parameter Initialization - Going Deeper!](https://www.bilibili.com/video/BV1PF411K7nb/?spm_id_from=333.788&vd_source=82cc9f8195ff57b14f4f1d470824ef31)