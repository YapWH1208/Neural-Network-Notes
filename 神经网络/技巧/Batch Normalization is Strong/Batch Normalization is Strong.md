# Theory
- Batch Normalization involves normalizing each individual batch after a linear transformation to ensure controlled ranges.

| Pros                     | Cons                    |
| ------------------------ | ----------------------- |
| Accelerates convergence in neural networks | Effective only for batches |
|                                            | Poor performance with sequential data |
|                                            | Efficiency impacted in distributed training |

- Optimization Methods:
	- Layer Normalization: Normalizes across the feature space of samples.
	- Instance Normalization: Normalizes each channel within the feature space of samples.
	- Group Normalization: A combination of Layer Normalization and Instance Normalization.
$$\large \text{Norm}(Z_i) = \gamma{{Z_{i}-\mu}\over{\sqrt{\sigma^2+\epsilon}}}+\beta$$

# References
- [[5-Minute Deep Learning] #06 Batch Normalization](https://www.bilibili.com/video/BV12d4y1f74C/?vd_source=82cc9f8195ff57b14f4f1d470824ef31)