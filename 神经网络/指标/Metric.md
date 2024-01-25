> TP (True Positive): Label true, Prediction true
> FN (False Negative): Label true, Prediction false
> TN (True Negative): Label false, Prediction false
> FP (False Positive): Label false, Prediction true

# Accuracy
$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$
In most of the time, accuracy can predict the correct rate. But when the samples are unbalance, for instance, when the true labels are 90% while the false labels are 10%. It will lead to a the accuracy metric to ahve a tendacy towards true labels.

# Precision
$$\text{Precision}=\frac{TP}{TP+FP}$$
Precision calculate the probability of label true in all the prediction true



# Recall
$$\text{Recall}=\frac{TP}{TP+FN}$$
Recall only calculate the probability of label true equal to prediction true.

When the recall rate is high, the false labels will be find out more easily.

# F1 Score
$$\text{F1 Score}=\frac{2\times Precision\times Recall}{Precision + Recall}$$
F1 score is use to balance the relationship between precision and recall

# ROC, AUC
## Sensitivity, Specificity, True Positive Rate, False Positive Rate
Sensitivity and specificity are two new metrics being introduced to ROC and AUC, these two metrics are the reasons where ROC and AUC can ignore the inbalance samples problem.
### Sensitivity
$$\text{Sensitivity}=\frac{TP}{TP+FN}$$
- Sensitivity is the same as recall but witha different name only
### Specificity
$$\text{Specificity}=\frac{TN}{FP+TN}$$
- Since we care more about positive samples, we need to see how many negative samples are incorrectly predicted as positive samples, so (1-specificity) is used instead of specificity.
### True Positive Rate (TPR)
$$\text{TPR}=\text{Sensitivity}=\frac{TP}{TP+FN}$$
### False Positive Rate (FPR)
$$\text{FPR}=1-\text{Specificity}=\frac{FP}{FP+TN}$$

The following is a diagram of the true positive rate and false positive rate. We find that TPR and FPR are based on the actual performance of 1 and 0 respectively, which means that they observe related probability issues in actual positive samples and negative samples respectively. Because of this, whether the sample is balanced or not will not be affected. Taking the previous example, of the total samples, 90% are positive samples and 10% are negative samples. We know that there is water in using accuracy, but using TPR and FPR are different. Here, TPR only focuses on how many of the 90% positive samples are actually covered, and has nothing to do with the 10%. Similarly, FPR only focuses on how many of the 10% negative samples are covered by errors, and it has nothing to do with the 90% of the negative samples. % has nothing to do, so it can be seen that if we start from the perspective of various results of actual performance, we can avoid the problem of sample imbalance. This is why TPR and FPR are selected as ROC/AUC indicators.

Or we can also think about it from another perspective: conditional probability. We assume that X is the predicted value and Y is the true value. Then these indicators can be expressed as conditional probabilities:

$$\text{Accuracy}=P(Y=1|X=1)$$
$$\text{Recall}=\text{Sensitivity}=P(X=1|Y=1)$$
$$\text{Specificity}=P(X=0|Y=0)$$

From the above three formulas, we can see: If we first use the actual results as the condition (recall rate, specificity), then we only need to consider one kind of sample, and first use the predicted value as the condition (precision rate), then we need to consider both Positive and negative samples. Therefore, indicators that are conditioned on actual results are not affected by sample imbalance. On the contrary, indicators that are conditioned on predicted results will be affected.

## ROC
> ROC curve, also known as receiver operating characteristic curve. This curve was first used in the field of radar signal detection to distinguish signals from noise. Later it was used to evaluate the predictive ability of the model, and the ROC curve was derived based on the confusion matrix.

The two main indicators in the ROC curve are the true positive rate and the false positive rate. The benefits of this choice are also explained above. The abscissa is the false positive rate (FPR), and the ordinate is the true rate (TPR). The following is a standard ROC curve chart.

### Threshold problem of ROC curve
Similar to the previous P-R curve, the ROC curve also draws the entire curve by traversing all thresholds. If we continuously traverse all thresholds, the predicted positive samples and negative samples are constantly changing, and accordingly the ROC curve will slide along the curve.

### How to judge the quality of ROC curve?
Changing the threshold only continuously changes the number of positive and negative samples predicted, that is, TPR and FPR, but the curve itself does not change. So how to judge whether a model's ROC curve is good? This still needs to return to our purpose: FPR represents the degree of response falsely reported by the model, while TPR represents the degree of coverage of the model's predicted response. What we hope for is, of course, the less false alarms, the better, and the more coverage, the better. So to sum up, the higher the TPR and the lower the FPR (that is, the steeper the ROC curve), the better the performance of the model. 

### ROC curve ignores sample imbalance
We have already explained why the ROC curve can ignore sample imbalance. Let's show how it works again in the form of a dynamic graph. We found that no matter how the ratio of red to blue samples changes, the ROC curve has no effect.

## AUC
To calculate points on the ROC curve, we could evaluate the logistic regression model multiple times using different classification thresholds, but this is very inefficient. Fortunately, there is an efficient ranking-based algorithm that can provide us with this information, called Area Under Curve.

What's interesting is that if we connect the diagonals, its area is exactly 0.5. The actual meaning of the diagonal line is: random judgment of response and non-response, the positive and negative sample coverage should be 50%, indicating a random effect. The steeper the ROC curve, the better, so the ideal value is 1, a square, and the worst random judgment is 0.5, so the general AUC value is between 0.5 and 1.

General judgment criteria for AUC
0.5 - 0.7: The effect is low, but it is very good for predicting stocks.
0.7 - 0.85: Average effect
0.85 - 0.95: works well
0.95 - 1: Very good, but generally unlikely

# Reference
- [一文看懂机器学习指标：准确率、精准率、召回率、F1、ROC曲线、AUC曲线 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/93107394#:~:text=%E5%B0%86%E8%A6%81%E7%BB%99%E5%A4%A7%E5%AE%B6%E4%BB%8B%E7%BB%8D%E7%9A%84,C%E6%9B%B2%E7%BA%BF%E3%80%81AUC%E2%80%A6)
- [一文让你彻底理解准确率，精准率，召回率，真正率，假正率，ROC/AUC - AIQ (6aiq.com)](https://www.6aiq.com/article/1549986548173)
