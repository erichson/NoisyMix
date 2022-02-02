## *NoisyMix*: Boosting Model Robustness and Accuracy

It is well-known that deep learning models are typically [brittle to input perturbations](https://arxiv.org/abs/1312.6199), limiting their applicability in many real-world problems. Four common methods to improve model robustness to input perturbations are:
- Data augmentations
- Stability training
- Mixup 
- Noise injections

**How can we leverage the strength of these methods to further improve both model robustness and test accuracy?**

*NoisyMix* is a training scheme that judiciously combines all of the above components in a single setup to boost both robustness and accuracy. It implements [feature mixup](https://arxiv.org/abs/1806.05236) and [noise injections](https://arxiv.org/abs/2110.02180) on top of a [stability training scheme](https://arxiv.org/abs/1604.04326) on an [AugMix-augmented](https://arxiv.org/abs/1912.02781) data set. We choose the distance measure in the stability objective to be the Jensen-Shannon divergence (JSD). 

The advantage of *NoisyMix* compared to other schemes is illustrated on the following binary classification task on a noisy toy dataset (without augmentation), where it can be seen that *NoisyMix* is most effective at smoothing the decision boundary and yields the best test accuracy.

<p align="center">
    <img src="figures/toy_result.png" height="400">
</p>

Compared to other data augmentation schemes, models trained with *NoisyMix* are more robust to common corruptions and generalize better. In particular, the advantage of *NoisyMix* is substantial when the models are evaluated on ImageNet-C and ImageNet-R, as shown by the following figure.

<p align="center">
    <img src="figures/cimagenet.png" height="300">
</p>


If you would like to use our code, you can simply train a ResNet-18 with *NoisyMix* on CIFAR-100 as follows.


```
export CUDA_VISIBLE_DEVICES=0; python3 cifar.py --arch preactresnet18 --augmix 1 --jsd 1 --alpha 1.0 --manifold_mixup 1 --add_noise_level 0.5 --mult_noise_level 0.5 --sparse_level 0.65 --seed 1
```

You can also simply train a Wide-ResNet-28x2 with *NoisyMix* on CIFAR-100 as follows.

```
export CUDA_VISIBLE_DEVICES=0; python3 cifar.py --arch wideresnet28 --augmix 1 --jsd 1 --alpha 1.0 --manifold_mixup 1 --add_noise_level 0.5 --mult_noise_level 0.5 --sparse_level 0.65 --seed 1
```




For more details, please refer to the [paper](). If you find this work useful and use it on your own research, please concider citing our paper. Please also consider citing [Noisy Feature Mixup](https://arxiv.org/abs/2110.02180) and [AugMix](https://arxiv.org/abs/1912.02781).

