# iclr2020-notes
personal notes from ICLR2020

## Robustness

- Efficient defence against physically-realizable attack is adversarial training by these recipy [Defending Against Physically Realizable Attacks on Image Classification](http://www.openreview.net/pdf?id=H1xscnEKDr)

- Skip connections improves adv attacks on another networks [Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets](http://www.openreview.net/pdf?id=BJlRs34Fvr)

- (attack-robust activation) k-winners-take-all defence [Enhancing Adversarial Defense by k-Winners-Take-All](http://www.openreview.net/pdf?id=Skgvy64tvr)

- (comments from sofa == unreliable) there is something called "network verification" which is a direction to verify(proove) certain properties of a model (e.g. robustness to certain type of perturbations - rotations, noise, adversarial, etc). The [work 1](http://www.openreview.net/pdf?id=SJxSDxrKDr) have some theoretical proofs(=verification) that adversarial training improves adversarial robustness and [work 2](http://www.openreview.net/pdf?id=B1evfa4tPB) is fast and efficient verification approach

## Optimization

- most likely the best high dimensional optimization without gradients; by Google Brain; maybe useful for hyperparameter search/NAS? [Gradientless Descent: High-Dimensional Zeroth-Order Optimization](http://www.openreview.net/pdf?id=Skep6TVYDB)

- double descent (test loss decreases then increases then decreases again with increase in numbe of parameters) is a frequent phenomena [openai paper well describing it]() BUT(!) it is NOT ALWAYS present acc to [this paper](http://www.openreview.net/pdf?id=H1gBsgBYwH)

## DL theory

- higher depth is beneficial (slide from [this paper presentation](http://www.openreview.net/pdf?id=BJe55gBtvH)) - the paper seems to provide some intuition explaining the examples, but I haven't checked it yet

![Depth vs Width results](https://user-images.githubusercontent.com/14358106/80409601-d0bfe200-88d1-11ea-84d4-936e0fae85fc.png)

- vanilla gradient descent is theoretically optimal (surprise!) in speed of convergence against all over gradient-based methods. But in practice it is not (no surprise) because the theoretical constraints are almost never satisfied. In [this work](http://www.openreview.net/pdf?id=BJgnXpVYwS) authors make some analysis and derived that for less-constrained functions grad descent with clipping is optimal. But they again didn't compare with Adam and others...

![Vanill Grad Descent is theoretically optimal (surprise!)](https://user-images.githubusercontent.com/14358106/80411958-bdaf1100-88d5-11ea-9b48-6b6cfedf22c4.png)



