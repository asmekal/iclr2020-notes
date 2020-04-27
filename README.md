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

- (see answers below) [How much Position Information Do Convolutional Neural Networks Encode?](http://www.openreview.net/pdf?id=rJeB36NKvB)

![how much position information is endoded?](https://user-images.githubusercontent.com/14358106/80418518-36b36600-88e0-11ea-9b92-471f34ad71cd.png)

- higher depth is beneficial (slide from [this paper presentation](http://www.openreview.net/pdf?id=BJe55gBtvH)) - the paper seems to provide some intuition explaining the examples, but I haven't checked it yet

![Depth vs Width results](https://user-images.githubusercontent.com/14358106/80409601-d0bfe200-88d1-11ea-84d4-936e0fae85fc.png)

- vanilla gradient descent is theoretically optimal (surprise!) in speed of convergence against all over gradient-based methods. But in practice it is not (no surprise) because the theoretical constraints are almost never satisfied. In [this work](http://www.openreview.net/pdf?id=BJgnXpVYwS) authors make some analysis and derived that for less-constrained functions grad descent with clipping is optimal. But they again didn't compare with Adam and others...

![Vanill Grad Descent is theoretically optimal (surprise!)](https://user-images.githubusercontent.com/14358106/80411958-bdaf1100-88d5-11ea-9b48-6b6cfedf22c4.png)

## Audio

- DeepMind made speed synthesis via GAN (and proved that high fidelity speed synthesis with GANs is possible). The [paper](http://www.openreview.net/pdf?id=r1gfQgSFDr) has several tricks: 1)G and D conditioned on linguistic features 2)44h of training data 3)residual blocks with progressive dilation in G 4)several discriminators 5)another unconditioned discriminators (only realism checking) 6)FID and KID from speech recognition model features to track training progress 7)padding masks to generate longer samples (see paper for details).

![key contributions to success](https://user-images.githubusercontent.com/14358106/80421217-c9ee9a80-88e4-11ea-8640-eb0d1d5a231a.png)

- guys implemented conventional audio filters and the results are just fantastic quality - they used tiny models and have high quality results [paper](http://www.openreview.net/pdf?id=B1x1ma4tDr) [github](https://github.com/magenta/ddsp)

## Video

- "baseline" for Video Continuation Generation which somehow works; based on VideoBERT + autoregressive model [Scaling Autoregressive Video Models](http://www.openreview.net/pdf?id=rJgsskrFwH)

## GANs

- Spectral Norm + Rank Norm to improve generalization (decrease generalization gap). Experiments show that this joint normalization improves both classification and GAN performance. [Stable Rank Normalization for Improved Generalization in Neural Networks and GANs](http://www.openreview.net/pdf?id=H1enKkrFDB)

![generalization gap upper bound](https://user-images.githubusercontent.com/14358106/80425335-1c7f8500-88ec-11ea-8a00-2e4584449989.png)

- RealnessGAN - instead of hard labels 0 and 1 for GAN treat them as random variables A0 and A1. Seems to stabilize training as the authors were able to train DCGAN on 1024x1024. Proven theoretical guarantees of convergence. A0 and A1 was taken as different discrete distributions (so D had N output probabilities instead of single) [Real or Not Real, that is the Question](https://openreview.net/pdf?id=B1lPaCNtPB)

![RealnessGAN](https://user-images.githubusercontent.com/14358106/80428366-5784b700-88f2-11ea-850e-daeee64cc0df.png)

- [iclr.video](https://iclr.cc/virtual/poster_Hkxzx0NtDB.html) use classifier as energy function; helps to improve applications of generative models to target tasks (OOD detection, adversarial robustness, etc) [Your classifier is secretly an energy based model and you should treat it like one](http://www.openreview.net/pdf?id=Hkxzx0NtDB) 

- physics-motivated model for videos (beautiful motivation, but works with only very simple systems of objects so far). The idea is to learn encoder from pixel space, Hamiltonian network (which rules the system state) and decoder from latent space back to pixel space. The system evolution is going by adjusting state +alpha * dt, where alpha is speed. In practice it is useless but I liked the idea and motivation. [Hamiltonian Generative Networks](http://www.openreview.net/pdf?id=HJenn6VFvB)

