# iclr2020-notes
personal notes from ICLR2020

## Random notes

- amazing pranking [presentation](https://iclr.cc/virtual/poster_H1l_0JBYwS.html)

## Attribution

- New attribution method where unimportant features on feature map are replaced with noise. Converges in ~10 iterations, author also approximated it with single NN which do the same in a single pass. Seems to *really explain* NN predictions (not just exploit the structure of the image). [Restricting the Flow: Information Bottlenecks for Attribution](http://www.openreview.net/pdf?id=S1xWh1rYwB)

![Sanity Check for attribution methods](https://user-images.githubusercontent.com/14358106/80468399-cccea700-8947-11ea-9909-79512cfaf484.png)

![Alpha-Beta LRP fails sanity check (which may be an important property??? as we do not need to train network)](https://user-images.githubusercontent.com/14358106/80468566-00113600-8948-11ea-8732-741e977ed72f.png)

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

- Face from audio via GAN (insane but works) [From Inference to Generation: End-to-end Fully Self-supervised Generation of Human Face from Speech](http://www.openreview.net/pdf?id=H1guaREYPr)

![image](https://user-images.githubusercontent.com/14358106/80505495-14235a80-897d-11ea-8e57-2cb7b42409ca.png)

- DeepMind made speed synthesis via GAN (and proved that high fidelity speed synthesis with GANs is possible). The [paper](http://www.openreview.net/pdf?id=r1gfQgSFDr) has several tricks: 1)G and D conditioned on linguistic features 2)44h of training data 3)residual blocks with progressive dilation in G 4)several discriminators 5)another unconditioned discriminators (only realism checking) 6)FID and KID from speech recognition model features to track training progress 7)padding masks to generate longer samples (see paper for details).

![key contributions to success](https://user-images.githubusercontent.com/14358106/80421217-c9ee9a80-88e4-11ea-8640-eb0d1d5a231a.png)

- guys implemented conventional audio filters and the results are just fantastic quality - they used tiny models and have high quality results [paper](http://www.openreview.net/pdf?id=B1x1ma4tDr) [github](https://github.com/magenta/ddsp)

- harmonic convolution [Deep Audio Priors Emerge From Harmonic Convolutional Networks](http://www.openreview.net/pdf?id=rygjHxrYDB)

## Video

- "baseline" for Video Continuation Generation which somehow works; based on VideoBERT + autoregressive model [Scaling Autoregressive Video Models](http://www.openreview.net/pdf?id=rJgsskrFwH)

## Generative Models

- Spectral Norm + Rank Norm to improve generalization (decrease generalization gap). Experiments show that this joint normalization improves both classification and GAN performance. [Stable Rank Normalization for Improved Generalization in Neural Networks and GANs](http://www.openreview.net/pdf?id=H1enKkrFDB)

![generalization gap upper bound](https://user-images.githubusercontent.com/14358106/80425335-1c7f8500-88ec-11ea-8a00-2e4584449989.png)

- RealnessGAN - instead of hard labels 0 and 1 for GAN treat them as random variables A0 and A1. Seems to stabilize training as the authors were able to train DCGAN on 1024x1024. Proven theoretical guarantees of convergence. A0 and A1 was taken as different discrete distributions (so D had N output probabilities instead of single) [Real or Not Real, that is the Question](https://openreview.net/pdf?id=B1lPaCNtPB)

![RealnessGAN](https://user-images.githubusercontent.com/14358106/80428366-5784b700-88f2-11ea-850e-daeee64cc0df.png)

- [iclr.video](https://iclr.cc/virtual/poster_Hkxzx0NtDB.html) use classifier as energy function; helps to improve applications of generative models to target tasks (OOD detection, adversarial robustness, etc) [Your classifier is secretly an energy based model and you should treat it like one](http://www.openreview.net/pdf?id=Hkxzx0NtDB) 

- physics-motivated model for videos (beautiful motivation, but works with only very simple systems of objects so far). The idea is to learn encoder from pixel space, Hamiltonian network (which rules the system state) and decoder from latent space back to pixel space. The system evolution is going by adjusting state +alpha * dt, where alpha is speed. In practice it is useless but I liked the idea and motivation. [Hamiltonian Generative Networks](http://www.openreview.net/pdf?id=HJenn6VFvB)

- optimal strategy for both adversarial attack and defence. Done with GAN training, and shown that found generator-attacker really outperforms other attacker approaches [Optimal Strategies Against Generative Attacks](http://www.openreview.net/pdf?id=BkgzMCVtPB)

- make use of (originally) misclassified examples. Approach achieved SOTA on MNIST and CIFAR10 adversarial defence... [Improving Adversarial Robustness Requires Revisiting Misclassified Examples](http://www.openreview.net/pdf?id=rklOg6EFwS)

## NAS

- [FasterSeg: Searching for Faster Real-time Semantic Segmentation](http://www.openreview.net/pdf?id=BJgqQ6NYvB)

## Representation

- Supervised learning still performs (much) better, but here are major improvements for unsupervised learning [Self-labelling via simultaneous clustering and representation learning](http://www.openreview.net/pdf?id=Hyx-jyBFPr)

- Representation learned progressively from last NN layer to first NN layer - have reasonable results and ability to control the generation [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](http://www.openreview.net/pdf?id=SJxpsxrYPS)

- reasonable experiments on EMNIST for varying symbols with large amount of control [Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)](http://www.openreview.net/pdf?id=rygeHgSFDH)

## NLP

- :grey_question: Controllable generation with language models [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](http://www.openreview.net/pdf?id=H1edEyBKDS)

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](http://www.openreview.net/pdf?id=r1xMH1BtvB)

![ELECTRA pretraining](https://user-images.githubusercontent.com/14358106/80499323-82fcb580-8975-11ea-87b3-8ea7a678630b.png)

- unsupervised text style transfer [code](https://github.com/cindyxinyiwang/deep-latent-sequence-model) [paper](http://www.openreview.net/pdf?id=HJlA0C4tPS)

## Speed improvements

- Automatic search for equivalent (but simpler or faster) set of operations, which is useful for NN inference speed optimization [Deep Symbolic Superoptimization Without Human Knowledge](http://www.openreview.net/pdf?id=r1egIyBFPS)

- Autoregressive decoder speed up [Decoding As Dynamic Programming For Recurrent Autoregressive Models](http://www.openreview.net/pdf?id=HklOo0VFDH)

## Other (to be organized)

- Decision Trees with criteria = linear model on each node with NN approximation. More robust (but most likely slower) [Locally Constant Networks](http://www.openreview.net/pdf?id=Bke8UR4FPB)

- [Neural tangents](https://github.com/google/neural-tangents) - library for infinite width NNs [paper](http://www.openreview.net/pdf?id=SklD9yrFPS)

![image](https://user-images.githubusercontent.com/14358106/80477440-98151c80-8954-11ea-979a-529d02de1642.png)

![resource requirements](https://user-images.githubusercontent.com/14358106/80477580-da3e5e00-8954-11ea-839c-d8da506761bc.png)

