# iclr2020-notes
personal notes from ICLR2020

## Random notes

- amazing pranking [presentation](https://iclr.cc/virtual/poster_H1l_0JBYwS.html)

- ICLR community seem to embrace tiny datasets (hello, MNIST and CIFAR) and Alexnet model

## Attack and Defence

- optimal strategy for both adversarial attack and defence. Done with GAN training, and shown that found generator-attacker really outperforms other attacker approaches [Optimal Strategies Against Generative Attacks](http://www.openreview.net/pdf?id=BkgzMCVtPB)

- make use of (originally) misclassified examples. Approach achieved SOTA on MNIST and CIFAR10 adversarial defence... [Improving Adversarial Robustness Requires Revisiting Misclassified Examples](http://www.openreview.net/pdf?id=rklOg6EFwS)

- poisoning network predictions to fool attacker and increase number of attacks needed until succes [Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks](http://www.openreview.net/pdf?id=SyevYxHtDB)

- mixed precision DNNs are more robust to adversarial attacks (than original non-quantized nets) [EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness Against Adversarial Attacks](http://www.openreview.net/pdf?id=HJem3yHKwH)

- [amazing presentation](https://iclr.cc/virtual/poster_Byl5NREFDr.html) how to attack NLP model with rediculously simple strategy to get only slightly inferior model (attack cost ~few hungred dollars according to the authors) [Thieves on Sesame Street! Model Extraction of BERT-based APIs](http://www.openreview.net/pdf?id=Byl5NREFDr) - now the question is how we can do the same for computer vision tasks?:)

- Efficient defence against physically-realizable attack is adversarial training by these recipy [Defending Against Physically Realizable Attacks on Image Classification](http://www.openreview.net/pdf?id=H1xscnEKDr)

- Skip connections improves adv attacks on another networks [Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets](http://www.openreview.net/pdf?id=BJlRs34Fvr)

- (attack-robust activation) k-winners-take-all defence [Enhancing Adversarial Defense by k-Winners-Take-All](http://www.openreview.net/pdf?id=Skgvy64tvr)

- (comments from sofa == unreliable) there is something called "network verification" which is a direction to verify(proove) certain properties of a model (e.g. robustness to certain type of perturbations - rotations, noise, adversarial, etc). The [work 1](http://www.openreview.net/pdf?id=SJxSDxrKDr) have some theoretical proofs(=verification) that adversarial training improves adversarial robustness and [work 2](http://www.openreview.net/pdf?id=B1evfa4tPB) is fast and efficient verification approach

## Attribution

- New attribution method where unimportant features on feature map are replaced with noise. Converges in ~10 iterations, author also approximated it with single NN which do the same in a single pass. Seems to *really explain* NN predictions (not just exploit the structure of the image). [Restricting the Flow: Information Bottlenecks for Attribution](http://www.openreview.net/pdf?id=S1xWh1rYwB)

![Sanity Check for attribution methods](https://user-images.githubusercontent.com/14358106/80468399-cccea700-8947-11ea-9909-79512cfaf484.png)

![Alpha-Beta LRP fails sanity check (which may be an important property??? as we do not need to train network)](https://user-images.githubusercontent.com/14358106/80468566-00113600-8948-11ea-8732-741e977ed72f.png)

## Robustness

- :question: deformable kernels - seems to be more robust [Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation](http://www.openreview.net/pdf?id=SkxSv6VFvS)

## Optimization

- most likely the best high dimensional optimization without gradients; by Google Brain; maybe useful for hyperparameter search/NAS? [Gradientless Descent: High-Dimensional Zeroth-Order Optimization](http://www.openreview.net/pdf?id=Skep6TVYDB)

- double descent (test loss decreases then increases then decreases again with increase in numbe of parameters) is a frequent phenomena [openai paper well describing it]() BUT(!) it is NOT ALWAYS present acc to [this paper](http://www.openreview.net/pdf?id=H1gBsgBYwH)

- Prox-SGD - sgd with explicit regularization which allows to produce sparser networks with no accuracy loss [ProxSGD: Training Structured Neural Networks under Regularization and Constraints](http://www.openreview.net/pdf?id=HygpthEtvr)

## DL theory

- (see answers below) [How much Position Information Do Convolutional Neural Networks Encode?](http://www.openreview.net/pdf?id=rJeB36NKvB)

![how much position information is endoded?](https://user-images.githubusercontent.com/14358106/80418518-36b36600-88e0-11ea-9b92-471f34ad71cd.png)

- higher depth is beneficial (slide from [this paper presentation](http://www.openreview.net/pdf?id=BJe55gBtvH)) - the paper seems to provide some intuition explaining the examples, but I haven't checked it yet

![Depth vs Width results](https://user-images.githubusercontent.com/14358106/80409601-d0bfe200-88d1-11ea-84d4-936e0fae85fc.png)

- vanilla gradient descent is theoretically optimal (surprise!) in speed of convergence against all over gradient-based methods. But in practice it is not (no surprise) because the theoretical constraints are almost never satisfied. In [this work](http://www.openreview.net/pdf?id=BJgnXpVYwS) authors make some analysis and derived that for less-constrained functions grad descent with clipping is optimal. But they again didn't compare with Adam and others...

![Vanill Grad Descent is theoretically optimal (surprise!)](https://user-images.githubusercontent.com/14358106/80411958-bdaf1100-88d5-11ea-9b48-6b6cfedf22c4.png)

- another work on gradient clipping proved that it does not fight against noisy labels, but 'partial' clipping (see paper for details) does [Can gradient clipping mitigate label noise?](http://www.openreview.net/pdf?id=rklB76EKPr)

![gradient clipping does not help against bad labels, but 'partial' clipping does](https://user-images.githubusercontent.com/14358106/80533600-8eb4a000-89a6-11ea-817b-e360eb681d83.png)

- the image below can give some intuition why compression/quantization works. The paper proves that permulation and rescaling (see below) are the only function-preserving transformation [Functional vs. parametric equivalence of ReLU networks](http://www.openreview.net/pdf?id=Bylx-TNKvH)

![parameter-equivalent networks (for ReLU activation)](https://user-images.githubusercontent.com/14358106/80527877-bd7a4880-899d-11ea-9ede-44670ca34a8c.png)

## Audio

- Face from audio via GAN (insane but works) [From Inference to Generation: End-to-end Fully Self-supervised Generation of Human Face from Speech](http://www.openreview.net/pdf?id=H1guaREYPr)

![image](https://user-images.githubusercontent.com/14358106/80505495-14235a80-897d-11ea-8e57-2cb7b42409ca.png)

- DeepMind made speed synthesis via GAN (and proved that high fidelity speed synthesis with GANs is possible). The [paper](http://www.openreview.net/pdf?id=r1gfQgSFDr) has several tricks: 1)G and D conditioned on linguistic features 2)44h of training data 3)residual blocks with progressive dilation in G 4)several discriminators 5)another unconditioned discriminators (only realism checking) 6)FID and KID from speech recognition model features to track training progress 7)padding masks to generate longer samples (see paper for details).

![key contributions to success](https://user-images.githubusercontent.com/14358106/80421217-c9ee9a80-88e4-11ea-8640-eb0d1d5a231a.png)

- guys implemented conventional audio filters and the results are just fantastic quality - they used tiny models and have high quality results [paper](http://www.openreview.net/pdf?id=B1x1ma4tDr) [github](https://github.com/magenta/ddsp)

- harmonic convolution [Deep Audio Priors Emerge From Harmonic Convolutional Networks](http://www.openreview.net/pdf?id=rygjHxrYDB)

- nlp-inspired pretraining for speech representing it as discrite vocabulary [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](http://www.openreview.net/pdf?id=rylwJxrYDS)

## Video

- "baseline" for Video Continuation Generation which somehow works; based on VideoBERT + autoregressive model [Scaling Autoregressive Video Models](http://www.openreview.net/pdf?id=rJgsskrFwH)

## Generative Models

- Spectral Norm + Rank Norm to improve generalization (decrease generalization gap). Experiments show that this joint normalization improves both classification and GAN performance. [Stable Rank Normalization for Improved Generalization in Neural Networks and GANs](http://www.openreview.net/pdf?id=H1enKkrFDB)

![generalization gap upper bound](https://user-images.githubusercontent.com/14358106/80425335-1c7f8500-88ec-11ea-8a00-2e4584449989.png)

- RealnessGAN - instead of hard labels 0 and 1 for GAN treat them as random variables A0 and A1. Seems to stabilize training as the authors were able to train DCGAN on 1024x1024. Proven theoretical guarantees of convergence. A0 and A1 was taken as different discrete distributions (so D had N output probabilities instead of single) [Real or Not Real, that is the Question](https://openreview.net/pdf?id=B1lPaCNtPB)

![RealnessGAN](https://user-images.githubusercontent.com/14358106/80428366-5784b700-88f2-11ea-850e-daeee64cc0df.png)

- [iclr.video](https://iclr.cc/virtual/poster_Hkxzx0NtDB.html) use classifier as energy function; helps to improve applications of generative models to target tasks (OOD detection, adversarial robustness, etc) [Your classifier is secretly an energy based model and you should treat it like one](http://www.openreview.net/pdf?id=Hkxzx0NtDB)

- To what extent we can manipulate generated image features (zoom in/out or shift or ...) [On the "steerability" of generative adversarial networks](http://www.openreview.net/pdf?id=HylsTT4FvB)

- physics-motivated model for videos (beautiful motivation, but works with only very simple systems of objects so far). The idea is to learn encoder from pixel space, Hamiltonian network (which rules the system state) and decoder from latent space back to pixel space. The system evolution is going by adjusting state +alpha * dt, where alpha is speed. In practice it is useless but I liked the idea and motivation. [Hamiltonian Generative Networks](http://www.openreview.net/pdf?id=HJenn6VFvB)

- visualization tool and (new) metrics to monitor/estimate convergence of GAN [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](http://www.openreview.net/pdf?id=HJeVnCEKwH)

## NAS

- [FasterSeg: Searching for Faster Real-time Semantic Segmentation](http://www.openreview.net/pdf?id=BJgqQ6NYvB)

- [NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search](http://www.openreview.net/pdf?id=SJx9ngStPH)

## Representation

- Supervised learning still performs (much) better, but here are major improvements for unsupervised learning [Self-labelling via simultaneous clustering and representation learning](http://www.openreview.net/pdf?id=Hyx-jyBFPr)

- Representation learned progressively from last NN layer to first NN layer - have reasonable results and ability to control the generation [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](http://www.openreview.net/pdf?id=SJxpsxrYPS)

- reasonable experiments on EMNIST for varying symbols with large amount of control [Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)](http://www.openreview.net/pdf?id=rygeHgSFDH)

## Vision (surprise!)

- :question: Space2Vec - embedding for spacial locations (as far as I understood only for geolocations not for (x,y) on the image) -> use in classification [Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells](http://www.openreview.net/pdf?id=rJljdh4KDH)

## NLP

- BLUE is finally sentenced to death [BERTScore: Evaluating Text Generation with BERT](http://www.openreview.net/pdf?id=SkeHuCVFDr)

![GPUs go brrr](https://user-images.githubusercontent.com/14358106/80547651-6c7c4b80-89c1-11ea-9956-72f6d9c7d7b4.png)

- Nuclear Sampling: instead of top-k most probable words take top-p (sum of top-m words probabilities >= p) [The Curious Case of Neural Text Degeneration](http://www.openreview.net/pdf?id=rygGQyrFvH)

- Unlikelihood (which *outperforms* nuclear sampling from above :point_up: significantly) - the idea is to penalize unlikely situations (negative examples which are either 1)random or 2)repeting words) [Neural Text Generation With Unlikelihood Training](http://www.openreview.net/pdf?id=SJeYe0NtvH)

- :grey_question: Controllable generation with language models [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](http://www.openreview.net/pdf?id=H1edEyBKDS)

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](http://www.openreview.net/pdf?id=r1xMH1BtvB)

![ELECTRA pretraining](https://user-images.githubusercontent.com/14358106/80499323-82fcb580-8975-11ea-87b3-8ea7a678630b.png)

- yes [Are Transformers universal approximators of sequence-to-sequence functions?](http://www.openreview.net/pdf?id=ByxRM0Ntvr)

- The main problem with text GANs (acc to authors) is that Discriminator easily overpowering Generator. To improve Generator training it is rewarded when current generated sentence is better than previously generated sentence [Self-Adversarial Learning with Comparative Discrimination for Text Generation](http://www.openreview.net/pdf?id=B1l8L6EtDS)

- unsupervised text style transfer [code](https://github.com/cindyxinyiwang/deep-latent-sequence-model) [paper](http://www.openreview.net/pdf?id=HJlA0C4tPS)

## Anomaly detection

- main idea is to detect anomaly regions as regions with high difference between original and AE-reconstructed image; solved by gradient minimization of Reconstruction loss (x_i) + ||x_i - x_orig||; looks like anomaly localization improves [Iterative energy-based projection on a normal data manifold for anomaly localization](http://www.openreview.net/pdf?id=HJx81ySKwr)

## Speed improvements

- Automatic search for equivalent (but simpler or faster) set of operations, which is useful for NN inference speed optimization [Deep Symbolic Superoptimization Without Human Knowledge](http://www.openreview.net/pdf?id=r1egIyBFPS)

- Autoregressive decoder speed up [Decoding As Dynamic Programming For Recurrent Autoregressive Models](http://www.openreview.net/pdf?id=HklOo0VFDH)

## Other (to be organized)

- how to measure quality on test set with noisy labels? see :point_down: (exact formula in the paper) [Discrepancy Ratio: Evaluating Model Performance When Even Experts Disagree on the Truth](http://www.openreview.net/pdf?id=Byg-wJSYDS)

![discrepancy formula](https://user-images.githubusercontent.com/14358106/80544948-dba27180-89ba-11ea-9037-3704f61af177.png)

- New NN+tree end2end trained module for tabular data outperforms XGBoost and CatBoost in several datasets [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](http://www.openreview.net/pdf?id=r1eiu2VtwH)

- Decision Trees with criteria = linear model on each node with NN approximation. More robust (but most likely slower) [Locally Constant Networks](http://www.openreview.net/pdf?id=Bke8UR4FPB)

- [Neural tangents](https://github.com/google/neural-tangents) - library for infinite width NNs [paper](http://www.openreview.net/pdf?id=SklD9yrFPS)

![image](https://user-images.githubusercontent.com/14358106/80477440-98151c80-8954-11ea-979a-529d02de1642.png)

![resource requirements](https://user-images.githubusercontent.com/14358106/80477580-da3e5e00-8954-11ea-839c-d8da506761bc.png)

