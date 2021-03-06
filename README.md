# iclr2020-notes
personal notes from ICLR2020

## General Notes

### Main links (official)

Main page https://iclr.cc/

[Virtual format notes (medium)](https://medium.com/@iclr_conf/format-for-the-iclr2020-virtual-conference-76716ddea640)

All contents https://iclr.cc/virtual_2020/

[Reflections after the conference (medium)](https://medium.com/@iclr_conf/gone-virtual-lessons-from-iclr2020-1743ce6164a3)

### Comments about format

ICLR'20 due to well-known circumstences were held online world-wide from 26 to 30 April. In a week after conference ended organizers made all of the materials public - https://iclr.cc/virtual_2020/

All of the papers were presented by 5 min video - both posters and orals. Great works were up to 15 mins. Five poster sessions every day, each work was presented on exactly two of them. During presentation time authors held meetings in zoom rooms and can be asked directly. But for me much simpler way to ask was to write in rocket-chat (each poster had its own channel there) and ask there not waiting until particular time.

Oral presentations were cool and useful (not always, but often), but I was missing the posters in the form of 'virtual pieces of paper' because it would be much simpler to understand the idea of the work seeing its entirely in one image instead of scrolling the slides.

### Me on conference

From ~650 accepted papers I looked on ~100, summarizing (in most cases just minor comment or mention) them below in this document. I tried to group them by contents if possible.

This time I were unable to look on all interesting papers I wanted to. Which means that each section of this document can miss importatn works in the domain. I still have ~70 important papers in the queue to watch later... Maybe one day I will take a look and summarize them, maybe...

In general I tried to broaden my horizons in many topics like GANs, NAS, Optimization, DL Theory, Adversarial examples, Vision/NLP/Audio/Video processing. I skipped entirely prunning/quantization and too theoretical works.

If the paper has :question: - it means that paper looks good, but I didn't understand something and I am not confident in the comment I write here.

## Key takeaways (TL;DR)

### Attack & Defence

- Now you can steal model by API which returns only its predictions (and save a lot of money $$$) - checked for BERT-like NLP models, could the same be done for CV tasks?

- There is a way to defend from physically-realizable attacks (doing adversarial in the real world)

- Proper usage of misclassified examples can improve robustness to adversarial examples

### Optimization

- There are methods to estimate generalization gap knowing only train error (metric). The formula is `test_loss < train_loss + G` where G is generalization gap. And this `G` can be estimated on train data only. Which means that in theory you can use more training data joining validation set as it is no longer needed to select the best epoch or when to stop training. Joining validation to training may be used for some critical applications where each point of accuracy is really important (because increasing your data by 20% does not give you much in most cases - you will need 10x increase to go significantly better). Anyway, some works to estimate `G` are in this report, see below.

- Exponential learning rate also works! Yes, you can increase lr to, say, `1e22` and you will still converge. Moreover authors of that work found a way to train with exponential lr with same quality as constant/cosine schedule. The key result here - **there are always many good lr schedules** which will lead to same results. However, the questions "so which one of them is the best?" or "which lr schedule generalize well for more problems?" still remains.

- ICLR community seem to embrace tiny datasets (hello, MNIST and CIFAR) and Alexnet model which makes research simpler and faster, but often useless for applications, due to lack of generalization of larger datasets

### DL Theory

- Backpropaganda: several conventional myths disproved empirically. 1)**suboptimal local minima DO exist** (that's why initialization matters); 2)**l2 norm regularization DECREASES performance**; 3)**rank minimization DECREASES performance**

- Vanilla Grad Descent under certain assumptions of the problem is optimal (which means that **no other gradient-based method can converge faster!**) but in practice these assumptions not (never?=)) satisfied, so your Adam will work better. One of the works prove that on weaker set of assumptions clipped SGD is optimal (most likely it is still useless for real problems).

### GANs

- **(insane) Faces can be synthesized from audio** - it really works to some extent

- DeepMind proved that high-fidelity speed can be generated with GANs (which is faster than augoregressive wavenet)

- Stable Rank Normalization: Spectral Norm + Rank Norm to improve generalization (decrease generalization gap)

### NAS

- Robust DARTS - super simple approach to train DARTS much better by early stopping + continuation with more regularization (see details below)

### Layers

- Deconvolution - **modification to vanilla convolution which is free on inference, improves robustness for all conventional models where it was applied**. The only cost is training time increase (up to 10x in my test). See details below

### NLP

- Just see NLP section, everything there is good:)

# Summaries and comments

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

- method which explains *how* and *why* particular regions of the input image are responsible for classifier prediction. The method is to train external Generator (which will generate degraded image similar to input, e.g. if classifier outputs 0.9 probability, generator receives input image and desire to make classifier have 0.1 probability) and Discriminator trained with KL divergence from actual output of the classifier and L1 distance loss to the original input. The methos is applicable to the cases then *how* and *why* of the classifier model are important (e.g. medical applications). However, it requires to train additional model (which is GAN so the training won't be easy) instead of most of attribution methods which can be directly applied to classifier any additional training and external models [Explanation by Progressive Exaggeration](http://www.openreview.net/pdf?id=H1xFWgrFPS)

- New attribution method where unimportant features on feature map are replaced with noise. Converges in ~10 iterations, author also approximated it with single NN which do the same in a single pass. Seems to *really explain* NN predictions (not just exploit the structure of the image). [Restricting the Flow: Information Bottlenecks for Attribution](http://www.openreview.net/pdf?id=S1xWh1rYwB)

![Sanity Check for attribution methods](https://user-images.githubusercontent.com/14358106/80468399-cccea700-8947-11ea-9909-79512cfaf484.png)

![Alpha-Beta LRP fails sanity check (which may be an important property??? as we do not need to train network)](https://user-images.githubusercontent.com/14358106/80468566-00113600-8948-11ea-8732-741e977ed72f.png)

## Robustness

- :question: deformable kernels - seems to be more robust [Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation](http://www.openreview.net/pdf?id=SkxSv6VFvS)

## Optimization

### Generalization gap estimation

**This part is about estimation of test loss knowing train loss, i.e. $L_{test} = L_{train} + G$, where $G$ is generalization gap**. In theory if the estimation is reliable it will allow to merge train and validation (so to have more data) and estimate best model / best training epoch reliably *without validation data*.

- GSNR = mean^2 / variance or gradients gives an estimation of generalization on test set (the higher - the better) [Understanding Why Neural Networks Generalize Well Through GSNR of Parameters](http://www.openreview.net/pdf?id=HyevIJStwH)

- another work on generalization gap estimation (upper bound). It turns out that if we have a trained network and replace one fixed layer the loss will not increase drastically for most of the layers chosen. In fact only small amount of layers (called 'critical' in the paper) will severly affect performance after such transformation. The method could predict e.g. that trained resnet18 has larger generalization gap than resnet34 [The intriguing role of module criticality in the generalization of deep networks](http://www.openreview.net/pdf?id=S1e4jkSKvB)

### [Generalization gap ends]

- exponential lr also converge (to the comparable results) for wide range of DL models which has normalization, which rases a questions about learning schedulers in general - is it even worth trying to wary them [An Exponential Learning Rate Schedule for Deep Learning](http://www.openreview.net/pdf?id=rJg8TeSFDH)

![exponential lr decay really works...](https://user-images.githubusercontent.com/14358106/80638665-7e192e00-8a69-11ea-8d91-88f05bc251c4.png)

![what about learning rates](https://user-images.githubusercontent.com/14358106/80637732-262df780-8a68-11ea-8fd6-24b5315199c6.png)

- most likely the best high dimensional optimization without gradients; by Google Brain; maybe useful for hyperparameter search/NAS? [Gradientless Descent: High-Dimensional Zeroth-Order Optimization](http://www.openreview.net/pdf?id=Skep6TVYDB)

- double descent (test loss decreases then increases then decreases again with increase in numbe of parameters) is a frequent phenomena [openai paper well describing it]() BUT(!) it is NOT ALWAYS present acc to [this paper](http://www.openreview.net/pdf?id=H1gBsgBYwH)

- Prox-SGD - sgd with explicit regularization which allows to produce sparser networks with no accuracy loss [ProxSGD: Training Structured Neural Networks under Regularization and Constraints](http://www.openreview.net/pdf?id=HygpthEtvr)

## DL theory

- disproving false claims (see slides below): suboptimal local minima DO exist (that's why initialization matters), l2 norm regularization DECREASES performance, rank minimization DECREASES performance [Truth or backpropaganda? An empirical investigation of deep learning theory](http://www.openreview.net/pdf?id=HyxyIgHFvr)

![myths outline](https://user-images.githubusercontent.com/14358106/80647920-ecfd8380-8a77-11ea-8f48-6003b1ce97aa.png)
![suboptimal local minima](https://user-images.githubusercontent.com/14358106/80648042-2504c680-8a78-11ea-9b69-6c6cc2284ecf.png)
![l2 norm regularization decrease performance](https://user-images.githubusercontent.com/14358106/80648084-364dd300-8a78-11ea-899d-a30651743234.png)
![rank minimization decrease performance](https://user-images.githubusercontent.com/14358106/80648129-49f93980-8a78-11ea-9886-6707bd35024f.png)


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

- [What Can Neural Networks Reason About?](http://www.openreview.net/pdf?id=rJxbJeHFPS)

![image](https://user-images.githubusercontent.com/14358106/80643828-65147b00-8a71-11ea-90ee-fae0699a52a0.png)

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

- theoretical analysis of GANs lead to combining several discriminator objective into 'supergan' which shouls converge more stable in theory [Smoothness and Stability in GANs](http://www.openreview.net/pdf?id=HJeOekHKwr)

- Spectral Norm + Rank Norm to improve generalization (decrease generalization gap). Experiments show that this joint normalization improves both classification and GAN performance. [Stable Rank Normalization for Improved Generalization in Neural Networks and GANs](http://www.openreview.net/pdf?id=H1enKkrFDB)

![generalization gap upper bound](https://user-images.githubusercontent.com/14358106/80425335-1c7f8500-88ec-11ea-8a00-2e4584449989.png)

- RealnessGAN - instead of hard labels 0 and 1 for GAN treat them as random variables A0 and A1. Seems to stabilize training as the authors were able to train DCGAN on 1024x1024. Proven theoretical guarantees of convergence. A0 and A1 was taken as different discrete distributions (so D had N output probabilities instead of single) [Real or Not Real, that is the Question](https://openreview.net/pdf?id=B1lPaCNtPB)

![RealnessGAN](https://user-images.githubusercontent.com/14358106/80428366-5784b700-88f2-11ea-850e-daeee64cc0df.png)

- [iclr.video](https://iclr.cc/virtual/poster_Hkxzx0NtDB.html) use classifier as energy function; helps to improve applications of generative models to target tasks (OOD detection, adversarial robustness, etc) [Your classifier is secretly an energy based model and you should treat it like one](http://www.openreview.net/pdf?id=Hkxzx0NtDB)

- To what extent we can manipulate generated image features (zoom in/out or shift or ...) [On the "steerability" of generative adversarial networks](http://www.openreview.net/pdf?id=HylsTT4FvB)

- physics-motivated model for videos (beautiful motivation, but works with only very simple systems of objects so far). The idea is to learn encoder from pixel space, Hamiltonian network (which rules the system state) and decoder from latent space back to pixel space. The system evolution is going by adjusting state +alpha * dt, where alpha is speed. In practice it is useless but I liked the idea and motivation. [Hamiltonian Generative Networks](http://www.openreview.net/pdf?id=HJenn6VFvB)

- visualization tool and (new) metrics to monitor/estimate convergence of GAN [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](http://www.openreview.net/pdf?id=HJeVnCEKwH)

## NAS

- The authors show that DARTS method overfits on validation data and do not generalize well on test data for several search spaces. The reason for poor generalization is highly correlated with sharpness of the minima. The sharpness of the minima can be efficiently estimated by computing largest singular value of full Hessian w.r.t. alpha on randomly taken validation batch. We can then track this metric (largest eigenvalue) and do early stopping (DARTS-ES) when it starts to grow too rapidly. DARTS-ES already increases performance, but this can be further improved. Authors show that regularization (e.g. L2 norm) stabilizes largest eigenvalue metric, thus they propose the following method. Whenever sharp increases of eigenvalue is observed, the model goes back to the last checkpoint before that behavious starts, and training continues with higher regularization. This approach improve generalization significantly in many settings, including original DARTS search space. [Understanding and Robustifying Differentiable Architecture Search](http://www.openreview.net/pdf?id=H1gDNyrKDS)

![robust darts takeaways](https://user-images.githubusercontent.com/14358106/80723246-12869d80-8b09-11ea-90c3-d51aaee8dba5.png)

- 1.5 GPU hours on cifar search (4-10 times faster than DARTS) - results better than DARTS [PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](https://openreview.net/pdf?id=BJlS634tPr)

- Early research about how to initialize Meta-networks (networks which emit other networks, e.g. NAS). Authors propose efficient initialization, but there are some restrictions in the search space. This is a very early research and acc to authors there are a lot of low hanging fruits in that direction. In [presentation](https://iclr.cc/virtual/poster_H1lma24tPB.html) authors very clearly explain that initialization is crusial as with bad initialization models will not converge at all. [Principled Weight Initialization for Hypernetworks](http://www.openreview.net/pdf?id=H1lma24tPB)

- [FasterSeg: Searching for Faster Real-time Semantic Segmentation](http://www.openreview.net/pdf?id=BJgqQ6NYvB)

- 10 NAS algorithms on search space of 15k architectures (all 15k trained) [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](http://www.openreview.net/pdf?id=HJxyZkBKDr)

- [NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search](http://www.openreview.net/pdf?id=SJx9ngStPH)

## Representation

- How to make model better generalize on unseen (related but different) domains by plugging special feature transformation layers into model [Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation](http://www.openreview.net/pdf?id=SJl5Np4tPr)

- Supervised learning still performs (much) better, but here are major improvements for unsupervised learning [Self-labelling via simultaneous clustering and representation learning](http://www.openreview.net/pdf?id=Hyx-jyBFPr)

- Representation learned progressively from last NN layer to first NN layer - have reasonable results and ability to control the generation [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](http://www.openreview.net/pdf?id=SJxpsxrYPS)

- reasonable experiments on EMNIST for varying symbols with large amount of control [Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)](http://www.openreview.net/pdf?id=rygeHgSFDH)

## Vision (surprise!)

- :question: Space2Vec - embedding for spacial locations (as far as I understood only for geolocations not for (x,y) on the image) -> use in classification [Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells](http://www.openreview.net/pdf?id=rJljdh4KDH)

## Layers

- superior results against for 10 most popular architectures replacing batchnorm with deconvolution. Can be efficiently calculated at a fraction of conv layer cost. Authors replaced batchnorms with network deconvolution and gain superior results on all architectures they tried. There is also a connection with brain (a speciall kind of vision operation). The intuition behind - the layer is decorrelating neighbouring pixels and channels so it is much easier for NN to learn. [Network Deconvolution](http://www.openreview.net/pdf?id=rkeu30EtvS)

![what if...](https://user-images.githubusercontent.com/14358106/80729217-92fccc80-8b10-11ea-8e98-0a8a1edaba3b.png)

![deconvolution operation](https://user-images.githubusercontent.com/14358106/80730943-e5d78380-8b12-11ea-804a-388e527760fb.png)

![deconvolution applied to image](https://user-images.githubusercontent.com/14358106/80731086-1f0ff380-8b13-11ea-8981-f7f5ab4b03b5.png)

![deconvolution algorithm](https://user-images.githubusercontent.com/14358106/80731699-eb819900-8b13-11ea-98c7-c60a6f741cad.png)


- if you somehow know that your network should know exact multiplication/addition of tensor elements (e.g. physics, math problems) [Neural Arithmetic Units](https://openreview.net/pdf?id=H1gNOeHKPS)

## NLP

- BLUE is finally sentenced to death [BERTScore: Evaluating Text Generation with BERT](http://www.openreview.net/pdf?id=SkeHuCVFDr)

![GPUs go brrr](https://user-images.githubusercontent.com/14358106/80547651-6c7c4b80-89c1-11ea-9956-72f6d9c7d7b4.png)

- Nuclear Sampling: instead of top-k most probable words take top-p (sum of top-m words probabilities >= p) [The Curious Case of Neural Text Degeneration](http://www.openreview.net/pdf?id=rygGQyrFvH)

- Unlikelihood (which *outperforms* nuclear sampling from above :point_up: significantly) - the idea is to penalize unlikely situations (negative examples which are either 1)random or 2)repeting words) [Neural Text Generation With Unlikelihood Training](http://www.openreview.net/pdf?id=SJeYe0NtvH)

- :grey_question: Controllable generation with language models [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](http://www.openreview.net/pdf?id=H1edEyBKDS)

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](http://www.openreview.net/pdf?id=r1xMH1BtvB)

![ELECTRA pretraining](https://user-images.githubusercontent.com/14358106/80499323-82fcb580-8975-11ea-87b3-8ea7a678630b.png)

- [Reformer: The Efficient Transformer](http://www.openreview.net/pdf?id=rkgNKkHtvB)

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](http://www.openreview.net/pdf?id=H1eA7AEtvS)

- tiny change in LSTM which make it better than your Transformer, die BERT! (ok, it won't but the idea is simple but working) [Mogrifier LSTM](http://www.openreview.net/pdf?id=SJe5P6EYvS)

- yes [Are Transformers universal approximators of sequence-to-sequence functions?](http://www.openreview.net/pdf?id=ByxRM0Ntvr)

- Transformer solves math problems much better than Wolphram alpha (with pretty straightforward approach) [Deep Learning For Symbolic Mathematics](http://www.openreview.net/pdf?id=S1eZYeHFDS)

- The main problem with text GANs (acc to authors) is that Discriminator easily overpowering Generator. To improve Generator training it is rewarded when current generated sentence is better than previously generated sentence [Self-Adversarial Learning with Comparative Discrimination for Text Generation](http://www.openreview.net/pdf?id=B1l8L6EtDS)

- unsupervised text style transfer [code](https://github.com/cindyxinyiwang/deep-latent-sequence-model) [paper](http://www.openreview.net/pdf?id=HJlA0C4tPS)

## Anomaly detection

- main idea is to detect anomaly regions as regions with high difference between original and AE-reconstructed image; solved by gradient minimization of Reconstruction loss (x_i) + ||x_i - x_orig||; looks like anomaly localization improves [Iterative energy-based projection on a normal data manifold for anomaly localization](http://www.openreview.net/pdf?id=HJx81ySKwr)

## Speed improvements

- Automatic search for equivalent (but simpler or faster) set of operations, which is useful for NN inference speed optimization [Deep Symbolic Superoptimization Without Human Knowledge](http://www.openreview.net/pdf?id=r1egIyBFPS)

- Autoregressive decoder speed up [Decoding As Dynamic Programming For Recurrent Autoregressive Models](http://www.openreview.net/pdf?id=HklOo0VFDH)

## Other (random topics)

- BackPACK - wrapper for pytorch to estimate several gradient statistics, works reasonably fast, but some features do not support branching and custom forward implementations [BackPACK: Packing more into Backprop](http://www.openreview.net/pdf?id=BJlrF24twB)

- how to learn from rule-based (automatically) generated markup - method can significantly improve performance [Learning from Rules Generalizing Labeled Exemplars](http://www.openreview.net/pdf?id=SkeuexBtDr)

- how to measure quality on test set with noisy labels? see :point_down: (exact formula in the paper) [Discrepancy Ratio: Evaluating Model Performance When Even Experts Disagree on the Truth](http://www.openreview.net/pdf?id=Byg-wJSYDS)

![discrepancy formula](https://user-images.githubusercontent.com/14358106/80544948-dba27180-89ba-11ea-9037-3704f61af177.png)

- New NN+tree end2end trained module for tabular data outperforms XGBoost and CatBoost in several datasets [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](http://www.openreview.net/pdf?id=r1eiu2VtwH)

- Decision Trees with criteria = linear model on each node with NN approximation. More robust (but most likely slower) [Locally Constant Networks](http://www.openreview.net/pdf?id=Bke8UR4FPB)

- [Neural tangents](https://github.com/google/neural-tangents) - library for infinite width NNs [paper](http://www.openreview.net/pdf?id=SklD9yrFPS)

![image](https://user-images.githubusercontent.com/14358106/80477440-98151c80-8954-11ea-979a-529d02de1642.png)

![resource requirements](https://user-images.githubusercontent.com/14358106/80477580-da3e5e00-8954-11ea-839c-d8da506761bc.png)

- exploring continuous game of life [Intrinsically Motivated Discovery of Diverse Patterns in Self-Organizing Systems](http://www.openreview.net/pdf?id=rkg6sJHYDr)

- amazing pranking [presentation](https://iclr.cc/virtual/poster_H1l_0JBYwS.html) - I hated the presenter very much sitting at home=)

