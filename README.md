# iccv2019-notes
personal notes from iccv2019

## Intro

Over 7,500 participants, 4 days of main conference + 60 workshops and 12 tutorials.
1075 accepted papers (10% orals) on the main conference alone.

It was absolutely infeasible to track everything so I almost completely skipped the following topics
- Autonomous driving
- 3D, Point clouds
- Video analysis
- Computer Vision in medical images
- Captioning, Visual Grounding, Visual Question Answering

I spend a few time to:
- Domain adaptation, Zero-shot, few-shot, unsupervised, self-supervised, semi-supervised. **TL;DR** motivation - to learn as fast as humans with less/no data by few examples. In practice it still works poor enough and can not compare to supervised methods. **At the current stage we can not do this successfully enough, but when we will it will be a giant step forward.**
- Knowledge distillation, federated learning. **TL;DR** many papers have controversal results - sometimes it works, sometimes don't, sometimes very useful, sometimes useless. **You can try but do not expect much**
- Deepfakes in images and videos. **TL;DR** you can not completely trust any digital image/video anymore. There are huge movement in the area and already several datasets present. The problem is - when you know the "deepfake attack" method and trained on on the data which was produced with this method you can take ~70-95% accuracy (which is itself not much), but *when you don't know the method your deepfake detector may be close to random (50%)*

I took a closer look on:
- Semantic and instance segmentation, object detection
- New architectures, modules, losses, augmentations, optimizetion methods
- Neural architecture search
- Interpretibility
- Text detection and recognition
- Network compression
- GANs, style transfer

## Augmentation

- [7/10][They are independant] [Are adversarial robustness and common perturbation robustness independant attibutes](https://arxiv.org/abs/1909.02436)

- CutMix: Regularizing Strategy to Train Strong Classifiers with Localizable Features

- Online Hyper-parameter Learning for Auto-Augmentation Strategy

## Modules

- [10/10][OctConv is both faster and more accurate; drop-in replacement for vanilla conv] [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049). *The trick is that the final architecture have to be optimized (in terms of framework matrix operations) otherwise it will be actially **slower**. Gives up to 30% speed up with better accuracy*. The idea of the paper is to explicitly decompose features into high-frequent (H, W, C_h) and low-frequent (H // octave, W // octave, C_l) and process them separately than exchange the information.

- [9/10][Suspicious layer which surprisingly improves both accuracy and speed and is a drop-in replacement for vanilla convolution] [Dynamic Multi-scale Filters for Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf) Replace vanilla conv with the following 2-branch structure: the first branch computes KxK kernel via adaptive_pool(KxK) -> conv1x1; the second branch applies 1x1 conv to features; then 2 branches merge via depthwise conv **with kernel computed from the first branch** and after that additional 1x1 conv. Ablation study shows that it can give +7% mIoU compared to vanilla conv. Now why it's suspicious? The computed kernel's top left element is essentially taken from image top left part, and the same for bottom right kernel element (it's essentially image bottom right part). And this **very different** elements are applied to very similar local features. My intuition fails to explain why it make sense, maybe we have to add additional global pooling for the kernel and firstly convolve kernel with it.

- [8/10][Fast and efficient feature upsampling with very little overhead] [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188)

- [4/10][Learned pooling works slightly better but slower] [LIP: Local Importance-based Pooling](https://arxiv.org/pdf/1908.04156.pdf)

- [4/10][Learned pooling with global features; slightly better but slower] [Global Feature Guided Local Pooling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kobayashi_Global_Feature_Guided_Local_Pooling_ICCV_2019_paper.pdf)

- [5/10][Local visual attention in the autoregressive models order] [AttentionRNN: Structured Spatial Attention Mechanism](https://arxiv.org/pdf/1905.09400.pdf)

## Semantic segmentation

- [10/10][Non-uniform downsampling of high-resolution images] [Efficient segmentation: learning downsampling near semantic boundaries](https://arxiv.org/pdf/1907.07156.pdf). Network for creation of non-uniform downsampling grid aimed to increase space for semantic boundaries. The results are reasonably better than uniform downsampling. 3-steps: 1)non-uniform downsampling (image is downsampled to *very small* resulution (32x32 or 64x64 for example), on this resolution downsampling network is trained, ground truth are derived from a reasonable optimization problem on ground truth segmentation map) == very fast stage 2)main segmentation network runs on non-uniform downsampled image 3)the result is upsampled (which can be done as we know downsampling strategy)

### Adversarial training

- [9/10][Instead of using **global** discriminator between ground truth and predicted segmentation map they use "gambler" which predicts from (image, predicted segmap) to CE-weights to maximize sum(weights * CELoss)] [I Bet You Are Wrong: Gambling Adversarial Networks for Structured Semantic Segmentation](http://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Samson_I_Bet_You_Are_Wrong_Gambling_Adversarial_Networks_for_Structured_ICCVW_2019_paper.html) Instead of using **global** discriminator between ground truth and predicted segmentation map they use "gambler" which predicts from (image, predicted segmap) to CE-weights to maximize sum(weights * CELoss). Seem to improve perfornamce a lot compared to previous adversarial training approaches. Additional benefit is that gambler does not see GT so it is less sensitive to errors in GT

### Context aggregation

- [9/10][SOTA on cityscapes-val, proposed global-local context module to aggregate multidimensional features] [Adaptive Context Network for Scene Parsing](https://arxiv.org/pdf/1911.01664.pdf)

- [6/10][Self-attention on ASPP features flattened + concated] [Asymmetric Non-local Neural Networks for Semantic Segmentation](https://arxiv.org/pdf/1908.07678.pdf) Instead of global self-attention (which is very costly) they 1)use ASPP 2)flatten all ASPP maps 3)concat the resulted 1x1 maps 4)use attention on this concated features (which means that you can select 0.1 * global (1x1) pool + 0.3 * 2x2pool[0,0] + 0.01 * 2x2pool[0,1] + ...). The module improves final metric obviously.

### Make use of class prior

- [8/10][Per class centers computation (based on coarse segmentation) + attention on them -> fine segmentation] [ACFNet: Attentional clas feature network for semantic segmentation](https://arxiv.org/pdf/1909.09408.pdf)

### Make use of boundaries

- [8/10][Separate (chip) shape stream from image gradients and dual-task learning (shape + segmentation)] [Gated-SCNN: Gated Shape CNN for Semantic Segmentation](https://arxiv.org/pdf/1907.05740.pdf). 1)very cheap 3-layer shape stream which accepts image gradients + 1st layer CNN features and exchanges information with the main backbone via gating mechanism 2)dual loss (edge detection + semantic segmentation) + consensus regularization penalty (checks that semantic segmentation output is consistant with predicted edges)

- [8/10][Another approach for using boundary: first, learn boundary as N+1' class then introduce UAGs and some crazy staff] [Boundary-Aware Feature Propagation for Scene Segmentation](https://arxiv.org/pdf/1909.00179.pdf)

### Salient object detection (note that everyone exploits edges & boundaries in some way)

- EGNet: Edge Guidance Network for Salient Object Detection

- Selectivity or Invariance: Boundary aware Salient Object Detection

- Stacked Cross Refined Network for Edge-aware Salient Object Detection

### Other

- [6/10][Detect unknown objects using optical flow] [Towards segmenting everything that moves](https://arxiv.org/abs/1902.03715)

- [5/10][Typically works better and established SOTA, but obviously slower, nothing surprising] [Recurrent u-net for resource constrained segmentation](https://arxiv.org/pdf/1906.04913.pdf). Recurrence in several layers close to lowest-resolution ones.

- [???][Reformulate loss for convex objects; *I didn't understand that; looks like computational geometry thing can't say how useful it is with NNs*] [Convex Shape Prior for Multi-object Segmentation Using a Single Level Set Function](http://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Convex_Shape_Prior_for_Multi-Object_Segmentation_Using_a_Single_Level_ICCV_2019_paper.pdf)

## Instance segmentation

- [9/10][Learn prototypes and coefficients to combine them; can be 3-10x faster than MaskRCNN and have comparable accuracy] [YOLACT Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf) Each anchor predicts bbox + classes + **prototypes weights**. The separate branch predicts prototypes.

- [9/10][Backbone + point proposal -> mask of the object with point] [AdaptIS: Adaptive Instance Selection Network](https://arxiv.org/pdf/1909.07829.pdf) Proposed network is capable of generating instance mask by specifying point on that instance. Backbone extracts features. Features + point -> small net with AdaIN (where norms are computed from point info) -> instance mask. To get all objects on the image authors trained separate "point proposal" branch which is trained after everything else is frozen and predicts binary label "will point be good for object mask prediction?". From this branch top k% points are sampled and used for predicting objects.

## Object detection

- [7/10][Dense object detection by simply predicting bbox coordinates. Simple and efficient.] [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf) Directly predict 4D distances to object (top, left, bottom, right) in each foregroung pixel + NMS -> SOTA + simplisity + integration with depth prediction + no anchors computation (i.e. IoU) and anchor hyperparameters. Details: 1)each object is predicted on only one feature map (on training) and on testing in case of multiple resolutions alarmed for an object only the smallest is chosen; 2)propose "centerness" branch which predicts normalized ([0, 1]) distance between pixel and center, this branch is used in NMS multiplied by classification.

- [6/10][Global scale predicted in each resnet block, dilations are selected based on that] [POD: Practical object detection with scale-sensitive network](https://arxiv.org/pdf/1909.02225.pdf)

- [6/10][To improve detection of different scales objects replace some convs with 3 conv branches with same params but different dilations] [Scale-aware Trident Network for Object Detection](https://arxiv.org/pdf/1901.01892.pdf)

- [7/10][Change of target - ~~bbox~~ reppoints - arbitrary points whose circumference locates object accurately] [RepPoints: Point set Representation for object detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_RepPoints_Point_Set_Representation_for_Object_Detection_ICCV_2019_paper.pdf) These reppoint may be iteratively refined in prediction, they are learn by localization and classification losses.

- [6/10][Cycle-gan (clean image<->image with pathology) on small regions specified by masks] [Generative Modelling for small data object detection](https://arxiv.org/pdf/1910.07169.pdf)

- [5/10][2-stage detector with smaller backbone - works close-to-realtime on GPU][ThunderNet: Towards real time generic object detection on mobile device](https://arxiv.org/pdf/1903.11752.pdf)

## Text detection and recognition

- SNICER: Single noisy image denoicing and rectification for improving licence plate recognition

- State-of-the-art in action: unconstrained text detection

- Convolutional character networks

- Large-scale Tag-based Font retrieval with Generative Feature Learning

- Chinese Street View Text: Large-scale Chinese Reading and partially supervised learning

- TextDragon: An End-to-End Framework for Arbitraty Shaped Text Spotting

- Efficient and accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

- What's wrong with Scene Text Recognition Model Comparations? Dataset and Model analysys

- Towards unconstrained text spotting

- Controllable artistic text style transfer via shape-matching GAN

## Content generation, generative models, GANs, style transfer

- [10/10 BEST PAPER ICCV2019] [SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164) Use generators and discriminators on multiple resolutions and train on single image patches. Multiple applications without additional training including super resolution, image editing, single image animation, paint2image. [video](https://www.youtube.com/watch?v=Xc9Rkbg6IZA)

- [8/10][InGAN: Capturing and Retargeting the "DNA" of a Natural Image](http://www.wisdom.weizmann.ac.il/~vision/ingan/resources/ingan.pdf) GAN trained on patches of a single image and able to produce similar images of different shapes.

- [6/10][Single net adversarial attack for multiple target classes] [Once a MAN: Towards Multi-target attack via learning multi-target adversarial network once](https://arxiv.org/abs/1908.05185). Single model to produce adversarial examples towords any class (surprisingly, all the previous works use one model for one class, in this work - one work for all classes)

- [8/10][FUNIT: Few-shot Unsupervised Image-to-Image translation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf)

- [7/10][Lifelong GAN: Continual learning for Conditional Image Generation](https://arxiv.org/abs/1907.10107) With access to previous models and new data the task is to be able to produce both new and old classes. 

- [8/10][PuppetGAN: Cross-domain Image Manipulation by Demonstration](http://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf) Manipulate separate attributes of an image (e.g. mouth, rotation, lightning, etc) from target image

- [7/10][Couple of tricks to make "aging" more personalized] [S2GAN: Sharing Aging Factors Across Ages and Sharing Aging Trends Among Individuals](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_S2GAN_Share_Aging_Factors_Across_Ages_and_Share_Aging_Trends_ICCV_2019_paper.pdf)

- [7/10][User edits image a little in sketch in certain place -> realistic edited image] [sc-fegan: face editing generative adversarial network with user's sketch and color](https://arxiv.org/pdf/1902.06838.pdf)

- [10/10][Using pretrained GAN adapt for new classes and domains (even for 100 samples dataset) by training **only batch statistics**] [Image generation from small datasets via Batch Statistic Adaptation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noguchi_Image_Generation_From_Small_Datasets_via_Batch_Statistics_Adaptation_ICCV_2019_paper.pdf) With large pretrained generator (e.g. BigGAN) train only BatchNorm params (gamma and beta) and that's it - works even on very small datasets, results looks very good!

### GAN Training Stability improvements

- [9/10][Increase of stability + SOTA GAN metrics; different term for WGAN to optimize quadratic wassershtein distance] [Wassershtein GAN with quadratic transport cost](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Wasserstein_GAN_With_Quadratic_Transport_Cost_ICCV_2019_paper.pdf)

- [9/10][Spec-regularization >> spec-norm (in terms of both stability an final results)] [Spectral regularization for combating mode collapse in GANs](https://arxiv.org/abs/1908.10999)

### Video synthesys

- [9/10] Few-Shot Adversarial Learning of Realistic Neural Talking Head Models [video](https://youtu.be/P2uZF-5F1wI)

- Markov decision process for video generation

source person -> pose; target person + source pose -> synthesys
- Dance Dance Generation: Motion Transfer for Internet Videos

- Everybody Dance Now (University of California) [video](https://youtu.be/PCBTZh41Ris?t=158)

### Image extension

(also SinGAN and InGAN)

- Boundless: Generative Adversarial Network for Image Extension

- Very Long Natural Scenary Image Prediction by Outpaining

### Style transfer

- [examples **really looks** like simple color transform] Photorealistic style transfer via Wavelet Transforms

- A closed-form solution to universal style transfer

- Understanding whitening and coloring transform for universal style transfer

- [5/10] [Style transfer on entire image + semantic segmentation masks = style transfer for selected object classes] [Class-based styling: real-time localized style transfer with semantic segmentation](https://arxiv.org/abs/1908.11525)

### Fashion, clothes try-on

In general all these methods still work quite poorly, but they work somehow

- FW-GAN (Flow navigated warping gan for video virtual try-on)

- Personalized Fashion Design (Cong Yu et al)

## Neural Architecture Search

- [9/10][Sampling from random graph model gives results compared with current SOTA-NAS, which means that all that the current NAS do is not really better than random search] [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569)

- [8/10][Most of classification loss functions can be represented from parametrized loss function with 2 params, authors tried to find the optimal for these params][AM-LFS: AutoML for Loss Function Search](https://arxiv.org/pdf/1905.07375.pdf)

- [9/10][Have improvements vs handcrafted GANs on Cifar10 in terms of IS] [AutoGAN: Neural architecture search for generative adversarial networks](https://arxiv.org/pdf/1908.03835.pdf)

- [8/10][Evaluator which predicts how likely model will have lower validation score] [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733)

## Compression

**My knowledge of compression techniques is quite limeted so do not really trust these quality marks**

- [8/10] [Automated Multi-Stage Compression of Neural Networks](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.pdf) - tensor decompositions, two repetitive steps:
compression and fine-tuning, 10-15x compression rate with 1-2% metric drop (depend on dataset). [pytorch code](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.pdf)

- [8/10][4bit quantization + finetuning] [DSConv: Efficient Convolutional Operator](https://arxiv.org/pdf/1901.01928.pdf)

- [6/10][Yolo compression success story, known techniques applied properly] SlimYOLOv3: narrower faster and better for real-time

- Accelerate CNN via Recursive Bayesian Pruning

- [6/10][Speed-quality tradeoff without retraining; but the results are worse than SOTA] [adaptive inference cost with convolutional neural mixture models](https://arxiv.org/pdf/1908.06694.pdf) The idea is to work with mixture of nets (each layer may be applied or not applied). Inference cost is O(N*(N-1)/2)) where N is number of layers - which is relatively slow. In pruning we omit some layers thus having some speedup. The main benefit is that net does not need to be retrained, but the approach seems complicated to implement and works worse in quality compared to SOTA.

- [??? TODO] [Workshop: Compact and Efficient Feature Representation and Learning in Computer Vision 2019](http://www.ee.oulu.fi/~lili/CEFRLatICCV2019.html)

## Anomaly detection

- Real time aerial suspicious analysis (asana): system for identification and re-identification of suspicious individuals in crowds using the bayesian scatter-net hybrid network

- Detecting the unexpected by image resynthesis (anomaly detection)

- memorizing normality to detect anomaly: memory-augmented deep autoencoder for unsupervised anomaly detection

# Workshops
## Workshop: Image and Video Synthesis: How, Why and What If?
-[decompose into parts] hierarchical text2image (bbox->mask->image)
-generate by modifying part of the image
-[unsupervised] landmark autoencoder
-[small trick] best-of-many objective for diversity

-[early stage research] decompose gans according to physics (e.g. background + foreground). in general image = shape + textures + lightning
-[huge success] correlation in gans between objects and activations in some neurons 
(turning on/off some neuron activation we can edit produced images). 
correlation can be found automaticly, David Bau


# Early notes (27-28 Oct)

## Workshop: Multi-Discipline Approach for Learning Concepts - Zero-Shot, One-Shot, Few-Shot and Beyond

Invited talk (author http://www.ceessnoek.info/)

few-shot object detection without annotation (his ICCV2019 paper) (no need for bbox ground truth, only object presences)

object2action (detect object in every frame and estimate their movement?) Localize actions
+no need for videos (zero-shot == no videos)
https://arxiv.org/pdf/1510.06939.pdf

adding language avareness (objects are not really independant, if there is a football player, there is likely a ball)
action tricks:
- >penalize objects thta are not unique for an action
- >promote objects that have high uniqueness among action regardless theri relation to the action (leaks, lol)

Sportlight speedrun

- Enhancing Visual Embeddings through Weakly Supervised Captioning for Zero-Shot Learning
- ProtoGAN: Towards Few Shot Learning for Action Recognition
- Picking groups instead of samples: A close look at Static Pool-based Meta-Active Learning
- Adversarial Joint-Distribution Learning for Novel Class Sketch-Based Image Retrieval
- Weakly Supervised One Shot Segmentation
- Object Grounding via Iterative Context Reasoning

## Workshop: Low Power Computer Vision

- MUSCO: Multi-Stage Compression of neural networks (skoltech oseledets)


## Workshop: Image and Video Synthesis: How, Why and What If?
-hierarchical text2image (bbox->mask->image)
-generate by modifying part of the image
-landmark autoencoder
-best-of-many objective for diversity

-decompose gans according to physics (e.g. background + foreground). in general image = shape + textures + lightning
-correlation in gans between objects and activations in some neurons (turning on/off some neuron activation we can edit produced images). correlation can be found automaticly, David Bau

## TODO

- ~~Learning lightweighted LANE Detection CNNs by self-attention distilation~~

- continual learning by asymmetric loss approximation with single-side overestimation

### Sounds
- the sound of motions
- recursive visual sound separation using minus-plus net

### Imbalanced classes
- Gaussian margin for max-margin class imbalanced learning

- generative adversarial minority oversampling



- Local Relation Networks for Image Recognition


### Knowledge distillation
- Be Your Own Teacher: Improve the performance of CNN via Self-distillation

- On the Efficacy of Knowledge Distillation

### Self-supervised
- Scaling and benchmarking self-supervised visual representation learning
- S4L



- [ransac-like to fit arbitraty figures] Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm

- Noise flow: noise modeling with conditional normalizing flows

- Rethinging Imagenet Pre-training



- Deep Self-learning From Noisy Labels

- Selective Sparse Sampling for Fine-Grained Image Recognition

- ~~Dynamic anchor feature selection for single shot object detection~~

- VideoBERT: A Joint model for Video and Language Representation Learning

- PR Product: A substitute for inner product in neural networks

### Losses

- Anchor Loss: Modulating Loss Scale Based on Prediction

- SoftTripletLoss: Deep metric learning without triplet sampling

### Interpretability

- Explaining Neural Networks Semantically and Qualitatively

- fooling network interpretation in image classification

- Seeing what a GAN cannot generate



- Deep Meta Metric Learning

- [slow net on 1/N frames, fast net on (N-1)/N frames] Slow-Fast Networks for Video Recognition

- ~~Transductive Learning for Zero-shot Object Detection~~



- Self-training with progressive augmentation for Unsupervised Person Re-identification

- Learning to paint with model-based deep reinforcement learning

- Joint Demosaicing and Denoising by Fine-tuning of Bursts of Raw Images

- Improving CNN Classifiers by Estimating Test-time Priors

### Clustering

- Subspace structure-aware spectral clustering for robust subspace clustering

- Invariant information clustering for unsupervised image classification and segmentation

- GAN-Tree: An incrimentally Learned Hierarchical Generative Framework for Multi-Modal Data Distributions

- Deep Comprehensive Maining for Image Clustering

### Human unsertainty for training

- Human unsertainty makes classification more robust

- (force machines look same regions as humans helps, but the annotation cost?) Taking a HINT: Leveraging Explanations to Make Vision and Language Models More Grounded

### Motion in the dark

- Seeing Motion in the dark
- Learning to see moving objects in the dark

### Other (random)

- Joint Acne Image Grading and Counting via Label Distribution Learning

- [comic colorization] Tag2Pix: Line Art Colorization Using Text Tag With SECat and Changing Loss

# Other notes:

- Book "Explainable AI: Interpreting, explaining and visualizing deep learning"

- Are dilated convolutions used in sequential modeling? cnns in general?
