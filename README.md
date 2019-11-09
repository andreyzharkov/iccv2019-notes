# iccv2019-notes
personal notes from iccv2019

## Intro

Over 7,500 participants, 4 days of main conference + 60 workshops and 12 tutorials.
~1200 accepted papers (20% orals) on the main conference alone.

It was absolutely infeasible to track everything so I almost completely skipped the following topics
- Autonomous driving
- 3D, Point clouds
- Video analysis

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

- Are adversarial robustness and common perturbation robustness independant attibutes

- 

## Semantic segmentation

- [detect unknown objects using optical flow] Towards segmenting everything that moves

## Instance segmentation

## Object detection

- [Yolo prunning] SlimYOLOv3: narrower faster and better for real-time

## Text detection and recognition

- SNICER: Single noisy image denoicing and rectification for improving licence plate recognition

- State-of-the-art in action: unconstrained text detection



## Content generation, generative models

- 

- Dance Dance Generation: Motion Transfer for Internet Videos

- I bet you are wrong: gambling adversarial networks for semantic segmentation

- Markov decision process for video generation

-

## Compression

- [9/10] [Automated Multi-Stage Compression of Neural Networks](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.pdf) - tensor decompositions, two repetitive steps:
compression and fine-tuning, 10-15x compression rate with 1-2% metric drop (depend on dataset). [pytorch code](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.pdf)

- [??? TODO] [Workshop: Compact and Efficient Feature Representation and Learning in Computer Vision 2019](http://www.ee.oulu.fi/~lili/CEFRLatICCV2019.html)

## Other

- Real time aerial suspicious analysis (asana): system for identification and re-identification of suspicious individuals in crowds using the bayesian scatter-net hybrid network

- 

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
