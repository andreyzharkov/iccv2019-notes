# iccv2019-notes
personal notes from iccv2019

## Augmentation

- Are adversarial robustness and common perturbation robustness independant attibutes

- 

## Semantic segmentation

- [detect unknown objects using optical flow] Towords segmenting everything that moves

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
