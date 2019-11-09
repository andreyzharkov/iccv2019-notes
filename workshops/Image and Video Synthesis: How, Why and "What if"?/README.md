# Image and Video Synthesis: How, Why and "What if"?

[homepage](https://sites.google.com/berkeley.edu/iccv-2019-image-and-video-syn)

## TL;DR

- hierarchical text2image (bbox->mask->image)
- generate by modifying part of the image
- landmark autoencoder
- best-of-many objective for diversity
- decompose gans according to physics (e.g. background + foreground). in general image = shape + textures + lightning
- correlation in gans between objects and activations in some neurons (turning on/off some neuron activation we can edit produced images). correlation can be found automaticly

## [8/10] Generative models as data visualization (Phillip Isola)

slides in this repo

TL;DR - traversing latent space of gan as an embedding to produce more/less memorable images, zoom in and out, etc. **Works for arbitrary images, not for a specific GAN model**

## [9/10] The Structure and Interpretation of a GAN (David Bau)

slides in interpretibility wowkshop

correlation in gans between objects and activations in some neurons (turning on/off some neuron activation we can edit produced images). correlation can be found automaticly

producing a door in a reasonable place works fine, but producing door in the clouds would not work

cheese hypothesis - regularized gans are more representative

- [GAN Dissection](http://gandissect.csail.mit.edu/)
- [GAN Paint](http://ganpaint.io/)
- [Seeing what gan cannot generate](http://ganseeing.csail.mit.edu/)
- [Davig Bau homepage](https://people.csail.mit.edu/davidbau/home/)
