# Pix2Pix / CycleGAN
Experiments using Pix2Pix and CycleGAN frameworks to reconstruct images from sketches.

## Objective
The objective of the experiment is to test the viability and quality of the generated images for the sketch modeling task, by using the Pix2Pix and CycleGAN approaches.

## Experiment Page
The complete experiment page can be accessed on (PT-BR only): 

https://www.notion.so/vinitrevisan/Transforma-o-de-Dom-nio-afb4dd35130540de82cfb12f116c5eda

---

## Contents

***main.py***

File with the experiment core. Training, test and validation of the networks. Parameters of the experiment.

***networks.py***

File with the classes and functions to create the generators and discriminators used on the experiments.

***losses.py***

File with the functions used to evaluate the losses to train the GANs

***metrics.py***

File with the functions used to evaluate the quality metrics (FID, IS, L1, Accuracy).

***utils.py***

File with all utilities functions, such as plot control, image processing, and exception handling.

***misc/***

Folder with miscellaneous code used to prepare the images and generate the datasets.

***generalization_images_car_cycle/***

Folder with images used for the generalization test on the CycleGAN approach.

***generalization_images_car_sketches/***

Folder with images used for the generalization test on the Pix2Pix approach.

---

## Also part of this project

Autoencoders ([github](https://github.com/vinyluis/Autoencoders))

Pix2Pix-CycleGAN ([github](https://github.com/vinyluis/Pix2Pix-CycleGAN))

Unsupervised-GANs ([github](https://github.com/vinyluis/Unsupervised-GANs))

Main experiment page ([Notion [PT-BR]](https://www.notion.so/vinitrevisan/Estudos-e-Experimentos-4027d5e5256e4efc80fe1cd4dc018553))

