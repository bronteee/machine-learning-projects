# Final Project Plan: Exploratory Analysis of fMRI Signals and Stimuli Using S4 Architecture

Bronte Sihan Li, April 2023

## Introduction and Background

The study of artificial intelligence is incomplete without examining the intelligent agent itself, the human brain. fMRI, which has gained popularity in the recent years, is a non-invasive method to measure brain activity by measuring the blood oxygenation level dependent (BOLD) signal, which is a proxy for neural activity. As machine learning has advanced, it has allowed us to analyze fMRI data more easily and efficiently.

## Data Description

The BOLD5000 dataset is one of the largest fMRI datasets containing measurements from 3 subjects given 5000 image stimuli with comprehensive behavioral data where the image sets have significant alignment with current computer vision datasets including ImageNet, COCO, etc. For more details, see [4, 5](#references). The dataset is available on the BOLD5000 website [5](#references).

## Problem Statement

In the author's last project working with fMRI data using traditional methods of machine learning, the performance of the prediction model was very limited which may be attributed to both noisiness of the data and limitations of the model architectures. However, the fact that the classification achieved statistically significant accuracy suggests that there is some information in the fMRI data that can be used to predict the stimuli images and it will be interesting to see how deep learning and long sequence learning can help further the analysis of brain activity vs. stimuli. 

The S4 architecture (SSM), developed by Hazy Research Lab at Stanford, is a novel approach to incorporate long sequences of data into the neural network that may offer new insight on the fMRI data as brain signals are continuous and has state. There are a couple of goals for this project: one is to explore the S4 architecture and its performance on fMRI data using an annotated guide on the model [1, 3, 6](#references), analyzing the relationship between fMRI signals and stimuli images, the other (stretch goal) is to attempt to train a SSM neural network and examine its brain-likeness [7](#references).

## Proposed Methods and Solution

Ideas on how to approach the problem include but are not limited to:

* Train SSM neural network on standard computer vision datasets and evaluate performance
* Train SSM neural network with both fMRI data and stimuli image data and evaluate performance
* Brain score analysis of SSM neural network

## References
1. https://srush.github.io/annotated-s4/
2. https://github.com/HazyResearch/state-spaces/blob/main/configs/model/README.md
3. Gu, Albert et al. “Efficiently Modeling Long Sequences with Structured State Spaces.” ArXiv abs/2111.00396 (2021): n. pag.
4. Chang, N., Pyles, J.A., Marcus, A. et al. BOLD5000, a public fMRI dataset while viewing 5000 visual images. Sci Data 6, 49 (2019). https://doi.org/10.1038/s41597-019-0052-3
5. https://bold5000-dataset.github.io/website/
6. https://hazyresearch.stanford.edu/blog/2023-03-27-long-learning
7. Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like? Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J. Majaj, Rishi Rajalingham, Elias B. Issa, Kohitij Kar, Pouya Bashivan, Jonathan Prescott-Roy, Kailyn Schmidt, Daniel L. K. Yamins, James J. DiCarlo bioRxiv 407007; doi: https://doi.org/10.1101/407007