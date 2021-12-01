Towards trustworthy explanations with gradient-based attribution methods

Summary:

This project focuses on an analysis of various regularization strategies on a genomic dataset. The metric used to evaluate the performance of each strategy is the quality of model explanations, quantified as interpretability performance. Regularization strategies tested include manifold mixup, batch normalization, batch size, dropout, and learning rate. We find that while manifold mixup leads to improved performance at high network widths, its performance is surpassed by a specific combination of traditional regularizers such as dropout, learning rate, batch size, and batch normalization. This repository contains code to reproduce this specific combination of regularizers both with and without manifold mixup. To use this code, clone the repository, install the prerequisites found in requirements.txt, and run run.sh as a bash script. Results will be generated and logged to their own folder titled 'results'.

Link to paper: https://openreview.net/forum?id=LGgo0wPM2MF