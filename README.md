# [A Broad Study of Pre-training for Domain Generalization and Adaptation (ECCV 2022)](https://arxiv.org/pdf/2203.11819.pdf)
[Donghyun Kim](http://cs-people.bu.edu/donhk/), [Kaihong Wang](https://cs-people.bu.edu/kaiwkh/), [Stan Sclaroff](https://www.cs.bu.edu/fac/sclaroff/), and [Kate Saenko](http://ai.bu.edu/ksaenko.html).
#### [[Project Page]]()  [[Paper]](https://arxiv.org/pdf/2203.11819.pdf)
![Overview](images/fig1.jpg)



## Introduction

While domain transfer methods (e.g., domain adaptation, domain generalization) have been
proposed to learn transferable representations across domains, they are
typically applied to ResNet backbones pre-trained on ImageNet. Thus,
existing works pay little attention to the effects of pre-training on domain
transfer tasks. In this paper, we provide a broad study and in-depth analysis of pre-training for domain adaptation and generalization, namely:
network architectures, size, pre-training loss, and datasets. This repository contains PyTorch implementation of the single domain generalization experiments, which can be used as a baselin for domain transfer tasks including domain generalization and adpatation. This implementation is based on [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library).
