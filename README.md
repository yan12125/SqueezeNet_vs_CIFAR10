# SqueezeNet vs. CIFAR10 (in progress)
## Zac Hancock (zshancock@gmail.com)

### Updates
This project is in progress - I uploaded my notebooks 01-08-2019
Training was committed but the model is already different - I intend to train the model overnight with roughly 250 epochs to encourage
discussion about the accuracy-plateau, etc. 

**To Do List ~ ETA within a week**
- [X] Open repo, add notebooks, start README.md
- [ ] Retrain with 250 epochs
- [ ] Upload updated deploy_squeezenet.ipynb notebook
- [ ] complete README.md with new discussion and graphics


The **squeezenet_architecture.ipynb** notebook is going to likely remain the same unless I fine tune some of the parameters or rearrange the layers (chiefly, where I downsample with max pooling). The actual SqueezeNet architecture is different than what I will refer to as 'Squeeze Net' so I encourage you to read the paper (cited below) and visit the [Deepscale/SqueezeNet github page](https://github.com/deepscale/squeezenet). Most of the modifications I made were to better suit the CIFAR-10 dataset, whereas the the original SqueezeNet was optimized for ImageNet-1k - the primary differences being input size and classes : 32.32.3 for CIFAR with 10 classes and 224.224.3 for ImageNet with 1000 classes. 

## Introduction

Inspired by the 'SqueezeNet' architecture proposed by Forrest Iandola et al. (2016), created a smaller model for CIFAR-10 data set using similar components (fire module, etc). 

## Citation

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

```
Author = Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally and Kurt Keutzer
Title = SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size
Journal = {arXiv:1602.07360}
Year = 2016
```
