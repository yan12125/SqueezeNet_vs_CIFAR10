# SqueezeNet vs. CIFAR10 (in progress)
## Zac Hancock (zshancock@gmail.com)

### Updates
This project is in progress
Training was committed but the model is already different - I intend to train the model overnight with roughly 250 epochs to encourage
discussion about the accuracy-plateau, etc. 

**To Do List ~ ETA within a week**
- [X] Open repo, add notebooks, start README.md
- [X] Add visualize_cifar.ipynb notebook
- [ ] Upload updated deploy_squeezenet.ipynb notebook (~250 epochs?)
- [ ] complete README.md with new discussion and graphics


*The **squeezenet_architecture.ipynb** notebook is going to likely remain the same unless I fine tune some of the parameters or rearrange the layers (chiefly, where I downsample with max pooling). The actual SqueezeNet architecture is different than what I will refer to as 'Squeeze Net' so I encourage you to read the paper (cited below) and visit the [Deepscale/SqueezeNet github page](https://github.com/deepscale/squeezenet). Most of the modifications I made were to better suit the CIFAR-10 dataset, whereas the the original SqueezeNet was optimized for ImageNet-1k - the primary differences being input size and classes : 32.32.3 for CIFAR with 10 classes and 224.224.3 for ImageNet with 1000 classes.*

## The CIFAR-10 Data

This is a baseline data set of tiny 32 x 32 x 3 images with 10 classes. Because of its size, it was appropriate for my local machine. There are 50,000 training images and 10,000 training images.  

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/cifar_visual.JPG)

**25 Images from the CIFAR-10 dataset (notice: this only shows 9 of the 10 classes - no 'bird' shown).**

## The Model

Inspired by the 'SqueezeNet' architecture proposed by Forrest Iandola et al. (2016), created a smaller model for CIFAR-10 data set using similar components (fire module, etc). The basis of the **fire module** is shown below (Iandola 2016). Essentially, the fire module implements a strategy wherein it minimizes the input parameters by utilizing a 'squeeze layer' that only uses 1x1 filters. After the 'squeeze layer' is a series of both 1x1 and 3x3 filters in the 'expand layer'. The expand layer is then concatenated. 

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/fire_module.JPG)

```
def fire_mod(x, fire_id, squeeze=16, expand=64):
    
    # initalize naming convention of components of the fire module
    squeeze1x1 = 'squeeze1x1'
    expand1x1 = 'expand1x1'
    expand3x3 = 'expand3x3'
    relu = 'relu.'
    fid = 'fire' + str(fire_id) + '/'
    
    # define the squeeze layer ~ (1,1) filter
    x = layers.Convolution2D(squeeze, (1,1), padding = 'valid', name= fid + squeeze1x1)(x)
    x = layers.Activation('relu', name= fid + relu + squeeze1x1)(x)
    
    # define the expand layer's (1,1) filters
    expand_1x1 = layers.Convolution2D(expand, (1,1), padding='valid', name= fid + expand1x1)(x)
    expand_1x1 = layers.Activation('relu', name= fid + relu + expand1x1)(expand_1x1)
    
    # define the expand layer's (3,3) filters
    expand_3x3 = layers.Convolution2D(expand, (3,3), padding='same', name= fid + expand3x3)(x)
    expand_3x3 = layers.Activation('relu', name= fid + relu + expand3x3)(expand_3x3)
    
    # Concatenate
    x = layers.concatenate([expand_1x1, expand_3x3], axis = 3, name = fid + 'concat')
    
    return x

```

Using the fire module outlined above, the architecture was completed. Max Pooling happens after the very first convolution layer, followed by 4 fire modules. After the last fire module, 50% dropout is committed before the last convolution layer. Global pooling is committed right before softmax activation into 10 classes. The original SqueezeNet proposed by Iandola was much larger, but the CIFAR images are considerably smaller than ImageNet ~ additionally, my local machine could struggle with a larger model. 

```
def SqueezeNet(input_shape = (32,32,3), classes = 10):
        
    img_input = layers.Input(shape=input_shape)
    
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model
```

## Results

The first training of this architecture after 10 epochs resulted in near 50% training accuracy, without overfitting. This led the analyst to possibly retrain with as many as 250 epochs because the plateau of accuracy could be higher (it continued to increase all of 10 epochs). A later commit may include new weights for more epochs and training, but admittedly this will require cloud computing because my local machine will probably struggle. 

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/accuracy_and_loss.JPG)
**Results of the CNN versus the CIFAR-10**

## Citations

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

```
Author = Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally and Kurt Keutzer
Title = SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
Journal = {arXiv:1602.07360}
Year = 2016
```

CIFAR-10 Documentation ~
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

```
Author = Alex Krizhevsky
Title = Learning Multiple Layers of Features from Tiny Images
Year = 2009
```

