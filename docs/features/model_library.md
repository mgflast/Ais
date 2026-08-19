# The model library

Ais comes with a _network library_ of multiple different neural network architectures. These differ in depth, number of parameters, dimensionality (2D, 2.5D, or 3D compatible), and in design — VGG-style stacks, UNets, and one GAN. When you're training a new network, it is often useful to start with a small net and only switch to a larger one when you have more training data to support it. This is because the size of the training data needs to be appropriate for the size of the network - a 100 million parameter network trained with just 10 samples will not be very useful.

## Available architectures
The network library currently contains 15 different architectures. They can be split into a couple of groups:

- **VGGNet S / M / L / X** — VGG-style stacks of convolution and pooling layers, in increasing size. VGGNet M is the default, and a good first choice for most features.
- **UNet S / L** — standard UNets.
- **Eman2, InceptionNet, ResNet** — a few classic designs, originally included for comparison to other work. These can be interesting to play around with, but we do not recommend using them generally.
- **ezm-2d-dice, ezm-2d-bxe** — the easymode backbone: deeper and wider UNets, trained with a masked, combined dice and binary cross-entropy loss. The two variants differ in the weighting of the loss components. 
- **ezm-3d, ezm-3d-bxe** — the easymode 3D networks — also UNets, but built from 3D convolutions.
- **ezm-3d-filament** — a special case of the easymode 3D networks, used for filaments. Rather than directly predicting labels, this architecture predicts a distance map and then uses a fixed-weight layer to draw fixed shape cylindrical labels. This network can output genuine 3D filament shapes, even when trained with 2D annotated slices.
- **Pix2pix** — a generative adversarial network, which is architecturally the most unique.

## 2D, 2.5D, and 3D
Three input/output geometries exist in Ais: slice in, slice out (2D); slab in, slice out, where the slab depth is treated as image channels rather than as a spatial dimension (2.5D, 2D convolutions only); and slab in, slab out, using true 3D convolutions (3D). The same architectures handle 2D and 2.5D — the box depth set during training decides which you get. Only networks with `3d` in the name are 3D.

Annotation in Ais is all in 2D. We prefer annotating in 2D, because annotation in 3D is far more time consuming. Even though annotations are 2D, you can still train 3D or 2.5D networks in Ais. This is done via loss masking and positional jitter applied during training: the single annotated slice supervises the output at a varying depth within the slab, so that the network learns to predict at every slab position rather than only at the centre.

When extracting data in the GUI or with `ais extract -depth` you can choose to extract just a 2D slice, or a 3D slab. Feeding 3D slabs into 2D networks gets you a 2.5D network. 3D slabs into `3d` networks gets your a proper 3D network. 2D slices into `3d` networks will not work.

## Choosing one
If you want to generate training data rapidly using model-assisted annotation, it can be useful to choose a small model for the initial training. VGGNet M, the default model, is a good option. Only choose a larger network - VGGNet L, UNet L, or ezm-2d* - when you've prepared a larger training dataset (rule of thumb: 100+ boxes of 128x128 pixels).

Other than making sure the training data size and network size are approximately compatible, the best way to choose a network is by comparison. Use the same data to train two different architectures, then open the resulting networks in Ais and inspect their output.

## Custom architectures
You can also add your own custom architectures to Ais. Either install them from the GUI: `Settings > Model settings > Model library > Install a model` or by copying the model .py file to the user models directory (`~/.Ais/models` on Linux, `C:\Users\<you>\.Ais\models` on Windows). For more information, see [Custom model architectures](custom_architectures.md).
