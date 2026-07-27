![Ais](docs/res/ais_banner.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/mgflast/Ais/blob/master/Ais/LICENSE.txt)
[![Downloads](https://img.shields.io/pypi/dm/Ais-cryoET)](https://pypi.org/project/Ais-cryoET/)
[![Documentation Status](https://readthedocs.org/projects/ais-cryoet/badge/?version=latest)](https://ais-cryoet.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/mgflast/Ais)

# Segmentation with Ais #
## Fast and user-friendly annotation and segmentation of cryo-electron tomography data using convolutional neural networks ##

This repository comprises a standalone version of Ais, the segmentation editor for cryoET. For the version integrated into the correlative microscopy data processing suite _scNodes_, see the [scNodes](https://www.github.com/bionanopatterning/scNodes) repository.

A timelapse video of the full workflow, from reconstructed tomograms to segmented volumes showing membranes, ribosomes, mitochondrial granules, and microtubuli, is available on our [YouTube channel](https://www.youtube.com/watch?v=2JIBVJf3kYQ&ab_channel=scNodes).

Contact: mlast@mrc-lmb.cam.ac.uk

### Installation ###
Ais works on Windows and Linux machines but not on MacOS. Install as follows: 
```
conda create --name ais
conda activate ais
conda install python==3.9
conda install pip
pip install git+https://github.com/mgflast/Ais
```

Then run using either of the following commands:
```
ais
ais-cryoet
```
#### Tensorflow & CUDA compatibility ####
Compatibility between Python, tensorflow, and CUDA versions can be an issue. The following combination was used during development and is known to work:

Python 3.9<br/>
Tensorflow 2.8.0<br/>
CUDA 11.8<br/>
cuDNN 8.6<br/>
protobuf 3.20.0<br/>

The software will work without CUDA, but only on the CPU. This is much slower but still reasonably interactive if the tomograms aren't too big (in XY). We do recommend installing CUDA and cuDNN in order for tensorflow to be able to use the GPU. See: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html for help installing CUDA and cuDNN. 

## Features ##
### Annotate any number of features ###
Open any number of tomograms and define any features you want to eventually segment. Step one is to annotate - very sparsely - in order to prepare training data for your networks. Usually, 5 to 10 minutes of annotation is enough to get started. After that, train an initial net, and check what needs further improvement.

https://github.com/user-attachments/assets/7831ad7d-811c-4846-8e0e-002487b54ade

### Train a network & watch it learn ###
All steps other than annotation can be done in the CLI. When training in the GUI, though, you can keep an eye on the network output as the training progresses. Not as fast as it would be in the CLI, but a nice way to see how the network is learning.

https://github.com/user-attachments/assets/9544569c-15d8-490b-a2fd-cf0f406d0dbd

### Model-assisted annotation ###
When networks still need a bit of improvement before you're ready to segment your data, *model assisted annotation* helps you quickly polish the training data. Screen the output of a model, copy it to the training annotations, and edit any mistakes that need fixing. 

https://github.com/user-attachments/assets/8333d05d-aa60-4e04-bf65-e147dc4ef05c

### Active contouring brush ###
Before using model-assisted annotation, turn on _flood_ mode to use the active contouring brush. Then play around with the filters and sensitivity, and watch the brush snap to the edges of your features automatically. 

https://github.com/user-attachments/assets/89519081-1eb0-4ff3-adb5-b2b3aad6ed3c

### Segment with multiple models ###
https://github.com/user-attachments/assets/69b40bc2-04f3-4c77-b1ad-9c5d07f40f9d

### Inspect results ###
After segmenting your data, you can visualize the results the Ais _rendering tab_. You can also set up picking jobs here, or if you've already ran 'ais pick' in the terminal you can inspect the resulting particle coordinates in the context of the tomograms.

https://github.com/user-attachments/assets/12aa1ea4-38eb-465c-8443-72f863a9c0bd

### Connect Ais to a Pom database ###
Ais integrates with [Pom](https://github.com/mgflast/Pom), a tool to present large cryoET datasets as searchable databases. Use Pom to organise the data, and Ais to mine it.

https://github.com/user-attachments/assets/32febf3e-fcba-4850-81ca-c4140a84b9d8

### A feature library helps you organise your work ###
If you're using Ais often, or are setting up a large new project where you plan to segment many different features, the _feature library_ is a useful way to organise your work. Name and style a feature once, then automatically grab those settings whenever you create a new annotation for that feature.

https://github.com/user-attachments/assets/7c8060ae-6a22-412e-b7d4-e5f430bf1524


###  ###cat pom/sub 

## See our other tools ##

<p align="center">
  <a href="https://github.com/mgflast/Pom"><img src="https://github.com/mgflast/Pom/raw/main/docs/res/pom_banner.png" width="49%"></a>
  <a href="https://github.com/mgflast/easymode"><img src="https://github.com/mgflast/easymode/raw/master/assets/easymode_banner.png" width="49%"></a>
</p>

Mart So-Last, 2026 | mgflast@gmail.com 