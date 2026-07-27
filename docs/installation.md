# Installation

## Installing Ais

Ais works on Windows and Linux systems, but not on macOS. To ensure Ais's requirements do not clash with other packages, we recommend creating a separate environment to install Ais in. Using anaconda prompt, for example, run:

```
conda create --name ais
conda activate ais
conda install python==3.9
conda install pip
pip install git+https://github.com/mgflast/Ais
```

Run Ais with either of these commands:

```
ais
ais-cryoet
```

Alternatively, clone the project from [github.com/mgflast/Ais](https://www.github.com/mgflast/Ais) into an IDE of your choice.

## CUDA & TensorFlow

To enable processing on the GPU, TensorFlow must be set up to use CUDA. This can be a bit of a pain, as only particular combinations of versions of TensorFlow, CUDA, cuDNN, and protobuf (a Python package) tend to work. When installing Ais via pip, the versioning should be handled, but CUDA must still be manually installed. For instructions, see:

- [Installing CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Installing cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

When running Ais from within an IDE some paths may need to be manually specified. In PyCharm, add the path to the zlib `.dll` to the run configuration environment variables as follows to enable TensorFlow using the GPU:

```
LIBRARY_PATH=C:\Program Files\zlib123dllx64\dll_x64
```
