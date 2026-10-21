# Installation

## Installing Ais

Ais works on Windows and Linux systems, but not on macOS. To ensure Ais's requirements do not clash with other packages, we recommend creating a separate environment to install Ais in. Using anaconda prompt, for example, run:

```
conda create --name ais
conda activate ais
conda install python==3.10
conda install pip
pip install git+https://github.com/mgflast/Ais
```

Run Ais with either of these commands:

```
ais
ais-cryoet
```

## CUDA & TensorFlow

To enable processing on the GPU, TensorFlow must be set up to use CUDA. This can be a bit of a pain, as only particular combinations of versions of TensorFlow, CUDA, cuDNN, and protobuf (a Python package) tend to work. When installing Ais via pip, the versioning should be handled, but CUDA must still be manually installed. For instructions, see:

- [Installing CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Installing cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

Ais supports TensorFlow 2.8.2 to 2.21. The simplest setup is TensorFlow 2.21 (Python 3.10 to 3.13), which ships its own CUDA and cuDNN libraries, so that only the NVIDIA driver needs to be installed:

```
pip install "tensorflow[and-cuda]==2.21.*"
pip install ais-cryoet
```

On native Windows, TensorFlow 2.10 is the last version with GPU support.

TensorFlow 2.16 and later come with Keras 3. Models trained there are saved in the Keras 3 format, which this version of Ais reads with every supported TensorFlow, but which Ais 1.2.40 and earlier cannot read. To use such a model with an older Ais, convert it once with `ais convert model.scnm [converted.scnm]` in an environment with Keras 2: TensorFlow 2.15 or older, or a newer TensorFlow with the matching tf-keras installed (e.g. `pip install "tf-keras==2.21.*"`) and the environment variable `TF_USE_LEGACY_KERAS=1`. Models trained with TensorFlow 2.15 or older need no conversion.

When running Ais from within an IDE some paths may need to be manually specified. In PyCharm, add the path to the zlib `.dll` to the run configuration environment variables as follows to enable TensorFlow using the GPU:

```
LIBRARY_PATH=C:\Program Files\zlib123dllx64\dll_x64
```
