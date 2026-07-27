from setuptools import setup, find_packages
import shutil
# how to release:
# UPDATE VERSION IN 3 PLACES: Ais/core/config.py, setup.py, docs/conf.py

# push to pypi:
# python setup.py sdist
# twine upload dist/*

VERSION = '1.2.0'

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='Ais-cryoET',
    version=VERSION,
    packages=find_packages(),
    entry_points={'console_scripts': ['ais=Ais.main:main', 'ais-cryoet=Ais.main:main']},
    url='https://github.com/mgflast/Ais',
    license='GPL v3',
    author='mgflast',
    author_email='mgflast@gmail.com',
    description='Segmentation of cryo-electron tomography data - https://mgflast.github.io/Ais/ https://aiscryoet.org/',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['*.png', '*.glsl', '*.pdf', '*.txt']},
    include_package_data=False,  # weirdly, the above filetypes _are_ included when this parameter is set to False.
    install_requires=[
        "imgui>=2.0.0",
        "tensorflow>=2.8.0,<=2.11.1",
        "protobuf>=3.19.0,<=3.20.3",
        "glfw>=2.5.5",
        "PyOpenGL>=3.1.6",
        "numpy>=1.23.2,<2.0",
        "mrcfile>=1.4.3",
        "Pillow>=9.2.0",
        "scipy>=1.11.0",
        "tifffile>=2022.8.12",
        "dill>=0.3.5.1",
        "pyperclip>=1.8.2",
        "scikit-image",
        "matplotlib",
        "starfile",
        "tqdm"
    ]
)

