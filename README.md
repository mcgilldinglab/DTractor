# DTractor
A model for cell type deconvolution of spatial transcriptomics with deep neural network, transfer learning and matrix factorization.

## Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)
- [Tutorials](#Tutorials)
- [Credits](#Credits)
- [Contacts](#Contacts)

## Overview 
<img title="DTractor Overview" alt="Alt text" src="/figures/main.png">
To gain comprehensive insights into cell functions and intricate interactions, it is imperative to disaggregate ST at cell type resolutions. Existing methods generally fail to fully utilize the rich spatial localization information inherent in spatial transcriptomics. As a result, they demonstrate inconsistent accuracy across different datasets, lack robustness, are often limited to assessing only specific cell types, or rely on marker genes from single-cell references that can be affected by high dropout rates and gene expression fluctuations, highlighting the need for further development in this area. To address this, we introduce DTractor (named from Deep neural network, TRAnsfer learning and matrix faCTORization) to craft an innovative computational methodology. This approach aims to effectively deconvolute cell types by integrating scRNA-seq and ST data, ensuring clear and actionable interpretations. This is accomplished through deep neural network training on both datasets, employing transfer learning from scRNA-seq reference data to ST data in the latent space, and performing iterative matrix factorizations. DTractor is a robust, versatile, and computationally efficient tool for mapping diverse cell types across different tissue regions, adept at handling varying spot, cell, and gene counts. It effectively maps both small and large numbers of cell types, accommodating varying levels of granularity—from general cell types to fine subtypes—and works across different ST protocols. Additionally, DTractor provides the flexibility to incorporate spatial regularizations, such as leveraging spatial location information, and can be easily adapted to accommodate new ideas. By intricately deconvoluting spatial data and understanding the gene expressions at both spatial and single-cell levels, such in-depth insights have the potential to pave the way for advanced cellular research, enabling the development of more precise diagnostic tools and paving the groundwork for targeted therapeutic strategies tailored to individual cellular behaviors and interactions.

## Installation
First, install [Anaconda](https://www.anaconda.com/). You can find specific instructions for different operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Second, create a new conda environment and activate it:

```sh
# Clone the repository
git clone https://github.com/mcgilldinglab/DTractor.git
cd DTractor

# Create a conda environment
conda create -n dtractor python=3.9
conda activate dtractor
````

Then, install the version of PyTorch compatible with your devices by following the [instructions on the official website](https://pytorch.org/get-started/locally/). 

Installing the DOLPHIN Package
1. Standard Installation
   
  a. download DTractor from this repository, go to the downloaded DTractor package root directory, and use the pip tool to install)
```sh
pip install .
```

or 

  b. install the package directly from GitHub

```shell
pip install git+https://github.com/mcgilldinglab/DTractor.git
```


2. Developer Mode Installation
```sh
pip install -e .
```

Validate That DTractor Is Successfully Installed
```python
import DTractor
```



## Usage

Import the package and use the functions as shown in the example notebooks. 
Data can be downloaded from [Google Drive](https://drive.google.com/file/d/1REJuo0juOS85F6VNS7rw4nt8BttZ3Xm0/view?usp=sharing)

Use the all in one `DTractor_main`. Please read our [Examples](example/pdac_run.ipynb) for details. 
```python
from DTractor import DTractor_main

# Define Reference single-cell dataset
adata_ref = "adata_ref_13402.h5ad"
adata_vis = "adata_vis_13402.h5ad"

dtractor = DTractor_main(adata_vis, adata_ref) #set up parameters
dtractor.run() #train model and visualize
```

## Tutorials
### Running DTractor on pdac dataset and visualization of results
[pdac_run.ipynb](example/pdac_run.ipynb)


## Credits
DTractor is jointly developed by [Jin Kweon](https://github.com/yjkweon24), [Chenyu Liu](https://github.com/theguardsgod), [Gregory Fonseca](https://www.mcgill.ca/expmed/dr-gregory-fonseca-0), and [Jun Ding](https://github.com/phoenixding) from McGill University.

## Contacts
Please don't hesitate to contact us if you have any questions and we will be happy to help:
* jin.kweon at mail.mcgill.ca 
* jun.ding at mcgill.ca
