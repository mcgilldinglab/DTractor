# DTractor
A model for cell type deconvolution of spatial transcriptomics with deep neural network, transfer learning and matrix factorization.

## Overview 
<img title="DTractor Overview" alt="Alt text" src="/figures/main.png">
To gain comprehensive insights into cell functions and intricate interactions, it is imperative to disaggregate ST at cell type resolutions. Existing methods generally fail to fully utilize the rich spatial localization information inherent in spatial transcriptomics. As a result, they demonstrate inconsistent accuracy across different datasets, lack robustness, are often limited to assessing only specific cell types, or rely on marker genes from single-cell references that can be affected by high dropout rates and gene expression fluctuations, highlighting the need for further development in this area. To address this, we introduce DTractor (named from Deep neural network, TRAnsfer learning and matrix faCTORization) to craft an innovative computational methodology. This approach aims to effectively deconvolute cell types by integrating scRNA-seq and ST data, ensuring clear and actionable interpretations. This is accomplished through deep neural network training on both datasets, employing transfer learning from scRNA-seq reference data to ST data in the latent space, and performing iterative matrix factorizations. DTractor is a robust, versatile, and computationally efficient tool for mapping diverse cell types across different tissue regions, adept at handling varying spot, cell, and gene counts. It effectively maps both small and large numbers of cell types, accommodating varying levels of granularity—from general cell types to fine subtypes—and works across different ST protocols. Additionally, DTractor provides the flexibility to incorporate spatial regularizations, such as leveraging spatial location information, and can be easily adapted to accommodate new ideas. By intricately deconvoluting spatial data and understanding the gene expressions at both spatial and single-cell levels, such in-depth insights have the potential to pave the way for advanced cellular research, enabling the development of more precise diagnostic tools and paving the groundwork for targeted therapeutic strategies tailored to individual cellular behaviors and interactions.

## Installation

## Tutorials

## Credits
DTractor is jointly developed by [Jin Kweon](https://github.com/yjkweon24), Chenyu Liu, [Gregory Fonseca](https://www.mcgill.ca/expmed/dr-gregory-fonseca-0), and [Jun Ding](https://github.com/phoenixding) from McGill University.

## Contacts
Please don't hesitate to contact us if you have any questions and we will be happy to help:
* jin.kweon at mail.mcgill.ca 
* jun.ding at mcgill.ca
