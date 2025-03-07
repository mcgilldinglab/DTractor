# import pandas as pd 
# import scanpy as sc
import numpy as np
# import torch
# import random
# import scvi
# import statsmodels.api as sm
# import torch.optim as optim
# import torch.nn.functional as F
import math
from .DTractor_spatial import *
from .data_import import *
from .VAE import *
from .plot_helper import *

class DTractor_pipeline:
    def __init__(self):
        # Note to users about preprocessing requirements
        print("\n" + "="*80)
        print("IMPORTANT PREPROCESSING REQUIREMENTS:")
        print("="*80)
        print("Before using this notebook, please ensure both datasets are properly preprocessed:")
        print("")
        print("For spatial transcriptomics data:")
        print("  - Filter low-quality spots (low gene count, high mitochondrial content)")
        print("  - Normalize counts (e.g., using sc.pp.normalize_total)")
        print("  - Log-transform data (e.g., using sc.pp.log1p)")
        print("  - Store spatial coordinates in .obsm['spatial']")
        print("  - Identify highly variable genes (e.g., using sc.pp.highly_variable_genes)")
        print("  - Filter out mitochondria-encoded genes if needed")
        print("  - batch corrections if needed")
        print("")
        print("For single-cell/nucleus reference data:")
        print("  - Filter low-quality cells (low gene count, high mitochondrial content)")
        print("  - Normalize counts (e.g., using sc.pp.normalize_total)")
        print("  - Log-transform data (e.g., using sc.pp.log1p)")
        print("  - Annotate cell types in .obs['celltype']")
        print("  - Identify highly variable genes (e.g., using sc.pp.highly_variable_genes)")
        print("  - Consider using cell2location's filter_genes() to select informative genes")
        print("    that separate cell types in your reference data")
        print("  - batch corrections if needed")
        print("="*80)
        print("")
        print("IMPORTANT DATA STRUCTURE REQUIREMENTS:")
        print("="*80)
        print("The AnnData objects must have specific structures:")
        print("  - Spatial transcriptomics data: Must have spatial coordinates in .obsm['spatial'] and .obs['annotation'] if ST has manual annotation reference")
        print("  - Single-cell/nucleus reference data: Must have cell type annotations in .obs['celltype']")
        print("="*80)

        # Initialize adata attributes to None
        self.adata_vis = None
        self.adata_ref = None
        
        self.adata_vis_copy = None  # Store processed data
        self.adata_ref_copy = None  # Store processed data
        
        self.spot_celltype = None
        self.st_approx_adam_torch = None
        
    def data_import(self, spatial_data_path, reference_data_path):
        self.spatial_data_path = spatial_data_path
        self.reference_data_path = reference_data_path
        
        # Read and validate the dataset
        # adata_vis = validate_spatial_data(sc.read_h5ad("spatial_example.h5ad"))
        self.adata_vis = validate_spatial_data(sc.read_h5ad(self.spatial_data_path))
        
        # Read and validate the dataset
        # adata_ref = validate_reference_data(sc.read_h5ad("reference_example.h5ad"))
        self.adata_ref = validate_reference_data(sc.read_h5ad(self.reference_data_path))


        # Find common genes between reference and visium datasets
        common_genes = np.intersect1d(self.adata_ref.var.index, self.adata_vis.var.index)
        print(f"\n\nFound {common_genes.shape[0]} common genes between reference and visium datasets")
        print("These common genes will be used for downstream analysis")

    def vae_train(self, seed=42):
        """
        Train VAE model with an optional seed for reproducibility.
        
        Parameters:
        - seed: An optional integer seed for reproducibility (default is 42).
        """
        if seed is not None:
            print(f"Training VAE with seed: {seed}")
        else:
            print("Training VAE without a fixed seed, 42.")
            
        self.adata_ref_copy, self.adata_vis_copy = run_scvi_analysis(self.adata_ref, self.adata_vis, seed)
        
    def run(self):
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Calculate cell type embeddings
        celltype_emb_mean, distance_sc, celltype_gene_matrix_torch = calculate_celltype_embeddings(self.adata_ref_copy)

        # Find spatial neighbors
        neighbors = find_spatial_neighbors(self.adata_vis_copy, k=5)

        # Estimate iterations
        est_iter, start_range, end_range = estimate_iterations(self.adata_vis_copy, self.adata_ref_copy)  

        # Set up tensors for deconvolution
        st, st_emb, spot_celltype, celltype_gene_matrix_torch = setup_deconvolution(self.adata_vis_copy, self.adata_ref_copy)

        # Print instructions for adam_st_torch parameters
        print("\n\nadam_st_torch function parameters:")
        print("  regularization_option: 1 = Fastest (just Frobenius norm, no regularization)")
        print("                         2 = Simple regularization with argsort (faster)")
        print("                         3 = Sophisticated regularization with softmax (slower but potentially better)")
        print("  iteration_option:      1 = Use estimated iterations (est_iter)")
        print("                         2 = Use range-based iterations (end_range)")
        print("                         3 = Use user-defined iterations")
        print("  user_defined_iterations: Number of iterations if iteration_option is 3")
        print("  similarity_weight:     Weight for similarity loss (should be 0 for regularization_option=1)")
        print("  celltype_distance_weight: Weight for celltype distance loss (should be 0 for regularization_option=1)\n\n")

        # Run the deconvolution function
        self.spot_celltype, self.st_approx_adam_torch = run_deconvolution(st, st_emb, spot_celltype, celltype_gene_matrix_torch,
                                                            regularization_option=1,
                                                            iteration_option=3,
                                                            user_defined_iterations=10000,
                                                            similarity_weight=0.1,
                                                            celltype_distance_weight=0.1)
        
    def plotting(self):
        # Run the visualization functions
        if self.spot_celltype is None or self.st_approx_adam_torch is None:
            print("Error: Run the deconvolution first before plotting.")
            return
            
        # Run the visualization functions
        plot_spatial_celltype_predictions(self.spot_celltype, self.adata_vis_copy, self.st_approx_adam_torch, self.adata_ref_copy)
        plot_pc1_spatial(self.spot_celltype, self.st_approx_adam_torch)
        plot_celltype_correlation(self.st_approx_adam_torch, self.adata_ref_copy)


    
