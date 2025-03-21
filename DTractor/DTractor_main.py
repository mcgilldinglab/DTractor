import numpy as np
import math
from .DTractor_spatial import *
from .data_import import *
from .VAE import *
from .plot_helper import *
from .proportion import *

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

        return self.adata_ref, self.adata_vis

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

        return self.adata_ref_copy, self.adata_vis_copy   

    def print_instructions(self):
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
        
    def run(self, seed=42, k=5, regularization_option=1, iteration_option=3, user_defined_iterations=10000, similarity_weight=0.1, celltype_distance_weight=0.1, dtractor_only_ref=None, dtractor_only_vis=None):
        # Clear GPU memory
        torch.cuda.empty_cache()

        self.current_ref_data = dtractor_only_ref if dtractor_only_ref is not None else self.adata_ref_copy
        self.current_vis_data = dtractor_only_vis if dtractor_only_vis is not None else self.adata_vis_copy
        # Calculate cell type embeddings
        celltype_emb_mean, distance_sc, celltype_gene_matrix_torch = calculate_celltype_embeddings(self.current_ref_data)

        # Find spatial neighbors
        
        neighbors = find_spatial_neighbors(self.current_vis_data, k=k)
         # Print message if regularization_option is not 1
        if regularization_option != 1:
            print(f"\nspatial regularization 2 neighbors assumption = {k}\n")
            
        # Estimate iterations
        est_iter, start_range, end_range = estimate_iterations(self.current_vis_data, self.current_ref_data)  

        # Set up tensors for deconvolution
        st, st_emb, spot_celltype, celltype_gene_matrix_torch = setup_deconvolution(self.current_vis_data, self.current_ref_data)

        # Run the deconvolution function
        self.spot_celltype, self.st_approx_adam_torch = run_deconvolution(st, st_emb, spot_celltype, celltype_gene_matrix_torch,
                                                            distance_sc=distance_sc,
                                                            neighbors=neighbors,
                                                            est_iter=est_iter,
                                                            start_range=start_range,
                                                            end_range=end_range,
                                                            adata_vis=self.current_vis_data,
                                                            regularization_option=regularization_option,
                                                            iteration_option=iteration_option,
                                                            user_defined_iterations=user_defined_iterations,
                                                            similarity_weight=similarity_weight,
                                                            celltype_distance_weight=celltype_distance_weight, 
                                                            seed=seed)
        
    def plotting(self):
        # Run the visualization functions
        if self.spot_celltype is None or self.st_approx_adam_torch is None:
            print("Error: Run the deconvolution first before plotting.")
            return

        ref_data = getattr(self, 'current_ref_data', self.adata_ref_copy)
        vis_data = getattr(self, 'current_vis_data', self.adata_vis_copy)
            
        # Run the visualization functions
        self.spot_celltype = plot_spatial_celltype_predictions(self.spot_celltype, vis_data, self.st_approx_adam_torch, ref_data)
        plot_pc1_spatial(self.spot_celltype, self.st_approx_adam_torch)
        plot_celltype_correlation(self.st_approx_adam_torch, ref_data)

    def prop_matrix(self):
        ref_data = getattr(self, 'current_ref_data', self.adata_ref_copy)
        
        self.spot_celltype = proportion(self.spot_celltype, ref_data)
        return self.spot_celltype
    
