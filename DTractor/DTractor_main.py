import pandas as pd 
import scanpy as sc
import numpy as np
import torch
import random
import scvi
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch.optim as optim
import torch.nn.functional as F
import plotly.express as px
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
seed = 42
from .DTractor_spatial import *
from .data_import import *


# Setting a fixed seed ensures reproducibility of results
# This is important for scientific work and debugging
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed 
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# def validate_spatial_data(adata_vis):
#     """
#     Validate spatial transcriptomics data in AnnData format.
    
#     Parameters
#     ----------
#     adata_vis : AnnData
#         AnnData object containing spatial transcriptomics data
        
#     Returns
#     -------
#     AnnData
#         The validated AnnData object
        
#     Raises
#     ------
#     TypeError
#         If adata_vis is not an AnnData object or X is not array/sparse
#     ValueError
#         If X dimensions or spatial coordinates are invalid
#     KeyError
#         If spatial coordinates are missing
#     """
#     # Check if it's an AnnData object
#     if not isinstance(adata_vis, sc.AnnData):
#         raise TypeError("adata_vis is not an AnnData object")

#     # Check if X is an array with numeric dtype
#     if not isinstance(adata_vis.X, np.ndarray) and not scipy.sparse.issparse(adata_vis.X):
#         raise TypeError("adata_vis.X is not an array or sparse matrix")
        
#     # Check dimensions of X
#     p, q = adata_vis.X.shape
#     if p <= 0 or q <= 0:
#         raise ValueError(f"adata_vis.X has invalid dimensions: {adata_vis.X.shape}")

#     # Check if spatial is in obsm
#     if 'spatial' not in adata_vis.obsm:
#         raise KeyError("adata_vis.obsm does not contain 'spatial'. Please ensure your AnnData object has spatial coordinates in obsm['spatial'].")
        
#     # Check if spatial is an n x 2 array
#     if adata_vis.obsm['spatial'].shape[1] != 2:
#         raise ValueError(f"adata_vis.obsm['spatial'] should be n x 2, but is {adata_vis.obsm['spatial'].shape}")

#     # Warning about AnnData dimensions
#     print("⚠️ WARNING: In AnnData, rows (obs) should be spots and columns (var) should be genes")
#     print(f"adata spatial data is valid: {adata_vis.shape[0]} spots x {adata_vis.shape[1]} genes")
    
#     return adata_vis



# def validate_reference_data(adata_ref):
#     """
#     Validate reference AnnData object for single-cell analysis.
    
#     Parameters
#     ----------
#     adata_ref : AnnData
#         The reference AnnData object to validate
        
#     Returns
#     -------
#     AnnData
#         The validated AnnData object
        
#     Raises
#     ------
#     TypeError
#         If adata_ref is not an AnnData object or X is not array/sparse
#     ValueError
#         If X dimensions are invalid
#     KeyError
#         If celltype annotations are missing
#     """
#     # Check if it's an AnnData object
#     if not isinstance(adata_ref, sc.AnnData):
#         raise TypeError("adata_ref is not an AnnData object")

#     # Check if X is an array with numeric dtype
#     if not isinstance(adata_ref.X, np.ndarray) and not scipy.sparse.issparse(adata_ref.X):
#         raise TypeError("adata_ref.X is not an array or sparse matrix")
        
#     # Check dimensions of X
#     p, q = adata_ref.X.shape
#     if p <= 0 or q <= 0:
#         raise ValueError(f"adata_ref.X has invalid dimensions: {adata_ref.X.shape}")

#     # Check if celltype is in obs
#     if 'celltype' not in adata_ref.obs:
#         raise KeyError("adata_ref.obs does not contain 'celltype'. Please ensure your AnnData object has celltype annotations.")

#     # Warning about AnnData dimensions
#     print("⚠️ WARNING: In AnnData, rows (obs) should be cells and columns (var) should be genes")
#     print(f"adata reference data is valid: {adata_ref.shape[0]} cells x {adata_ref.shape[1]} genes")
    
#     return adata_ref


def prepare_data(adata_ref, adata_vis):
    """
    Prepare reference and visium data by finding common genes and handling negative values.
    
    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset
    adata_vis : AnnData
        Visium spatial dataset
        
    Returns
    -------
    tuple
        Processed copies of reference and visium data
    """
    adata_ref_copy = adata_ref.copy()
    adata_vis_copy = adata_vis.copy()

    # Find common genes
    intersects = np.intersect1d(adata_ref.var.index, adata_vis.var.index)
    adata_vis_copy = adata_vis_copy[:, intersects].copy()
    adata_ref_copy = adata_ref_copy[:, intersects].copy()

    # Handle negative values in reference data
    def handle_negative_values(adata, data_name):
        min_value = np.min(adata.X)
        max_value = np.max(adata.X)
        negative_count = np.sum(adata.X < 0)
        negative_percentage = negative_count/(adata.shape[0] * adata.shape[1]) * 100

        print(f"\nChecking {data_name}:")
        print(f"Min value in data: {min_value}")
        print(f"Max value in data: {max_value}")
        print(f"Number of negative values: {negative_count}")
        print(f"Percentage of negative values: {negative_percentage:.4f}%")

        if negative_count > 0:
            print("\n⚠️ WARNING: Negative values detected in the expression matrix.")
            print("These are likely due to normalization or preprocessing steps.")
            print("Replacing all negative values with zeros to ensure compatibility with downstream analysis.")
            
            adata.X[adata.X < 0] = 0
            print(f"After replacement - Min value: {np.min(adata.X)}")
            assert np.min(adata.X) >= 0, "Failed to replace all negative values"

        adata.layers["data"] = adata.X
        return adata

    adata_ref_copy = handle_negative_values(adata_ref_copy, "reference data")
    adata_vis_copy = handle_negative_values(adata_vis_copy, "spatial data")

    return adata_ref_copy, adata_vis_copy

def train_scvi_model(adata, max_epochs=50, batch_size=128, device=None, seed=42, model_type="reference"):
    """
    Train a scVI model on the provided AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with setup already performed
    max_epochs : int, default=50
        Maximum number of training epochs
    batch_size : int, default=128
        Batch size for training
    device : int or bool, default=None
        GPU device number to use, False for CPU, None for auto-detection
    seed : int, default=42
        Random seed for reproducibility
    model_type : str, default="reference"
        Type of model to train ("reference" or "spatial")
        
    Returns
    -------
    scvi.model.SCVI
        Trained scVI model
    """
    set_seed(seed)
    
    gpu_available = torch.cuda.is_available()
    use_gpu = False
    if gpu_available:
        if device is None:
            use_gpu = True
            device_str = "GPU"
        elif device is False:
            use_gpu = False
            device_str = "CPU (forced)"
        else:
            use_gpu = True
            device_str = f"GPU (device {device})"
    else:
        device_str = "CPU (no GPU available)"

    if model_type == "reference":
        model = scvi.model.SCVI(adata,
                            n_hidden=256,
                            n_layers=3,
                            n_latent=16,
                            gene_likelihood="zinb",
                            dropout_rate=0.1,
                            latent_distribution="normal",
                            dispersion='gene-cell')
    else:
        model = scvi.model.SCVI(adata,
                            n_hidden=128,
                            n_layers=3,
                            n_latent=16,
                            gene_likelihood="zinb",
                            dropout_rate=0,
                            latent_distribution="normal")

    print(f"\nTraining {model_type} model on {device_str}")
    model.train(max_epochs=max_epochs,
                train_size=1,
                batch_size=batch_size,
                early_stopping=False,
                check_val_every_n_epoch=10,
                use_gpu=use_gpu)

    train_elbo = model.history['elbo_train'][1:]
    plt.figure(figsize=(10, 4))
    plt.plot(train_elbo)
    plt.xlabel('Epochs')
    plt.ylabel('ELBO loss')
    plt.title(f'Training loss curve - {model_type} model')
    plt.tight_layout()
    plt.show()

    return model

def get_latent_representation(model, adata):
    """
    Get latent representation from trained model and store in AnnData.
    
    Parameters
    ----------
    model : scvi.model.SCVI
        Trained scVI model
    adata : AnnData
        Data object to store latent representation
        
    Returns
    -------
    AnnData
        Data object with added latent representation
    """
    latent = model.get_latent_representation(give_mean=True, return_dist=True)
    adata.obsm['mus'] = latent[0]
    adata.obsm['var'] = latent[1]
    return adata

# Main workflow
def run_scvi_analysis(adata_ref, adata_vis, seed=42):
    """
    Run complete scVI analysis workflow.
    
    Parameters
    ----------
    adata_ref : AnnData
        Reference single-cell dataset
    adata_vis : AnnData
        Visium spatial dataset
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        Processed reference and visium data with latent representations
    """
    # Prepare data
    adata_ref_copy, adata_vis_copy = prepare_data(adata_ref, adata_vis)
    
    # Setup and train reference model
    set_seed(seed)
    scvi.model.SCVI.setup_anndata(adata_ref_copy, layer="data", labels_key="celltype")
    model_ref = train_scvi_model(adata_ref_copy, model_type="reference", seed=seed)
    adata_ref_copy = get_latent_representation(model_ref, adata_ref_copy)
    
    set_seed(seed)
    # Setup and train visium model
    scvi.model.SCVI.setup_anndata(adata_vis_copy, layer="data")
    model_vis = train_scvi_model(adata_vis_copy, max_epochs=70, batch_size=64, model_type="spatial", seed=seed)
    adata_vis_copy = get_latent_representation(model_vis, adata_vis_copy)
    
    # Add spatial information if available
    if (adata_vis.obs_names==adata_vis_copy.obs_names).sum() == adata_vis.shape[0]:
        adata_vis_copy.obsm['spatial'] = adata_vis.obsm['spatial']
    
    return adata_ref_copy, adata_vis_copy




class DTractor_pipeline:
    def __init__(self,spatial_data_path,reference_data_path):
        self.spatial_data_path = spatial_data_path
        self.reference_data_path = reference_data_path

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

    def data_import(self):
        # Read and validate the dataset
        # adata_vis = validate_spatial_data(sc.read_h5ad("spatial_example.h5ad"))
        adata_vis = validate_spatial_data(sc.read_h5ad(self.spatial_data_path))
        # Read and validate the dataset
        # adata_ref = validate_reference_data(sc.read_h5ad("reference_example.h5ad"))
        adata_ref = validate_reference_data(sc.read_h5ad(self.reference_data_path))


        # Find common genes between reference and visium datasets
        common_genes = np.intersect1d(adata_ref.var.index, adata_vis.var.index)
        print(f"Found {common_genes.shape[0]} common genes between reference and visium datasets")
        print("These common genes will be used for downstream analysis")
        
    def run(self):
        adata_ref_copy, adata_vis_copy = run_scvi_analysis(adata_ref, adata_vis)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Calculate cell type embeddings
        celltype_emb_mean, distance_sc, celltype_gene_matrix_torch = calculate_celltype_embeddings(adata_ref_copy)

        # Find spatial neighbors
        neighbors = find_spatial_neighbors(adata_vis_copy, k=5)

        # Estimate iterations
        est_iter, start_range, end_range = estimate_iterations(adata_vis_copy, adata_ref_copy)  

        # Set up tensors for deconvolution
        st, st_emb, spot_celltype, celltype_gene_matrix_torch = setup_deconvolution(adata_vis_copy, adata_ref_copy)

        # Print instructions for adam_st_torch parameters
        print("adam_st_torch function parameters:")
        print("  regularization_option: 1 = Fastest (just Frobenius norm, no regularization)")
        print("                         2 = Simple regularization with argsort (faster)")
        print("                         3 = Sophisticated regularization with softmax (slower but potentially better)")
        print("  iteration_option:      1 = Use estimated iterations (est_iter)")
        print("                         2 = Use range-based iterations (end_range)")
        print("                         3 = Use user-defined iterations")
        print("  user_defined_iterations: Number of iterations if iteration_option is 3")
        print("  similarity_weight:     Weight for similarity loss (should be 0 for regularization_option=1)")
        print("  celltype_distance_weight: Weight for celltype distance loss (should be 0 for regularization_option=1)")

        # Run the deconvolution function
        spot_celltype, st_approx_adam_torch = run_deconvolution(st, st_emb, spot_celltype, celltype_gene_matrix_torch,
                                                            regularization_option=1,
                                                            iteration_option=3,
                                                            user_defined_iterations=320000,
                                                            similarity_weight=0.1,
                                                            celltype_distance_weight=0.1)

        # Run the visualization functions
        plot_spatial_celltype_predictions(spot_celltype, adata_vis_copy, st_approx_adam_torch, adata_ref_copy)
        plot_pc1_spatial(spot_celltype, st_approx_adam_torch) 
        plot_celltype_correlation(st_approx_adam_torch, adata_ref_copy)

    
