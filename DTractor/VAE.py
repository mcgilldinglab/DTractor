import numpy as np
import scvi
import torch
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
