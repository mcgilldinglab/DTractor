import numpy as np
import torch
import random
import scvi
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors   
from sklearn import preprocessing
import torch.optim as optim
import torch.nn.functional as F
import statsmodels.api as sm
import scanpy as sc
from .DTractor_main import *

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
        

def plot_accuracy(train_loss):
    """
    Plot the training loss curve.
    
    Parameters:
    -----------
    train_loss : list
        List of training loss values.
    """
    num_epochs = len(train_loss)
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(1, num_epochs+1), train_loss, label='Training loss')
    plt.xlabel('Loop')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    


def calculate_celltype_embeddings(adata_ref_copy):
    """
    Calculate cell type embeddings based on mean values.
    
    Parameters:
    -----------
    adata_ref_copy : AnnData
        Reference dataset with cell type annotations.
        
    Returns:
    --------
    tuple: (celltype_emb_mean, distance_sc, celltype_gene_matrix_torch)
    """
    # Calculate cell type embeddings based on latent representation means
    celltype_emb_mean = []
    for celltype_i in adata_ref_copy.obs['celltype'].cat.categories:
        celltype_emb_mean.append(list(np.mean(adata_ref_copy.obsm['mus'][adata_ref_copy.obs['celltype'] == celltype_i], axis=0)))
    celltype_emb_mean = np.array(celltype_emb_mean)
    
    # Calculate distance matrix between cell types
    distance_sc = squareform(pdist(celltype_emb_mean, 'euclidean'))
    distance_sc = torch.tensor(distance_sc, dtype=torch.float)
    
    # Alternative cell type embedding based on gene expression (faster)
    x = []
    for celltype_i in adata_ref_copy.obs['celltype'].cat.categories:
        x.append(np.mean(adata_ref_copy.X[adata_ref_copy.obs['celltype'] == celltype_i], axis=0))
    celltype_gene_matrix_torch = np.vstack(x)
    celltype_gene_matrix_torch = torch.Tensor(celltype_gene_matrix_torch)
    if torch.cuda.is_available():
        celltype_gene_matrix_torch = celltype_gene_matrix_torch.cuda()
        
    return celltype_emb_mean, distance_sc, celltype_gene_matrix_torch



def find_spatial_neighbors(adata_vis_copy, k=5):
    """
    Find spatial neighbors for each spot.
    
    Parameters:
    -----------
    adata_vis_copy : AnnData
        Spatial dataset.
    k : int
        Number of neighbors to find.
        
    Returns:
    --------
    numpy.ndarray: Indices of neighbors for each spot.
    """
    N = adata_vis_copy.obsm['spatial'].shape[0]
    X = adata_vis_copy.obsm['spatial']
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices.copy()



def estimate_iterations(adata_vis_copy, adata_ref_copy):
    """
    Estimate the number of iterations needed for optimization.
    
    Parameters:
    -----------
    adata_vis_copy : AnnData
        Spatial dataset.
    adata_ref_copy : AnnData
        Reference dataset.
        
    Returns:
    --------
    int: Estimated number of iterations.
    """
    est_iter = (50 * adata_vis_copy.shape[0] + 10 * adata_vis_copy.shape[1] - 
                5 * adata_ref_copy.shape[0] + 10000 * len(adata_ref_copy.obs['celltype'].cat.categories))
    
    if est_iter > 50000:
        print("it's good to go")  # make sure the number of iterations is at least 50000
    else:
        est_iter = 100000
        
    if est_iter > 100000:
        start_range = est_iter - 100000
        end_range = est_iter + 100000
    else:
        start_range = 0
        end_range = 250000
        
    return est_iter, start_range, end_range





def compute_r2(st_approx_adam_torch):
    """
    Compute R2 score to validate deconvolution results.
    
    Parameters:
    -----------
    st_approx_adam_torch : torch.Tensor
        Deconvolved spot-cell type matrix.
        
    Returns:
    --------
    float: McFadden's R-squared value.
    """
    # Check if 'annotation' is available in the spatial dataset
    if 'annotation' not in adata_vis.obs.columns:
        print("Warning: 'annotation' column not found in spatial dataset. Cannot compute R2 score.")
        return None
    
    print("Computing R2 score using existing 'annotation' column in spatial dataset...")
    print("This helps validate that the previously provided annotation is valid.")
    
    pca = PCA(n_components=1)
    pca.fit(st_approx_adam_torch.T)

    # Get the first principal component (PC1)
    PC1 = pca.components_[0]

    le = preprocessing.LabelEncoder()
    label_entire = le.fit_transform(list(adata_vis.obs['annotation']))
    X = PC1
    model = sm.MNLogit(label_entire, sm.add_constant(X)).fit()
    mine_entire_redeannot_mcfadden_r2 = model.prsquared
    print("McFadden's R-squared redeconve annotation:", round(mine_entire_redeannot_mcfadden_r2, 4))
    return mine_entire_redeannot_mcfadden_r2

def softmax_rank(x):
    """
    Calculate soft ranks using softmax transformation.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor.
        
    Returns:
    --------
    torch.Tensor: Soft ranks.
    """
    # Normalize input by maximum value
    x_shifted = x / x.max(dim=-1, keepdim=True).values
    # Apply exponential transformation with large base for sharper distinctions
    exp_x = torch.pow(1024, x_shifted)
    # Apply softmax to get probability distribution
    soft_ranks = exp_x / exp_x.sum(dim=-1, keepdim=True)
    # Scale by input dimension
    return x.shape[1] * soft_ranks

def set_optimization_options(regularization_option=1, iteration_option=3, user_defined_iterations=None, 
                            similarity_weight=0.0, celltype_distance_weight=0.0):
    """
    Set optimization options for the deconvolution process.
    
    Parameters:
    -----------
    regularization_option : int (1, 2, or 3)
        Option 1: Fastest (just Frobenius norm, no regularization)
        Option 2: Simple regularization with argsort (faster)
        Option 3: Sophisticated regularization with softmax (slower but potentially better)
    
    iteration_option : int (1, 2, or 3)
        Option 1: Use estimated iterations (est_iter)
        Option 2: Use range-based iterations (end_range)
        Option 3: Use user-defined iterations
    
    user_defined_iterations : int
        Number of iterations if iteration_option is 3
        
    similarity_weight : float
        Weight for similarity loss (should be 0 for regularization_option=1)
        
    celltype_distance_weight : float
        Weight for celltype distance loss (should be 0 for regularization_option=1)
    
    Returns:
    --------
    dict: Dictionary containing all optimization options
    """
    # If regularization_option is 1, force weights to be 0
    if regularization_option == 1:
        similarity_weight = 0.0
        celltype_distance_weight = 0.0
    
    return {
        'regularization_option': regularization_option,
        'iteration_option': iteration_option,
        'user_defined_iterations': user_defined_iterations if user_defined_iterations else 250000,
        'similarity_weight': similarity_weight,
        'celltype_distance_weight': celltype_distance_weight
    }

def custom_loss(A, B, C, D, iteration, similarity_weight, celltype_distance_weight, regularization_option, ranks_sc=None):
    """
    Custom loss function for deconvolution optimization.
    
    Parameters:
    -----------
    A : torch.Tensor
        Spatial transcriptomics data.
    B : torch.Tensor
        Spatial transcriptomics embeddings.
    C : torch.Tensor
        Spot-celltype matrix (to be optimized).
    D : torch.Tensor
        Celltype-gene matrix.
    iteration : int
        Current iteration.
    similarity_weight : float
        Weight for similarity loss.
    celltype_distance_weight : float
        Weight for celltype distance loss.
    regularization_option : int
        Regularization option (1, 2, or 3).
    ranks_sc : torch.Tensor
        Cell type distance ranks.
        
    Returns:
    --------
    tuple: (total_loss, BC, celltype_distance_loss, similarity_loss, frobenius_loss)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate ranks_st and celltype_distance_loss based on regularization_option
    if regularization_option == 1:
        # Option 1: Skip ranks_st calculation entirely for fastest computation
        ranks_st = None
        celltype_distance_loss = torch.tensor(0.0, device=device)
    elif regularization_option == 2:
        # Option 2: Simple regularization with argsort
        distance_st = torch.norm(C.T.float()[:,None] - C.T.float(), dim=2, p=2)
        if torch.cuda.is_available():
            distance_st_tensor = torch.tensor(distance_st).cuda()
        else:
            distance_st_tensor = torch.tensor(distance_st)
            
        ranks_st = torch.tensor(torch.argsort(torch.argsort(distance_st_tensor, dim=1), dim=1), dtype=torch.float, requires_grad=True)
        celltype_distance_loss = F.mse_loss(ranks_sc.float(), ranks_st.float())
    elif regularization_option == 3:
        # Option 3: Sophisticated regularization with softmax_rank
        distance_st = torch.norm(C.T.float()[:,None] - C.T.float(), dim=2, p=2)
        if torch.cuda.is_available():
            ranks_st = softmax_rank(distance_st.cuda())
        else:
            ranks_st = softmax_rank(torch.tensor(distance_st))
        celltype_distance_loss = F.mse_loss(ranks_sc.float(), ranks_st.float())
    
    BC = torch.mm(B, C)
    
    base = 1.001 
    BC = torch.pow(base, BC) / torch.max(torch.pow(base, BC))
    
    predicted_A = torch.mm(BC, D)
    
    # Calculate similarity_loss based on regularization_option
    if regularization_option == 1:
        # Option 1: Skip similarity_loss calculation entirely
        similarity_loss = torch.tensor(0.0, device=device)
    elif regularization_option == 2:
        # Option 2: Simple regularization with argsort
        distance_st2 = torch.norm(BC.float()[:,None] - BC.float(), dim=2, p=2)
        if torch.cuda.is_available():
            distance_st2_tensor = torch.tensor(distance_st2).cuda()
        else:
            distance_st2_tensor = torch.tensor(distance_st2)
            
        rankings = torch.tensor(torch.argsort(torch.argsort(distance_st2_tensor, dim=1), dim=1), dtype=torch.float, requires_grad=True)
        similarity_loss = torch.sum(rankings[np.arange(len(neighbors))[:, None], neighbors])
    elif regularization_option == 3:
        # Option 3: Sophisticated regularization with softmax_rank
        distance_st2 = torch.norm(BC.float()[:,None] - BC.float(), dim=2, p=2)
        if torch.cuda.is_available():
            rankings = softmax_rank(distance_st2.cuda())
        else:
            rankings = softmax_rank(distance_st2)
        similarity_loss = torch.sum(rankings[np.arange(len(neighbors))[:, None], neighbors])
    
    fro_loss = torch.linalg.norm(A - predicted_A, 'fro')
    
    loss = (similarity_weight * similarity_loss) + (celltype_distance_weight * celltype_distance_loss) + fro_loss
    return loss, BC, celltype_distance_weight * celltype_distance_loss, similarity_weight * similarity_loss, fro_loss

train_loss_st_torch = []
def adam_st_torch(st, st_emb, spot_celltype, celltype_gene, 
                regularization_option=1, 
                iteration_option=3, 
                user_defined_iterations=None, 
                similarity_weight=0.0, 
                celltype_distance_weight=0.0, 
                seed=42):
    """
    Perform deconvolution using Adam optimizer with customizable optimization options.
    
    Parameters:
    -----------
    st : torch.Tensor
        Spatial transcriptomics data
    st_emb : torch.Tensor
        Spatial transcriptomics embeddings
    spot_celltype : torch.Tensor
        Spot-celltype matrix (to be optimized)
    celltype_gene : torch.Tensor
        Celltype-gene matrix
    regularization_option : int (1, 2, or 3)
        Option 1: Fastest (just Frobenius norm, no regularization)
        Option 2: Simple regularization with argsort (faster)
        Option 3: Sophisticated regularization with softmax (slower but potentially better)
    iteration_option : int (1, 2, or 3)
        Option 1: Use estimated iterations (est_iter)
        Option 2: Use range-based iterations (end_range)
        Option 3: Use user-defined iterations
    user_defined_iterations : int
        Number of iterations if iteration_option is 3
    similarity_weight : float
        Weight for similarity loss
    celltype_distance_weight : float
        Weight for celltype distance loss
    
    Returns:
    --------
    tuple: (st_approx_adam_torch, best_iteration)
    """
    set_seed(seed)
                    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create optimization options from parameters
    opt_options = set_optimization_options(
        regularization_option=regularization_option,
        iteration_option=iteration_option,
        user_defined_iterations=user_defined_iterations,
        similarity_weight=similarity_weight,
        celltype_distance_weight=celltype_distance_weight
    )
    
    # Extract options
    regularization_option = opt_options['regularization_option']
    iteration_option = opt_options['iteration_option']
    user_defined_iterations = opt_options['user_defined_iterations']
    similarity_weight = opt_options['similarity_weight']
    celltype_distance_weight = opt_options['celltype_distance_weight']

    # Process based on selected option
    if regularization_option == 1:
        # Option 1: Skip ranks_sc calculation entirely for fastest computation
        ranks_sc = None  # Will not be used in loss function
        print("Using Option 1: Fastest computation (Frobenius norm only)")
    elif regularization_option == 2:
        # Option 2: Simple regularization with argsort
        print("Using Option 2: Simple regularization")
        if torch.cuda.is_available():
            distance_sc_tensor = torch.tensor(distance_sc).cuda()
        else:
            distance_sc_tensor = torch.tensor(distance_sc)
        
        ranks_sc = torch.tensor(torch.argsort(torch.argsort(distance_sc_tensor, dim=1), dim=1), 
                            dtype=torch.float, requires_grad=True)
    elif regularization_option == 3:
        # Option 3: Sophisticated regularization with softmax_rank
        print("Using Option 3: Sophisticated regularization (slower)")
        if torch.cuda.is_available():
            ranks_sc = softmax_rank(distance_sc.cuda())
        else:
            distance_sc_tensor = torch.tensor(distance_sc)
            ranks_sc = softmax_rank(distance_sc_tensor)
        
    # Choose an optimization algorithm (e.g., Adam) and set hyperparameters
    learning_rate = 0.001
    optimizer = optim.Adam([spot_celltype], lr=learning_rate)
    
    # Set the number of training iterations based on iteration_option
    if iteration_option == 1:
        num_iterations = est_iter
    elif iteration_option == 2:
        num_iterations = end_range
    elif iteration_option == 3:
        num_iterations = user_defined_iterations
    else:
        # Default to end_range if option is invalid
        num_iterations = 250000

    prev_loss = float('inf')
    prev_celltypedist_loss = float('inf')
    best_r2 = 0
    best_iteration = 0
    
    for iteration in range(num_iterations):
        # Compute the loss
        loss, BC, celltypedistloss, similarityloss, froloss = custom_loss(st, st_emb, spot_celltype, celltype_gene, iteration, 
                                                                        similarity_weight, celltype_distance_weight, regularization_option, ranks_sc)
            
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Update C using the optimizer
        optimizer.step()

        prev_celltypedist_loss = celltypedistloss.item()
        
        prev_loss = loss.item()
        train_loss_st_torch.append(loss.item())
        # Print the loss at regular intervals
        if (iteration + 1) % 3000 == 0:
            # Only print once for each interval
            if iteration_option != 2:
                print(f"Iteration [{iteration + 1}/{num_iterations}], Entire Loss: {loss.item()}, Original: {froloss.item()}, Regularization 1: {similarityloss.item()}, Regularization 2: {celltypedistloss.item()}")
            
            # For iteration_option 2, compute R2 and track best performance
            if iteration_option == 2 and start_range <= iteration <= end_range:
                print(f"Iteration [{iteration + 1}/{num_iterations}], Entire Loss: {loss.item()}, Original: {froloss.item()}, Regularization 1: {similarityloss.item()}, Regularization 2: {celltypedistloss.item()}")
                
                # Compute R2 and update best model if improved
                r2 = compute_r2(BC.detach().cpu().numpy())
                if r2 > best_r2:
                    best_r2 = r2
                    best_iteration = iteration + 1
                    st_approx_adam_torch = BC
    if iteration_option == 1 or iteration_option == 3:          
        st_approx_adam_torch = BC
    
    # Print the best R2 score and its corresponding iteration only for iteration_option 2
    if iteration_option == 2:
        print(f"Best R2 Score: {best_r2}, achieved at iteration {best_iteration}")
    return st_approx_adam_torch, best_iteration

def setup_deconvolution(adata_vis_copy, adata_ref_copy):
    """
    Set up tensors for deconvolution.
    
    Parameters:
    -----------
    adata_vis_copy : AnnData
        Spatial dataset.
    adata_ref_copy : AnnData
        Reference dataset.
        
    Returns:
    --------
    tuple: (st, st_emb, spot_celltype, celltype_gene_matrix_torch)
    """
    torch.set_printoptions(precision=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    st = torch.tensor(adata_vis_copy.X.copy(), requires_grad=False).to(device)
    st_emb = torch.tensor(adata_vis_copy.obsm['mus'], requires_grad=False).to(device)
    spot_celltype = torch.randn(st_emb.shape[1], adata_ref_copy.obs['celltype'].cat.categories.shape[0], requires_grad=True, device=device)
    
    
    _, _, celltype_gene_matrix_torch = calculate_celltype_embeddings(adata_ref_copy)
    celltype_gene_matrix_torch = torch.tensor(celltype_gene_matrix_torch, requires_grad=False).to(device)
    
    return st, st_emb, spot_celltype, celltype_gene_matrix_torch



def run_deconvolution(st, st_emb, spot_celltype, celltype_gene_matrix_torch, 
                    regularization_option=1, iteration_option=3, 
                    user_defined_iterations=250000, similarity_weight=0.1, 
                    celltype_distance_weight=0.1, seed=42):
    """
    Run deconvolution using adam_st_torch and process results.
    
    Parameters
    ----------
    st : torch.Tensor
        Spatial transcriptomics data tensor
    st_emb : torch.Tensor 
        Spatial embeddings tensor
    spot_celltype : torch.Tensor
        Initial spot-celltype tensor
    celltype_gene_matrix_torch : torch.Tensor
        Celltype-gene matrix tensor
    regularization_option : int, optional
        Regularization option (1-3), by default 1
    iteration_option : int, optional 
        Iteration option (1-3), by default 3
    user_defined_iterations : int, optional
        Number of iterations if iteration_option=3, by default 250000
    similarity_weight : float, optional
        Weight for similarity loss, by default 0.1
    celltype_distance_weight : float, optional
        Weight for celltype distance loss, by default 0.1
        
    Returns
    -------
    tuple
        (spot_celltype AnnData object, st_approx_adam_torch array)
    """
    # Run deconvolution
    st_approx_adam_torch, bestiteration = adam_st_torch(st, st_emb, spot_celltype, celltype_gene_matrix_torch,
                                                    regularization_option=regularization_option,
                                                    iteration_option=iteration_option,
                                                    user_defined_iterations=user_defined_iterations,
                                                    similarity_weight=similarity_weight,
                                                    celltype_distance_weight=celltype_distance_weight,
                                                    seed=seed)
    st_approx_adam_torch = st_approx_adam_torch.detach().cpu().numpy()

    torch.cuda.empty_cache()

    plot_accuracy(train_loss_st_torch)

    st_approx_adam_torch_original = st_approx_adam_torch.copy()
    st_approx_adam_torch = st_approx_adam_torch / st_approx_adam_torch.sum(axis=1, keepdims=True)

    spot_celltype = sc.AnnData(st_approx_adam_torch)
    
    return spot_celltype, st_approx_adam_torch
