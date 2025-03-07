import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np

def plot_spatial_celltype_predictions(spot_celltype, adata_vis_copy, st_approx_adam_torch, adata_ref_copy):
    """
    Plot spatial distribution of predicted cell types.
    
    Parameters
    ----------
    spot_celltype : AnnData
        AnnData object containing spot-celltype information
    adata_vis_copy : AnnData
        AnnData object containing spatial data
    st_approx_adam_torch : np.ndarray
        Spot x cell type proportion matrix
    adata_ref_copy : AnnData
        Reference AnnData object containing cell type categories
        
    Returns
    -------
    None
    """
    # Add spatial coordinates
    spot_celltype.obsm['lat'] = adata_vis_copy.obsm['spatial'][:,0]
    spot_celltype.obsm['lon'] = adata_vis_copy.obsm['spatial'][:,1]
    
    if 'annotation' in adata_vis_copy.obs:
        spot_celltype.obs['annotation'] = list(adata_vis_copy.obs['annotation'])

    # Get most likely cell type for each spot
    ind = [] 
    for row in st_approx_adam_torch:
        max_col_pos = np.argmax(row)
        ind.append(max_col_pos)
    most_likely_celltype = []
    for elem in ind:
        pred = adata_ref_copy.obs['celltype'].cat.categories[elem]
        most_likely_celltype.append(pred)
    spot_celltype.obs["celltype_pred"] = most_likely_celltype

    # Create plot
    plt.figure(figsize=(6, 6))
    sns.set_style("white")
    sns.despine(trim=False)
    plt.tick_params(axis='both', which='major', labelsize=17, bottom=True, left=True)
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        
    sns.scatterplot(x=spot_celltype.obsm['lat'], 
                    y=spot_celltype.obsm['lon'], 
                    hue=spot_celltype.obs["celltype_pred"], 
                    palette='Paired', 
                    s=75)
    plt.title('DTractor', fontsize=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1), fontsize=13, markerscale=1.4)
    plt.grid(False)

def plot_pc1_spatial(spot_celltype, st_approx_adam_torch):
    """
    Plot first principal component of cell type composition in spatial coordinates.
    
    Parameters
    ----------
    spot_celltype : AnnData
        AnnData object containing spatial coordinates
    st_approx_adam_torch : np.ndarray
        Spot x cell type proportion matrix
        
    Returns
    -------
    None
    """
    pca = PCA(n_components=1)
    pca.fit(st_approx_adam_torch.T)
    PC1 = pca.components_[0]
    plt.figure(figsize=(8, 6))
    s = plt.scatter(spot_celltype.obsm['lat'], 
                spot_celltype.obsm['lon'], 
                c=PC1, 
                cmap='Spectral', 
                s=30)
    plt.xlabel('spatial1')
    plt.ylabel('spatial2')
    plt.title('PC1_celltype_composition_plot')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.3), fontsize=13)
    plt.colorbar(s)

def plot_celltype_correlation(st_approx_adam_torch, adata_ref_copy):
    """
    Plot correlation heatmap between cell types.
    
    Parameters
    ----------
    st_approx_adam_torch : np.ndarray
        Spot x cell type proportion matrix
    adata_ref_copy : AnnData
        Reference AnnData object containing cell type categories
        
    Returns
    -------
    None
    """
    f, ax = plt.subplots(figsize=(10, 10))
    corr = pd.DataFrame(st_approx_adam_torch).set_axis(adata_ref_copy.obs['celltype'].cat.categories, axis=1).corr()
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, 
                vmax=1.0,
                square=True, 
                ax=ax)
