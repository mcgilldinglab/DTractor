def validate_spatial_data(adata_vis):
    """
    Validate spatial transcriptomics data in AnnData format.
    
    Parameters
    ----------
    adata_vis : AnnData
        AnnData object containing spatial transcriptomics data
        
    Returns
    -------
    AnnData
        The validated AnnData object
        
    Raises
    ------
    TypeError
        If adata_vis is not an AnnData object or X is not array/sparse
    ValueError
        If X dimensions or spatial coordinates are invalid
    KeyError
        If spatial coordinates are missing
    """
    # Check if it's an AnnData object
    if not isinstance(adata_vis, sc.AnnData):
        raise TypeError("adata_vis is not an AnnData object")

    # Check if X is an array with numeric dtype
    if not isinstance(adata_vis.X, np.ndarray) and not scipy.sparse.issparse(adata_vis.X):
        raise TypeError("adata_vis.X is not an array or sparse matrix")
        
    # Check dimensions of X
    p, q = adata_vis.X.shape
    if p <= 0 or q <= 0:
        raise ValueError(f"adata_vis.X has invalid dimensions: {adata_vis.X.shape}")

    # Check if spatial is in obsm
    if 'spatial' not in adata_vis.obsm:
        raise KeyError("adata_vis.obsm does not contain 'spatial'. Please ensure your AnnData object has spatial coordinates in obsm['spatial'].")
        
    # Check if spatial is an n x 2 array
    if adata_vis.obsm['spatial'].shape[1] != 2:
        raise ValueError(f"adata_vis.obsm['spatial'] should be n x 2, but is {adata_vis.obsm['spatial'].shape}")

    # Warning about AnnData dimensions
    print("⚠️ WARNING: In AnnData, rows (obs) should be spots and columns (var) should be genes")
    print(f"adata spatial data is valid: {adata_vis.shape[0]} spots x {adata_vis.shape[1]} genes")
    
    return adata_vis


def validate_reference_data(adata_ref):
    """
    Validate reference AnnData object for single-cell analysis.
    
    Parameters
    ----------
    adata_ref : AnnData
        The reference AnnData object to validate
        
    Returns
    -------
    AnnData
        The validated AnnData object
        
    Raises
    ------
    TypeError
        If adata_ref is not an AnnData object or X is not array/sparse
    ValueError
        If X dimensions are invalid
    KeyError
        If celltype annotations are missing
    """
    # Check if it's an AnnData object
    if not isinstance(adata_ref, sc.AnnData):
        raise TypeError("adata_ref is not an AnnData object")

    # Check if X is an array with numeric dtype
    if not isinstance(adata_ref.X, np.ndarray) and not scipy.sparse.issparse(adata_ref.X):
        raise TypeError("adata_ref.X is not an array or sparse matrix")
        
    # Check dimensions of X
    p, q = adata_ref.X.shape
    if p <= 0 or q <= 0:
        raise ValueError(f"adata_ref.X has invalid dimensions: {adata_ref.X.shape}")

    # Check if celltype is in obs
    if 'celltype' not in adata_ref.obs:
        raise KeyError("adata_ref.obs does not contain 'celltype'. Please ensure your AnnData object has celltype annotations.")

    # Warning about AnnData dimensions
    print("⚠️ WARNING: In AnnData, rows (obs) should be cells and columns (var) should be genes")
    print(f"adata reference data is valid: {adata_ref.shape[0]} cells x {adata_ref.shape[1]} genes")
    
    return adata_ref
