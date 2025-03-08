def proportion(spot_celltype, adata_ref_copy):
  spot_celltype.var['celltype'] = adata_ref_copy.obs['celltype'].cat.categories.copy()
  return spot_celltype
