import numpy as np
import os
import ot
from scipy.stats import pearsonr as pe
import scanpy as sc
import networkx as nx
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


data_path = "data"
dataset_name = "DLPFC_151673"

data = "spatial"
marker_genes = 10
house_keeping_genes = 0
radius = 150
'''Merfish - 8, BC - 20, Visium DLPFC, STARmap - 7'''
SPOT_SIZE=150 

# hyperparams
NUM_CLUSTER = None
epoch = "150"
loss_type = "batch_15_100_PCs_2d5_z_bin_logX_A"


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='X_pca', random_seed=2022):
    # previous random_seed = 666
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)      

from sklearn.mixture import GaussianMixture
import numpy as np

'''normally cluster = 7, BC - 20'''
def mclust_py(adata, num_cluster=7, used_obsm='X_pca', random_state=666):
    Z = np.array(adata.obsm[used_obsm], dtype=np.float64)
    gmm = GaussianMixture(n_components=num_cluster, covariance_type='full',
                          random_state=random_state, reg_covar=1e-5)
    labels = gmm.fit_predict(Z)
    adata.obs['mclust'] = pd.Categorical(labels)
    return adata

# create for both black and white
def create_plot(adata, pos):
    
    marker_gene_indices = [0,1]
    
    for marker_gene_idx in marker_gene_indices:        
        sc.set_figure_params(figsize=(12, 12), dpi=100)
    
        marker_gene = adata.var_names[marker_gene_idx]
        
        gt = adata.layers["gt"][:, [marker_gene_idx]]
        masked = adata.layers["masked"][:, [marker_gene_idx]]
        pred = adata.layers["pred"][:, [marker_gene_idx]]
        mask = adata.layers["mask"][:, [marker_gene_idx]]
        
        adata_vis = adata[:, marker_gene_idx].copy()
        
        adata_vis.layers["gt"] = gt
        adata_vis.layers["masked"] = masked
        adata_vis.layers["pred"] = pred
        adata_vis.layers["mask"] = mask
        
        adata_vis.obsm["spatial"] = pos
        
        layers = ["gt", "mask", "masked", "pred"]
        titles = ["Ground Truth", "mask", "Masked Input", "Our Model"]
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 12))
        
        for i, (lyr, ttl) in enumerate(zip(layers, titles)):
            ax = axes[i]
            sc.pl.spatial(
                adata_vis,
                color=marker_gene,
                layer=lyr,
                img_key=None,
                spot_size=200,
                ax=ax,
                title=ttl,
                show=False,
                cmap="viridis"
            )
        
        plt.tight_layout()
        plt.savefig(f"figures/{data}/marker_gene_composite_{marker_genes}_{house_keeping_genes}_{data}_{epoch}_3_{marker_gene_idx}_eval_mask_all_gene_new_test.png", dpi=300)
        plt.close()
    

def _plot_run_name(dataset, sample, repeat, epochs, num_layers):
    fields = (dataset, sample,
              f"repeat{repeat}" if repeat is not None else None,
              f"epochs{epochs}" if epochs is not None else None,
              f"layers{num_layers}" if num_layers is not None else None)
    return "_".join(str(value) for value in fields if value is not None)


def create_cluster_plot(adata, gt, pos, ARI, key='mclust', slice=None, slice_num=None, output_dir="results/run", *, dataset=None, epochs=None, num_layers=None):
    
    
    # SPOT_SIZE = 200 - Visium
    '''200 - Visium, 300 - BC, 200-STARmap, MERFISH-50'''
    SPOT_SIZE=200
    TITLE_SIZE = 12
    
    
    fig, ax = plt.subplots(1, 2, figsize=(30, 30), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    _sc_0 = sc.pl.spatial(adata, img_key=None, color=key, title=[''],
                            legend_fontsize=20, ax=ax[0], show=False,frameon=False,
                            spot_size=SPOT_SIZE)
        # legend_loc = None
    _sc_0[0].set_title("ARI=" + str(ARI), size=TITLE_SIZE)
    _sc_1 = sc.pl.spatial(adata, img_key=None, color=['gt'], title=[''],
                            legend_fontsize=20, show=False, ax=ax[1], frameon=False,
                            spot_size=SPOT_SIZE)
    
    run_name = _plot_run_name(dataset, slice, slice_num, epochs, num_layers)
    fig_name = f"cluster_{run_name}_{key}.png"
    
    out_dir = str(Path(output_dir) / "figures" / (dataset or "") / "investigate" / "cluster" / str(slice))
    
    save_path = os.path.join(out_dir, fig_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


''''''
'''code from: https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/utils.py'''
def refine_label(adata, radius=50, key='label'):
    
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


def row_normalize(X, eps=1e-8):
    return X / (X.sum(axis=1, keepdims=True) + eps)


def plot_embedding(adata, key, slice=None, slice_num=None, output_dir="results/run", *, dataset=None, epochs=None, num_layers=None):
    
    sc.pl.umap(
    adata,
    color=["gt", "mclust"],
    ncols=2,
    size=20,
    title=[f"PCA {key}", f"mclust {key}"],
    show=False
    )
    
    run_name = _plot_run_name(dataset, slice, slice_num, epochs, num_layers)
    fig_name = f"umap_{run_name}_{key}.png"
    
    out_dir = str(Path(output_dir) / "figures" / (dataset or "") / "investigate" / "cluster" / str(slice) / "umap")
    
    save_path = os.path.join(out_dir, fig_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("saved_embedding")
    



def comprehensive_evaluate_test(X_pred, gex, mask): 
    
    
    pm = X_pred[mask]
    tm = gex[mask]
    
    y_pred = pm
    y_true = tm
       
    test_mse = np.square(y_true - y_pred).mean()
    test_l1 = np.abs(y_true - y_pred).mean()    
    test_cos_sim = cosine_similarity(np.asarray(y_pred).reshape(1, -1), np.asarray(y_true).reshape(1, -1))[0][0]
    test_cos_sim_2 = np.sum(pm*tm)/np.sqrt(np.sum(pm*pm)*np.sum(tm*tm))
    test_rmse = np.sqrt(test_mse)
    mrse = np.sqrt(np.mean((pm-tm)**2))
    
    pearsonval = pearsonr(tm.ravel(), pm.ravel())[0]
    spearman = spearmanr(tm.ravel(), pm.ravel())[0]
    rho = spearmanr(y_true.ravel(), y_pred.ravel())[0]
    
    print("test_l1: ", test_l1)
    print("test_cos_sim: ", test_cos_sim)
    print("test_rmse: ", test_rmse)
    print("test_cos_sim_2: ", test_cos_sim_2)
    print("\n")
    print("pearson r: ", pearsonval)
    print("spearman: ", rho)
    print("test_rmse_2: ", mrse)
    
    print("\n")
    print("test mse: ", test_mse)
    
    print("\n")
    print("pearson: ", pearsonval)
    print("spearman1: ", spearman)
    
    return test_l1, test_cos_sim, test_rmse
    
    

        

def comprehensive_evaluate(cell_embedding, gex, mask, pos, lbl, logger = None, emb = None, slice=None, slice_num=None, refinement=True, num_clusters=None, output_dir="results/run", *, dataset=None, epochs=None, num_layers=None):
    if num_clusters is None:
        raise ValueError("num_clusters must be explicitly configured")
    NUM_CLUSTER = num_clusters
    from sklearn.preprocessing import LabelEncoder
    gt = LabelEncoder().fit_transform(lbl)
    
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    mean = cell_embedding.mean(axis=0, keepdims=True)
    std  = cell_embedding.std(axis=0, keepdims=True) + 1e-6
    X_std = (cell_embedding - mean) / std
    
    d = min(30, cell_embedding.shape[1], cell_embedding.shape[0]-1)
    X_pca = PCA(n_components=d, random_state=0).fit_transform(cell_embedding)
        
    pred_km = KMeans(n_clusters=NUM_CLUSTER, n_init="auto", random_state=0).fit_predict(X_pca)
    print("KMeans ARI:", adjusted_rand_score(gt, pred_km))
    
    gmm = GaussianMixture(n_components=NUM_CLUSTER, covariance_type="full",
                      reg_covar=1e-4, random_state=0)
    pred_gmm = gmm.fit_predict(X_pca)
    print("GMM ARI:", adjusted_rand_score(gt, pred_gmm))
    
    emb = X_pca
     
    '''Create plot'''
    pos = pos.detach().cpu().numpy()
    
    pred = gex
             
    '''This will create cluster for gex - log-transformed gene expression'''
    adata_all = sc.AnnData(gex)
    
    adata = adata_all[:, 0:2].copy()
    
    adata = adata_all.copy()
    
    adata_all.layers["pred"] = pred
    
    '''log transformed gene expression with all entries'''
    adata_all.layers["gt"] = gex
    
    # adata_all.layers["masked"] = zero_out_x
    
    adata_all.layers["mask"] = mask
    
    # create_plot(adata_all, pos)
    
    
    key = "embedding"
    
    adata.obsm[key] = emb
    
    Z = emb
    
    
    print("NaN count:", np.isnan(Z).sum())
    '''Look into this'''
    print("Std across features:", np.std(Z, axis=0).mean(), "±", np.std(Z, axis=0).std())
    print("Min/Max:", Z.min(), Z.max())
      
    adata.obs["gt"] = gt
    adata.obs["gt"] = adata.obs["gt"].astype("category")
    
    shared_colors = ["#2ecc71", "#3498db"]
    
    adata.uns["mclust_colors"] = shared_colors
    adata.uns["gt_colors"] = shared_colors
    adata.obsm["spatial"] = pos
    
    
    
    sc.pp.neighbors(adata, use_rep=key, random_state=666)
    sc.tl.umap(adata, min_dist=0.1, spread=1.0)
    
    
    adata = mclust_R(adata, num_cluster=NUM_CLUSTER, used_obsm=key)
    
    adata.obs['mclust'] = pd.Series(match_cluster_labels(gt, adata.obs['mclust'].values),
                                            index=adata.obs.index, dtype='category')
    
    plot_embedding(adata, key, slice=slice, slice_num=slice_num, output_dir=output_dir,
                   dataset=dataset, epochs=epochs, num_layers=num_layers)
        
    pred_labels = adata.obs['mclust']
    
    NMI_score = normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic')
    ARI_score = adjusted_rand_score(gt, pred_labels)
    
    print("regular: \n", ARI_score, NMI_score)
    
    create_cluster_plot(adata, gt, pos, ARI_score, slice = slice, slice_num=slice_num, output_dir=output_dir,
                        dataset=dataset, epochs=epochs, num_layers=num_layers)
    
    '''This is for refining labels'''
    
    radius = 50
    
    if refinement:
          
        new_type = refine_label(adata, radius, key='mclust')
        adata.obs['refined'] = new_type 
        pred_labels = adata.obs['refined']
        
    NMI_score_refined = normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic')
    ARI_score_refined = adjusted_rand_score(gt, pred_labels)
    
    print("refined: \n", ARI_score_refined, NMI_score_refined)
    
    
    
    create_cluster_plot(adata, gt, pos, ARI_score_refined, slice = slice, slice_num=slice_num, key='refined', output_dir=output_dir,
                        dataset=dataset, epochs=epochs, num_layers=num_layers)
    
    return ARI_score, ARI_score_refined, NMI_score, NMI_score_refined

def save_emb(cell_embedding, gene_embedding, slice, slice_num, save_dir="results/run/embeddings"):
    
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"cell_embedding_{slice}_{slice_num}.npy"), cell_embedding)
    np.save(os.path.join(save_dir, f"gene_embedding_{slice}_{slice_num}.npy"), gene_embedding)


def zscore(Z):  # per-feature
    return StandardScaler().fit_transform(Z)