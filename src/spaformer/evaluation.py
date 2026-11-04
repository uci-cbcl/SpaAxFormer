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

mergedicts = {}
mergedicts['T CD8 memory'] = 'lymphocyte'
mergedicts['T CD8 naive'] = 'lymphocyte'
mergedicts['T CD4 naive'] = 'lymphocyte'
mergedicts['T CD4 memory'] = 'lymphocyte'
mergedicts['Treg'] = 'lymphocyte'
mergedicts['B-cell'] = 'lymphocyte'
mergedicts['plasmablast'] = 'lymphocyte'
mergedicts['NK'] = 'lymphocyte'
mergedicts['monocyte'] = 'Mcell'
mergedicts['macrophage'] = 'Mcell'
mergedicts['mDC'] = 'Mcell'
mergedicts['pDC'] = 'Mcell'
mergedicts['tumors'] = 'tumors'
mergedicts['epithelial'] = 'epithelial'
mergedicts['mast'] = 'mast'
mergedicts['endothelial'] = 'endothelial'
mergedicts['fibroblast'] = 'fibroblast'
mergedicts['neutrophil'] = 'neutrophil'


data_path = "data"
dataset_name = "DLPFC_151673"

run_id = f"T_with_spatial_CGG_no_CCC_16_6"
model = "biFormer"
mode = "nonzero"
mask_file = f"M_eval_mask_1.npy"
mask_idx_file = f"M_eval_idx_1.npy"
mask_rate = 50
epochs = 450

data_set = ["2-spiral", "zebra", "concentric", "Barista_seq", "Merfish", "Visium"]


data = "spatial"
marker_genes = 10
house_keeping_genes = 0
test_folder = f"experiment_3"
radius = 150
HVG = 3000
'''Merfish - 8, BC - 20, Visium DLPFC, STARmap - 7'''
SPOT_SIZE=150 

# hyperparams
NUM_CLUSTER = None
epoch = "450"
num_layers = "4"
loss_type = "batch_15_100_PCs_2d5_z_bin_logX_A"
model_type = "SpaAxFormer"


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
    
    # import rpy2.robjects as ro
    # G_values = ro.IntVector(range(1, 11)) 

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    # res = rmclust(adata.obsm[used_obsm], G_values, modelNames)
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
    
        # marker_gene_w_idx = 0
        # marker_gene_b_idx = 1
        
        # marker_gene_idx = marker_gene_w_idx
        
        # marker_gene = adata.var_names
        
        
        '''Mute this part for real data'''
        marker_gene = adata.var_names[marker_gene_idx]
        
        gt = adata.layers["gt"][:, [marker_gene_idx]]
        masked = adata.layers["masked"][:, [marker_gene_idx]]
        pred = adata.layers["pred"][:, [marker_gene_idx]]
        mask = adata.layers["mask"][:, [marker_gene_idx]]
        
        adata_vis = adata[:, marker_gene_idx].copy()
        
        # adata_vis = adata.copy()
        
        adata_vis.layers["gt"] = gt
        adata_vis.layers["masked"] = masked
        adata_vis.layers["pred"] = pred
        adata_vis.layers["mask"] = mask
        ''' End'''
        
        # adata_vis.layers["gt"] = adata_vis.layers["gt"][:, [marker_gene_idx]]
        # adata_vis.layers["masked"] = adata_vis.layers["masked"][:, [marker_gene_idx]]
        # adata_vis.layers["pred"] = adata_vis.layers["pred"][:, [marker_gene_idx]]
        # adata_vis.layers["mask"] = adata_vis.layers["mask"][:, [marker_gene_idx]]
        
        adata_vis.obsm["spatial"] = pos
        
        layers = ["gt", "mask", "masked", "pred"]
        titles = ["Ground Truth", "mask", "Masked Input", "Our Model"]
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 12))
        
        import pdb
        pdb.set_trace()
        
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
    

def create_cluster_plot(adata, gt, pos, ARI, key='mclust', slice=None, slice_num=None, output_dir="results/run"):
    
    
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
    
    # _sc_1[0].set_title("Ground Truth")
    # plt.savefig(f"figures/{data}/test/cluster_image_{data}_{slice}_{slice_num}_{epoch}_HVG_{HVG}_adjust_cell_embedding_{radius}_knn_10_gene_mask_no_weighted_layer_8_loss_08_02_mask_4_hybrid_CCC_GGG.png")
    # plt.savefig(f"figures/{data}/investigate/cluster_image_{data}_{slice}_{slice_num}_{epoch}_{key}_HVG_{HVG}_cell_embedding_{radius}_no_weighted_layer_8_08_02_mask_4_hybrid_all.png")
    
    # plt.savefig(f"figures/{data}/investigate/contrastive_loss/orth/cluster_image_{data}_{slice}_{slice_num}_{epoch}_{key}_HVG_{HVG}_MLP_05_orth_0005_late_InfoNCE_cell_embedding__no_weighted_layer_8_08_02_mask_4_all_hybrid_150_150_epoch_1500.png")
    
    fig_name = (
        f"cluster_image_{data}"
        f"_{slice}"
        f"_{slice_num}"
        f"_{epoch}"
        f"_{key}"
        f"_{HVG}"
        f"_cell_{num_layers}_L_mask_30_mask_zero_epoch_{epoch}_{model_type}.png"
    )

    
    
    # fig_name = (
    #     f"cluster_image_{data}"
    #     f"_{slice}"
    #     f"_{slice_num}"
    #     f"_{epoch}"
    #     f"_{key}"
    #     f"_{HVG}"
    #     f"_GT_use.png"
    # )
    
    
    out_dir = f"{output_dir}/figures/{data}/investigate/cluster/{slice}/{test_folder}"
    
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


def plot_embedding(adata, key, slice=None, slice_num=None, output_dir="results/run"):
    
    sc.pl.umap(
    adata,
    color=["gt", "mclust"],
    # color = ["gt"],
    ncols=2,
    size=20,
    title=[f"PCA {key}", f"mclust {key}"],
    # title=[f"GT {key}"],
    show=False
    )
    
    fig_name = (
        f"Umap_image_{data}"
        f"_{slice}"
        f"_{slice_num}"
        f"_{epoch}"
        f"_{key}"
        f"_{HVG}"
        f"_cell_{num_layers}_L_mask_30_mask_zero_epoch_{epoch}_{model_type}.png"
    )
    
    # _10_05_05_extra_neg_20_contrast_False_001
    
    # fig_name = (
    #     f"Umap_image_{data}"
    #     f"_{slice}"
    #     f"_{slice_num}"
    #     f"_{epoch}"
    #     f"_{key}"
    #     f"_{HVG}"
    #     f"_GT_use.png"
    # )
    
    
    
    out_dir = f"{output_dir}/figures/{data}/investigate/cluster/{slice}/{test_folder}/umap"
    
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
    
    # pm = pm1
    # pm2 = pm1
    
    
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
    
    
    
    # metrics_dict = {"Pearson": float(r), "Separman": float(rho), "RMSE": float(test_rmse)}
    # config_dict = {"mask": mask_file, "mask_rate": mask_rate, "epochs": 250}
    
    # outdir = os.path.join(data_path, "runs", dataset_name, model, mode)
    # Path(outdir).mkdir(parents=True, exist_ok=True)

    # np.save(os.path.join(outdir, f"pred_{run_id}.npy"), y_pred)
    
    # '''divider'''
    # csv_path = Path(os.path.join(outdir, "metrics_CGG_no_CCC_16.csv"))
    # df = pd.DataFrame([metrics_dict])
    # df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        
    # with open(os.path.join(outdir, "config.json"), "w") as f:
    #     json.dump(config_dict, f, indent=2)
    
    
    return test_l1, test_cos_sim, test_rmse
    
    

        

def comprehensive_evaluate(cell_embedding, gex, mask, pos, lbl, logger = None, emb = None, slice=None, slice_num=None, refinement=True, num_clusters=None, output_dir="results/run"):
    if num_clusters is None:
        raise ValueError("num_clusters must be explicitly configured")
    NUM_CLUSTER = num_clusters
    from sklearn.preprocessing import LabelEncoder
    gt = LabelEncoder().fit_transform(lbl)
    
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    
    '''Normalize embedding and combine cell embedding with gene embedding'''
    # gex_normalized = row_normalize(gex)
    # weighted_gene_emb = gex_normalized @ gene_embedding
    
    # emb = 0.8 * cell_embedding + 0.2 * weighted_gene_emb 
    

    mean = cell_embedding.mean(axis=0, keepdims=True)
    std  = cell_embedding.std(axis=0, keepdims=True) + 1e-6
    X_std = (cell_embedding - mean) / std
    
    
    # emb = cell_embedding
    # emb = X_std
    # X_std = cell_embedding
    
    d = min(30, cell_embedding.shape[1], cell_embedding.shape[0]-1)
    X_pca = PCA(n_components=d, random_state=0).fit_transform(cell_embedding)
    
    # X_pca = cell_embedding
    
    
    pred_km = KMeans(n_clusters=NUM_CLUSTER, n_init="auto", random_state=0).fit_predict(X_pca)
    print("KMeans ARI:", adjusted_rand_score(gt, pred_km))
    
    gmm = GaussianMixture(n_components=NUM_CLUSTER, covariance_type="full",
                      reg_covar=1e-4, random_state=0)
    pred_gmm = gmm.fit_predict(X_pca)
    print("GMM ARI:", adjusted_rand_score(gt, pred_gmm))
    
    emb = X_pca
    
    # emb = cell_embedding
    
    # import sys
    # sys.exit()
    
    
    # emb = weighted_gene_emb
    
    # A_w  = zscore(cell_embedding)
    # ZB_w = zscore(weighted_gene_emb)
    
    # p = 64
    # A_p  = PCA(p, random_state=0).fit_transform(A_w)
    # ZB_p = PCA(p, random_state=0).fit_transform(ZB_w)
    
    # lam = 0.1
    # emb = (A_p + lam*ZB_p) / np.sqrt(1 + lam**2)

    
    # weighted_gene_emb 
    
    # import pdb
    # pdb.set_trace()
    
    # r = pearsonr(y_true, y_pred)[0]
    # rho = spearmanr(y_true, y_pred)[0]
    # rmse = np.sqrt(((y_true - y_pred)**2).mean())
    # print(f"Pearson={r:.4f}  Spearman={rho:.4f}  RMSE={rmse:.4f}")
    
    
    '''mute for now'''    
    # metrics_dict = {"Pearson": float(r), "Separman": float(rho), "RMSE": float(rmse)}
    # config_dict = {"mask": mask_file, "mask_rate": mask_rate, "epochs": 250}
    
    # outdir = os.path.join(data_path, "runs", dataset_name, model, mode)
    # Path(outdir).mkdir(parents=True, exist_ok=True)

    # np.save(os.path.join(outdir, f"pred_{run_id}.npy"), y_pred)
    
    # '''divider'''
    # csv_path = Path(os.path.join(outdir, "metrics_CGG_no_CCC_16.csv"))
    # df = pd.DataFrame([metrics_dict])
    # df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        
    # with open(os.path.join(outdir, "config.json"), "w") as f:
    #     json.dump(config_dict, f, indent=2)
        

    # mask = mask.astype(float)
    
    # # import pdb
    # # pdb.set_trace()
    
    # '''This parts need to be debugged thoroughly - mask & 1-mask'''
    # pred = np.multiply(gex, (1-mask)) + np.multiply(temp, mask)

    # pred_mean = pred.mean()
    # gex_mean = gex.mean()
    
    # logging.info(f'Pred mean {pred_mean} True mean {gex_mean}')
    # logging.info(f'Overall rmse: {np.sqrt(np.mean((pred-gex)**2))}')
    # pm = pred[mask.astype(bool)]
    # tm = gex[mask.astype(bool)]
    # mrmse = np.sqrt(np.mean((pm-tm)**2))
    # mcos = np.sum(pm*tm)/np.sqrt(np.sum(pm*pm)*np.sum(tm*tm))
    # mpe = pe(pm, tm)[0]
    # logging.info(f'Masked rmse: {mrmse}')
    # logging.info(f'Masked cosine: {mcos}')
    # logging.info(f'Masked pearson: {mpe}')
    
    # lbl = lbl[lbl != "unknown"]
    
    '''for now, for test purpose (We should make this work for ARI, NMI score is present)'''
    
    '''Create plot'''
    pos = pos.detach().cpu().numpy()
    
    pred = gex
    
    # adata_all = sc.AnnData(pred)
            
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
    
    # from sklearn.preprocessing import StandardScaler
    # adata.obsm[key] = StandardScaler().fit_transform(adata.obsm[key])
    
    Z = emb
    
    
    print("NaN count:", np.isnan(Z).sum())
    '''Look into this'''
    print("Std across features:", np.std(Z, axis=0).mean(), "±", np.std(Z, axis=0).std())
    print("Min/Max:", Z.min(), Z.max())
    
    
    
    # adata = sc.AnnData(latent_embedding)
    
    # sc.pp.pca(adata, n_comps=20, random_state=1)
    
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
    
    plot_embedding(adata, key, slice=slice, slice_num=slice_num, output_dir=output_dir)
    
    '''Visium - 7, BC - 20, STARmap - 7'''
    # NUM_CLUSTER = None
    # NUM_CLUSTER = 20
    
    # adata = mclust_py(adata, num_cluster=7, used_obsm=key)

    
    # from sklearn.cluster import KMeans
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    # from sklearn.metrics import adjusted_rand_score
    
    # import pdb
    # pdb.set_trace()
    
    
    pred_labels = adata.obs['mclust']
    
    NMI_score = normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic')
    ARI_score = adjusted_rand_score(gt, pred_labels)
    
    print("regular: \n", ARI_score, NMI_score)
    
    create_cluster_plot(adata, gt, pos, ARI_score, slice = slice, slice_num=slice_num, output_dir=output_dir)
    
    '''This is for refining labels'''
    
    radius = 50
    
    if refinement:
          
        new_type = refine_label(adata, radius, key='mclust')
        adata.obs['refined'] = new_type 
        pred_labels = adata.obs['refined']
        
    NMI_score_refined = normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic')
    ARI_score_refined = adjusted_rand_score(gt, pred_labels)
    
    print("refined: \n", ARI_score_refined, NMI_score_refined)
    
    
    
    create_cluster_plot(adata, gt, pos, ARI_score_refined, slice = slice, slice_num=slice_num, key='refined', output_dir=output_dir)
    
    return ARI_score, ARI_score_refined, NMI_score, NMI_score_refined
    
    # logging.info(f'NMI: {NMI_score}')
    # logging.info(f'ARI: {ARI_score}')
    # if logger is not None:
    #     logger.note({'rmse': mrmse, 'cosine': mcos, 'pearson': mpe, 'nmi': NMI_score, 'ARI': ARI_score})


    
    # adata = sc.AnnData(gex)
    # sc.pp.pca(adata, n_comps=10, random_state=1)
    # sc.pp.neighbors(adata, metric='correlation')
    # sc.tl.leiden(adata, resolution=0.26, random_state=1)

    # for res in [0.3, 0.32, 0.34, 0.36]:
    #     if len(adata.obs['leiden'].unique()) >= gt.max()+1:
    #         break
    #     sc.tl.leiden(adata, resolution=res, random_state=1)
        
    # pred_labels = adata.obs['leiden']
    # gt = lbl
    # logging.info(f'{len(pred_labels.unique())} clusters detected  |  Ground-truth labels have {gt.max()+1} clusters.')

    # from sklearn.cluster import KMeans
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    # from sklearn.metrics import adjusted_rand_score
    # #pred = TSNE(3, init='random', learning_rate='auto',  perplexity=10, n_jobs=16).fit_transform(pred)
    # #kmeans = KMeans(n_clusters=18, random_state=200)
    # #pred_labels = kmeans.fit_predict(pred)
    


    # NMI_score = round(normalized_mutual_info_score(gt, pred_labels, average_method='arithmetic'), 3)
    # ARI_score = round(adjusted_rand_score(gt, pred_labels), 3)

    # logging.info(f'NMI: {NMI_score}')
    # logging.info(f'ARI: {ARI_score}')
    # if logger is not None:
    #     logger.note({'rmse': mrmse, 'cosine': mcos, 'pearson': mpe, 'nmi': NMI_score, 'ARI': ARI_score})


def save_emb(cell_embedding, gene_embedding, slice, slice_num, save_dir="results/run/embeddings"):
    
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"cell_embedding_{slice}_{slice_num}.npy"), cell_embedding)
    np.save(os.path.join(save_dir, f"gene_embedding_{slice}_{slice_num}.npy"), gene_embedding)


def zscore(Z):  # per-feature
    return StandardScaler().fit_transform(Z)