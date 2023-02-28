import os
import warnings
from pathlib import Path

import random
import pandas as pd
import patsy
import numpy as np
import ot
import seaborn as sns
import time

import matplotlib.pyplot as plt
import sklearn.manifold as skmf
import sklearn.decomposition as skdc
import sklearn.metrics as skmr
from lxml import etree
from umap import UMAP

import condo

from combat import combat

this_file = os.path.realpath('__file__')
# data_path = os.path.join(Path(this_file).parent.parent, 'data')
data_path = 'data/rxrx1'

expr = pd.read_csv(os.path.join(data_path, 'expr.csv'),)
expr_linear = pd.read_csv(os.path.join(data_path, 'expr_linear.csv'))
expr_mmd = pd.read_csv(os.path.join(data_path, 'expr_mmd.csv'))
expr_clinear = pd.read_csv(os.path.join(data_path, 'expr_clinear.csv'))
expr_pogmm = pd.read_csv(os.path.join(data_path, 'expr_pogmm.csv'))
expr_cmmd = pd.read_csv(os.path.join(data_path, 'expr_cmmd.csv'))

sns.set_context("talk")
dinfos = [
    (0, "Original", expr),
    #(1, "Combat", expr_combat),
    (1, "Gaussian OT", expr_linear),
    (2, "MMD", expr_mmd),
    (3, "ConDo Linear-ReverseKL", expr_clinear),
    (4, "ConDo PoGMM-ReverseKL", expr_pogmm),
    # (5, "ConDo GP-ReverseKL", expr_cgp),
    (5, "ConDo MMD", expr_cmmd),
]
fig = plt.figure(figsize=(5, 23), constrained_layout=True)
subfigs = fig.subfigures(nrows=len(dinfos), ncols=1)

for dix, dname, dset in dinfos:
    axes = subfigs[dix].subplots(nrows=1, ncols=3)
    sil_result = skmr.silhouette_score(dset, pheno.sirna_id, metric='euclidean')
    sil_batch = skmr.silhouette_score(dset, pheno.experiment, metric='euclidean')
    db_result = skmr.davies_bouldin_score(dset, pheno.sirna_id)
    db_batch = skmr.davies_bouldin_score(dset, pheno.experiment)
    dtitle = f"{dname}\n{db_batch:.2f} (batch), {db_result:.2f} (result)"
    # dtitle = f"{dname}\n{sil_batch:.2f} (batch), {sil_result:.2f} (result)"
    subfigs[dix].suptitle(dtitle);  

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsner = skmf.TSNE(n_components=2)
        tsne_embed = tsner.fit_transform(dset)
        tsne_data = pheno.copy()
        tsne_data["tsne 1"] = tsne_embed[:, 0]
        tsne_data["tsne 2"] = tsne_embed[:, 1]

    pcaer = skdc.PCA(n_components=2)
    pca_embed = pcaer.fit_transform(dset)
    pca_data = pheno.copy()
    pca_data["pc 1"] = pca_embed[:, 0]
    pca_data["pc 2"] = pca_embed[:, 1]       
    
    umaper = UMAP(n_components=2)
    umap_embed = umaper.fit_transform(dset)
    umap_data = pheno.copy()
    umap_data["umap 1"] = umap_embed[:, 0]
    umap_data["umap 2"] = umap_embed[:, 1]       
    sns.scatterplot(
        data=tsne_data, ax=axes[0],
        x="tsne 1", y="tsne 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=False,)
    if dix < 0:
        leg = "auto"
    else:
        leg = False
    sns.scatterplot(
        data=pca_data, ax=axes[1],
        x="pc 1", y="pc 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=leg);
    sns.scatterplot(
        data=umap_data, ax=axes[2],
        x="umap 1", y="umap 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=leg,)
    """
    if dix == 0:
        plt.legend(
            loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=2,
        );
    """
    """
    #(_, max_x) = axes[1].get_xlim();
    #(_, max_x) = axes[1].get_xlim();

    axes[1].text(
        #np.max(pca_embed[:,0])*2.0, 0.5, 
        1.2, 0.5,
        f"Silhouette\nBatch:{sil_batch:.2f}\nOutcome:{sil_result:.2f}",
        transform = axes[1].transAxes); 
    """
    axes[0].set_xticks([]);
    axes[1].set_xticks([]);
    axes[2].set_xticks([]);
    axes[0].set_yticks([]);
    axes[1].set_yticks([]);
    axes[2].set_yticks([]);
    axes[0].set_xlabel(None);
    axes[1].set_xlabel(None);
    axes[2].set_xlabel(None);
    axes[0].set_ylabel(None);
    axes[1].set_ylabel(None);
    axes[2].set_ylabel(None);
# fig.savefig("figure-Bladderbatch-noconfounding.pdf", bbox_inches="tight")
fig.savefig("results/figure-RxRx1-db-HUVEC-24_HUVEC-04-noconfounding.pdf", bbox_inches="tight")

sns.set_context("talk")
dinfos = [
    (0, "Original", expr),
    #(1, "Combat", expr_combat),
    (1, "Gaussian OT", expr_linear),
    (2, "MMD", expr_mmd),
    (3, "ConDo Linear-ReverseKL", expr_clinear),
    (4, "ConDo PoGMM-ReverseKL", expr_pogmm),
    # (5, "ConDo GP-ReverseKL", expr_cgp),
    (5, "ConDo MMD", expr_cmmd),
]
fig = plt.figure(figsize=(5, 23), constrained_layout=True)
subfigs = fig.subfigures(nrows=len(dinfos), ncols=1)

for dix, dname, dset in dinfos:
    axes = subfigs[dix].subplots(nrows=1, ncols=3)
    sil_result = skmr.silhouette_score(dset, pheno.sirna_id, metric='euclidean')
    sil_batch = skmr.silhouette_score(dset, pheno.experiment, metric='euclidean')
    db_result = skmr.davies_bouldin_score(dset, pheno.sirna_id)
    db_batch = skmr.davies_bouldin_score(dset, pheno.experiment)
    # dtitle = f"{dname}\n{db_batch:.2f} (batch), {db_result:.2f} (result)"
    dtitle = f"{dname}\n{sil_batch:.2f} (batch), {sil_result:.2f} (result)"
    subfigs[dix].suptitle(dtitle);  

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsner = skmf.TSNE(n_components=2)
        tsne_embed = tsner.fit_transform(dset)
        tsne_data = pheno.copy()
        tsne_data["tsne 1"] = tsne_embed[:, 0]
        tsne_data["tsne 2"] = tsne_embed[:, 1]

    pcaer = skdc.PCA(n_components=2)
    pca_embed = pcaer.fit_transform(dset)
    pca_data = pheno.copy()
    pca_data["pc 1"] = pca_embed[:, 0]
    pca_data["pc 2"] = pca_embed[:, 1]       
    
    umaper = UMAP(n_components=2)
    umap_embed = umaper.fit_transform(dset)
    umap_data = pheno.copy()
    umap_data["umap 1"] = umap_embed[:, 0]
    umap_data["umap 2"] = umap_embed[:, 1]       
    sns.scatterplot(
        data=tsne_data, ax=axes[0],
        x="tsne 1", y="tsne 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=False,)
    if dix < 0:
        leg = "auto"
    else:
        leg = False
    sns.scatterplot(
        data=pca_data, ax=axes[1],
        x="pc 1", y="pc 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=leg);
    sns.scatterplot(
        data=umap_data, ax=axes[2],
        x="umap 1", y="umap 2", hue="sirna_id", style="experiment", 
        edgecolor="none", legend=leg,)
    """
    if dix == 0:
        plt.legend(
            loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=2,
        );
    """
    """
    #(_, max_x) = axes[1].get_xlim();
    #(_, max_x) = axes[1].get_xlim();

    axes[1].text(
        #np.max(pca_embed[:,0])*2.0, 0.5, 
        1.2, 0.5,
        f"Silhouette\nBatch:{sil_batch:.2f}\nOutcome:{sil_result:.2f}",
        transform = axes[1].transAxes); 
    """
    axes[0].set_xticks([]);
    axes[1].set_xticks([]);
    axes[2].set_xticks([]);
    axes[0].set_yticks([]);
    axes[1].set_yticks([]);
    axes[2].set_yticks([]);
    axes[0].set_xlabel(None);
    axes[1].set_xlabel(None);
    axes[2].set_xlabel(None);
    axes[0].set_ylabel(None);
    axes[1].set_ylabel(None);
    axes[2].set_ylabel(None);
# fig.savefig("figure-Bladderbatch-noconfounding.pdf", bbox_inches="tight")
fig.savefig("results/figure-RxRx1-sil-HUVEC-24_HUVEC-04-noconfounding.pdf", bbox_inches="tight")

sns.set_context("talk")
sns.set(style="darkgrid", font_scale=1.2)

dinfos = [
    ("Gaussian OT", pd.DataFrame(expr_linear, columns=expr.columns, index=expr.index)),
    ("MMD", pd.DataFrame(expr_mmd, columns=expr.columns, index=expr.index)),
    ("ConDo Linear-ReverseKL", pd.DataFrame(expr_clinear, columns=expr.columns, index=expr.index)),
    ("ConDo PoGMM-ReverseKL", pd.DataFrame(expr_pogmm, columns=expr.columns, index=expr.index)),
    ("ConDo MMD", pd.DataFrame(expr_cmmd, columns=expr.columns, index=expr.index)),
]
num_feats = 5
feat_idxs = random.choices(expr.columns, k=num_feats)

for dname, dset in dinfos:
    fig = plt.figure(figsize=(num_feats, 23), constrained_layout=True)
    fig.suptitle(dname)
    subfigs = fig.subfigures(nrows=num_feats, ncols=1)
    for i, idx in enumerate(feat_idxs):
        axes = subfigs[i].subplots(nrows=1, ncols=1)
        dtitle = f"Feature = {idx}"
        subfigs[i].suptitle(dtitle);  

        bins = np.linspace(-10, 10, 50)

        axes.hist(expr[idx], bins, alpha=0.5, label='Original', color = "blue")
        axes.hist(dset[idx], bins, alpha=0.5, label='Corrected', color='green')
        axes.legend(loc='upper right')
        
        axes.set_xticks([]);
        axes.set_yticks([]);
        axes.set_xlabel(None);
        axes.set_ylabel(None);
    fig.savefig(f"results/figure-feat_dist-{dname}-RxRx1-noconfounding.pdf", bbox_inches="tight")