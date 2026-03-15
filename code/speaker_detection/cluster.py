# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import scipy
import sklearn
from scipy.sparse.linalg import eigsh

from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

try:
    import umap, hdbscan
except ImportError:
    raise ImportError(
        "Package \"umap\" or \"hdbscan\" not found. \
        Please install them first by \"pip install umap-learn hdbscan\".")
class VBxCluster:
    """
    VBx clustering, compatible placeholder.
    Essentially consistent with AHCluster implementation. If subsequent VBx processing is needed, resplitting and other functionalities can be integrated.
"""

    def __init__(self, fix_cos_thr=0.2):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X):
        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion='distance') - 1
        return labels

class SpectralCluster:
    """A spectral clustering method using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, pval=None, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)
        # print(num_of_spk)
        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0]-self.min_pnum)

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k

        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        # k-means
        _, labels, _ = k_means(emb, k, random_state=42)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    """
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With 
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(self, n_neighbors=20, n_components=60, min_samples=20, min_cluster_size=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X):
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0]-2),
            metric=self.metric,
        ).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size).fit_predict(umap_X)
        return labels

class AHCluster:
    """
    Agglomerative Hierarchical Clustering, a bottom-up approach which iteratively merges 
    the closest clusters until a termination condition is reached.
    This implementation is adapted from https://github.com/BUTSpeechFIT/vBx.
    """

    def __init__(self, fix_cos_thr=0.2):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X):
        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion='distance') - 1
        return labels


class CommonClustering:
    """Perfom clustering for input embeddings and output the labels.
    """

    def __init__(self, cluster_type, cluster_line=10, mer_cos=None, min_cluster_size=4, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        print(self.cluster_type)
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        elif self.cluster_type == 'AHC':
            self.cluster = AHCluster(**kwargs)
        elif self.cluster_type == 'VBx':
            self.cluster = VBxCluster(**kwargs)
        else:
            raise valueError(
                '%s is not currently supported.' % self.cluster_type
            )

    def __call__(self, X):
        # clustering and return the labels
        assert len(X.shape) == 2, 'Shape of input should be [N, C]'
        if X.shape[0] < self.cluster_line:
            return np.ones(X.shape[0], dtype=int)
        # clustering
        labels = self.cluster(X)
        # remove extremely minor cluster
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)

        labels = self.arrange_labels(labels)
        # merge similar  speaker
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)
            
        label_set = set(labels)
        centers = np.stack([X[labels == i].mean(0) for i in label_set])
        affinity = cosine_similarity(centers, centers)
        return labels,affinity,centers


    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize < self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels
        
        minor_cset = cset[minor_idx]
        major_idx = np.where(csize >= self.min_cluster_size)[0]
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) \
            for i in major_cset])
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) \
                for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels==c2]=c1
        return labels

    def arrange_labels(self, labels, a_st=0):
        # arrange labels in order from 0.
        new_labels = []
        labels_dict = {}
        idx = a_st
        for i in labels:
            if i not in labels_dict:
                labels_dict[i] = idx
                idx += 1
            new_labels.append(labels_dict[i])
        return np.array(new_labels)