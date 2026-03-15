import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict,Counter
import warnings
from typing import Any, Callable, Dict, Tuple, Union
from sklearn.cluster import KMeans
vflag = True


def get_most_common_or_by_score(counts, scores):
    if len(counts) == 0:
        return None
    count_values = list(counts.values())
    if len(set(count_values)) == 1 and len(count_values) > 1:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        return max(counts.items(), key=lambda x: x[1])[0]

def compute_mean_embedding(
    df: pd.DataFrame,
    condition_func: callable
) -> np.ndarray:  

    embeddings = []

    for _, row in df.iterrows():
        if condition_func(row):
            emb = load_embedding(row['files'])
            embeddings.append(emb)
    mean_emb = np.mean(np.array(embeddings), axis=0).reshape(1, -1)
    cnt = len(embeddings)
    return mean_emb,cnt

def compute_class_embeddings_and_similarity(
    df: pd.DataFrame,
    label_col: str,
    condition_func: callable = None
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, int], Dict[Any, Dict[Any, float]]]:

    unique_labels = set(df[label_col].dropna())

    class_embeddings = {}
    class_counts = {}

    for label in unique_labels:
        cond_func = condition_func(label)
        mean_emb, count = compute_mean_embedding(
            df=df,
            condition_func=cond_func
        )

        class_embeddings[label] = mean_emb
        class_counts[label] = count

    similarity_dict = defaultdict(dict)
    labels = list(class_embeddings.keys())
    for i, k1 in enumerate(labels):
        emb1 = class_embeddings[k1]
        for j, k2 in enumerate(labels):
            if j <= i:
                continue
            emb2 = class_embeddings[k2]
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarity_dict[k1][k2] = sim
            similarity_dict[k2][k1] = sim

    return class_embeddings, class_counts, dict(similarity_dict)


def load_embedding(file_path):
    """Load embedding from a .npy file"""
    emb_file = file_path.replace('.wav', '_src.npy').replace('../数据','/home/jovyan/work/qirui/speaker_recognition/data')
    return np.load(emb_file)
    
def compute_similarity(embedding1, embedding2):
    """Compute similarity between two embeddings"""
    return cosine_similarity(
        embedding1.reshape(1, -1), embedding2.reshape(1, -1)
    )[0][0]

def get_window_df(df, idx, offset):
    """Get a segment of the DataFrame within the specified window"""
    start_index = max(idx - offset, 0)
    end_index = min(idx + offset, len(df) - 1)
    return df.iloc[start_index:end_index+1]

def get_unique_speakers(df_window, label_col='vision_pred'):
    """Get deduplicated, non-null speakers from the window"""
    return df_window[label_col].dropna().unique()


def calculate_weighted_scores(sample_emb, embedding2, weights=None):
    scores = {}
    weighted_scores = {}

    if weights:
        for speaker in weights:
            if speaker in sample_emb:
                emb = sample_emb[speaker]
                result = compute_similarity(emb, embedding2)
                scores[speaker] = result
                weighted_scores[speaker] = result * weights.get(speaker, 1)
            else:
                scores[speaker] = 0.0
                weighted_scores[speaker] = 0.0
    else:
        weights = defaultdict(lambda: 1)
        for speaker in sample_emb:
            emb = sample_emb[speaker]
            result = compute_similarity(emb, embedding2)
            scores[speaker] = result
            weighted_scores[speaker] = result * weights.get(speaker, 1)
    return scores, weighted_scores
from collections import defaultdict


def calculate_weighted_scores_by_clusteremb(sample_emb, embedding2, weights=None):
    scores = {}
    weighted_scores = {}

    if weights:
        for speaker in weights:
            if speaker in sample_emb:
                sims = [compute_similarity(center, embedding2) for center in sample_emb[speaker]]
                best_sim = max(sims)
                scores[speaker] = best_sim
                weighted_scores[speaker] = best_sim * weights.get(speaker, 1)
            else:
                scores[speaker] = 0.0
                weighted_scores[speaker] = 0.0
    else:
        weights = defaultdict(lambda: 1)
        for speaker in sample_emb:
            sims = [compute_similarity(center, embedding2) for center in sample_emb[speaker]]
            best_sim = max(sims)
            scores[speaker] = best_sim
            weighted_scores[speaker] = best_sim * weights.get(speaker, 1)
    return scores, weighted_scores

def get_speakers_by_emb(df, sample_face_emb):
    matched_roles_per_row = []     
    matched_scores_per_row = []
    all_valid_roles = set()        

    for idx, row in df.iterrows():
        face_embs = row.get('vision_embeddings_all_role', [])
        matched_roles_per_emb = []
        matched_scores_per_emb = []
        for emb in face_embs:
            scores, _ = calculate_weighted_scores(sample_face_emb, emb)
            max_score = -float('inf')
            max_role = None
            for role, score in scores.items():
                if score > max_score:
                    max_score = score
                    max_role = role
            matched_roles_per_emb.append(max_role)
            matched_scores_per_emb.append(max_score)
            if max_role is not None and max_score > 0.5:
                all_valid_roles.add(max_role)
        matched_roles_per_row.append(matched_roles_per_emb)
        matched_scores_per_row.append(matched_scores_per_emb)

    return matched_roles_per_row,matched_scores_per_row

def get_unique_speakers_byface(df_window):
    """
    For each row in the window, retrieve all face embeddings (from the 'vision_embeddings_all_role' column, which may contain multiple embeddings per row).
    Each face embedding is compared with sample_face_emb to find the role with the highest similarity (using calculate_weighted_scores).
    If the highest similarity score is greater than 0.5, it is considered a valid matched role.
    The function returns two results: (1) the deduplicated list of valid matched roles, (2) the roles matched for each row.
"""

    matched_roles_per_row = []    
    all_valid_roles = set()       

    for idx, row in df_window.iterrows():
        faces = row.get('roles', [])
        faces_scores = row.get('role_scores', [])

        for index,face in enumerate(faces):
            score = faces_scores[index]
            if score > 0.5:
                all_valid_roles.add(face)
    return list(all_valid_roles)

def calculate_distance_weights_global(unique_speakers, idx, df, label_col='vision_pred'):
    """Calculate weights based on the distance to the current frame"""

    distances = {}
    for speaker in unique_speakers:
        label_indices = df[df[label_col] == speaker].index
        min_distance = min(abs(idx - sub_idx) for sub_idx in label_indices)
        distances[speaker] = min_distance

    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    weights = {}
    if len(sorted_distances) > 1:
        max_weight = 1
        min_weight = 0.6
        decrement = (max_weight - min_weight) / (len(sorted_distances) - 1)
        for k, _ in sorted_distances:
            weights[k] = max_weight
            max_weight -= decrement
    else:
        weights = {k: 1 for k, _ in distances.items()}
    return weights
def calculate_distance_weights(unique_speakers, idx, df_window, label_col='vision_pred'):
    distances = {}
    for speaker in unique_speakers:
        label_indices = df_window[df_window[label_col] == speaker].index
        min_distance = min(abs(idx - sub_idx) for sub_idx in label_indices)
        distances[speaker] = min_distance

    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    weights = {}
    if len(sorted_distances) > 1:
        max_weight = 1
        min_weight = 0.6 if vflag else 0.8   #v4
        decrement = (max_weight - min_weight) / (len(sorted_distances) - 1)
        for k, _ in sorted_distances:
            weights[k] = max_weight
            max_weight -= decrement
    else:
        weights = {k: 1 for k, _ in distances.items()}
    return weights

def calculate_weighted_scores(sample_emb, embedding2, weights=None):

    scores = {}
    weighted_scores = {}

    if weights:
        for speaker in weights:
            emb = sample_emb[speaker]
            result = compute_similarity(emb, embedding2)
            scores[speaker] = result
            weighted_scores[speaker] = result * weights.get(speaker, 1)
    else:
        weights = defaultdict(lambda: 1)
        for speaker in sample_emb:
            emb = sample_emb[speaker]
            result = compute_similarity(emb, embedding2)
            scores[speaker] = result
            weighted_scores[speaker] = result * weights.get(speaker, 1)
    return scores, weighted_scores


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) 
        return self.parent[x]

    def merge(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x  

def build_label_mapping(sim_v2v: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    all_labels: Set[str] = set(sim_v2v.keys())
    uf = UnionFind(all_labels)

    for k1 in sim_v2v:
        if not k1.startswith('a'):
            continue
        sim_dict = sim_v2v[k1]
        if not sim_dict:
            continue
        k2, sim = max(sim_dict.items(), key=lambda x: x[1])

        if (sim > 0.61) or (
                    sim > 0.44 and k1.startswith('a') and k2.startswith('a') and
                    abs(int(k1.split('_')[1]) - int(k2.split('_')[1])) < 6
                ): # v1
            uf.merge(k1, k2)

    return {label: uf.find(label) for label in all_labels}



def update_embeddings_with_new_labels(
    df: pd.DataFrame,
    label_col: str,
    files_col: str,
    sample_singel_emb: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:

    unique_labels = set(df[label_col].dropna())

    def condition_factory(label):
        return lambda row: row[label_col] == label

    for label in unique_labels:
        cond_func = condition_factory(label)
        mean_emb, count = compute_mean_embedding(
            df=df,
            condition_func=cond_func
        )
        if count > 0 and label.startswith('a'):
            sample_singel_emb[label] = mean_emb

    return sample_singel_emb

def update_embeddings_with_new_labels_clustered(
    df: pd.DataFrame,
    label_col: str,
    files_col: str,
    cluster_n: int = 3
) -> Dict[str, np.ndarray]:

    unique_labels = set(df[label_col].dropna())
    sample_multi_emb = {}
    for label in unique_labels:
        rows = df[df[label_col]==label]
        emb_list = []
        for _, row in rows.iterrows():
            emb_file = row[files_col].replace('wav', 'npy')
            if os.path.exists(emb_file):
                emb = np.load(emb_file)
                emb_list.append(emb)
        if emb_list and label.startswith('a'):
            stacked_emb = np.stack(emb_list)
            if len(emb_list) >= cluster_n:
                kmeans = KMeans(n_clusters=cluster_n, random_state=0)
                kmeans.fit(stacked_emb)
                centers = kmeans.cluster_centers_
            else:

                mean_emb = np.mean(stacked_emb, axis=0)
                centers = np.stack([mean_emb]*cluster_n)
            sample_multi_emb[label] = centers
    return sample_multi_emb

def build_similarity_matrix(
    sample_singel_emb: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:

    sim_v2v = defaultdict(dict)
    labels = list(sample_singel_emb.keys())

    for i, k1 in enumerate(labels):
        emb1 = sample_singel_emb[k1].reshape(1, -1)
        for j, k2 in enumerate(labels):
            if k1 == k2:
                continue
            emb2 = sample_singel_emb[k2].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            sim_v2v[k1][k2] = sim

    return dict(sim_v2v)
def build_similarity_matrix_clustered(sample_multi_emb: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:

    sim_v2v = defaultdict(dict)
    labels = list(sample_multi_emb.keys())
    for i, k1 in enumerate(labels):
        embs1 = sample_multi_emb[k1]  # shape (3, emb_dim)
        for j, k2 in enumerate(labels):
            if k1 == k2:
                continue
            embs2 = sample_multi_emb[k2]
            # 计算两人全部中心的两两最大余弦相似度
            sims = []
            for e1 in embs1:
                for e2 in embs2:
                    # 这里可用sklearn或者自己写cosine
                    sim = np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-8)
                    sims.append(sim)
            sim_v2v[k1][k2] = max(sims)
    return dict(sim_v2v)
