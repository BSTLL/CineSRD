from utils import *
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
cluster_flag=False 
cn = 1 
def get_new_pkl(pklpaths):
    with open(pklpaths, 'rb') as f:
        stat_obj = pickle.load(f)
        vision_embeddings=stat_obj['embeddings']
        vision_times=stat_obj['times']
        active_scores=stat_obj['score']
        embedding_hq = stat_obj['embedding_hq']
        qscore_hq = stat_obj['score_hq']
        embedding_rate = stat_obj['embedding_rate']
        qscore_src = stat_obj['best_quality_score']
    return vision_embeddings,vision_times,active_scores

def get_all_pkl(pklpaths):
    with open(pklpaths, 'rb') as f:
        stat_obj = pickle.load(f)
        vision_embeddings=stat_obj['embeddings']
        vision_times=stat_obj['times']
        active_scores=stat_obj['score']
    
    return vision_embeddings,vision_times,active_scores


def get_raw_emb(csvpaths,pklpaths,train_or_pred=True):

    audio_embeddings = []
    vision_embeddings = []
    vision_times = []
    labels = []
    active_scores = []
    ch_texts,slices,texts,indexs = [],[],[],[]
    start_time,end_time = [],[]
    for csvpath,pklpath in zip(csvpaths,pklpaths):
        df = pd.read_csv(csvpath)
        indexs.append(len(df))
        for i in range(len(df)):
            wavfile = df['files'].iloc[i]
            emb = load_embedding(wavfile)
            emb_file = wavfile.replace('.wav','.npy')   
            audio_embeddings.append(emb)
            if train_or_pred:
                labels.append(df['pred_speaker'].iloc[i])
            else:
                 labels.append('')

            texts.append(df['text'].iloc[i])
            ch_texts.append(df['text'].iloc[i])

            slices.append(df['files'].iloc[i])
            start_time.append(df['start_time'].iloc[i])
            end_time.append(df['end_time'].iloc[i])
    
        visual_embs_file = pklpath
        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            vision_embeddings.append(stat_obj['embeddings'])
            vision_times.append(stat_obj['times'])
            active_scores.append(stat_obj['score'])
            
    vision_embeddings = np.concatenate([array for array in vision_embeddings], axis=0)
    audio_embeddings = np.array(audio_embeddings)

    
    my_dict = { 'text':texts, 
               'ch_texts':ch_texts,
                'start_time':start_time,
               'end_time':end_time,
                'pred_speaker':labels,
                'files':slices
              }
    
    # 创建一个DataFrame
    df = pd.DataFrame(my_dict)

    return vision_embeddings,audio_embeddings,vision_times,indexs,active_scores,df
    
def get_raw_emb_new(csvpaths,pklpaths,train_or_pred=True):

    emb_col = 'embeddings' # embeddings、embedding_hq、embedding_rate
    audio_embeddings = []
    vision_embeddings = []
    vision_times = []
    labels = []
    active_scores = []
    ch_texts,slices,texts,indexs = [],[],[],[]
    start_time,end_time = [],[]
    for csvpath,pklpath in zip(csvpaths,pklpaths):
        df = pd.read_csv(csvpath)
        indexs.append(len(df))
    
        for i in range(len(df)):
            wavfile = df['files'].iloc[i]

            emb = load_embedding(wavfile)
            emb_file = wavfile.replace('wav','npy')   
            audio_embeddings.append(emb)
            if train_or_pred:
                labels.append(df['pred_speaker'].iloc[i])
            else:
                 labels.append('')

            texts.append(df['text'].iloc[i])
            ch_texts.append(df['text'].iloc[i])

            slices.append(df['files'].iloc[i])
            start_time.append(df['start_time'].iloc[i])
            end_time.append(df['end_time'].iloc[i])
    
        visual_embs_file = pklpath
    
        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            embeddings = stat_obj[emb_col]
            times = stat_obj['times']
            scores = stat_obj['score']

            if 'best_quality_score' in stat_obj:
                bqs = np.array(stat_obj['best_quality_score'])
                mask = bqs >= 0.2

                embeddings = np.array(stat_obj[emb_col])[mask]
                times = np.array(stat_obj['times'])[mask]
                scores = np.array(stat_obj['score'])[mask]
        
            good_idxs = []
            embeddings_fixed = []
        
            for i, emb in enumerate(embeddings):
                arr = np.array(emb)

                if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == 512:
                    arr = arr.squeeze(0)

                if arr.shape == (512,):
                    good_idxs.append(i)
                    embeddings_fixed.append(arr)
                else:
                    print(f"Skip idx={i}, embedding shape: {arr.shape}")
            times_fixed = [times[i] for i in good_idxs]
            scores_fixed = [scores[i] for i in good_idxs]

            embeddings_fixed = np.array(embeddings_fixed)

            vision_embeddings.append(embeddings_fixed)
            vision_times.append(times_fixed)
            active_scores.append(scores_fixed)

            
    vision_embeddings = np.concatenate([array for array in vision_embeddings], axis=0)
    audio_embeddings = np.array(audio_embeddings)  
    my_dict = { 'text':texts, 
               'ch_texts':ch_texts,
                'start_time':start_time,
               'end_time':end_time,
                'pred_speaker':labels,
                'files':slices
              }

    df = pd.DataFrame(my_dict)
    return vision_embeddings,audio_embeddings,vision_times,indexs,active_scores,df
    
import numpy as np
import pandas as pd
import pickle

def get_raw_emb_new2(csvpaths, pklpaths, train_or_pred=True):
    """
    Input:
    - csvpaths: a list of csv files, each containing the columns files, text, start_time, and end_time
    - pklpaths: results from active speaker recognition
    - train_or_pred: whether it's prediction or training
"""

    audio_embeddings = []
    vision_embeddings = []
    vision_times = []
    labels = []
    active_scores = []
    ch_texts, slices, texts, indexs = [], [], [], []
    start_time, end_time = [], []
    
    for csvpath, pklpath in zip(csvpaths, pklpaths):
        df = pd.read_csv(csvpath)
        indexs.append(len(df))
    
        for i in range(len(df)):
            wavfile = df['files'].iloc[i]
            emb = load_embedding(wavfile)
            audio_embeddings.append(emb)
            labels.append(df['说话人'].iloc[i] if train_or_pred else '')
            texts.append(df['text'].iloc[i])
            ch_texts.append(df['text'].iloc[i])
            slices.append(df['files'].iloc[i])
            start_time.append(df['start_time'].iloc[i])
            end_time.append(df['end_time'].iloc[i])
    
        visual_embs_file = pklpath

        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            embeddings_rate = stat_obj['embedding_rate'] # embedding_rate、embeddings
            embeddings_hq = stat_obj.get('embedding_hq', [])

            times = stat_obj['times']
            scores = stat_obj['score']
            bqs = np.array(stat_obj['best_quality_score']) if 'best_quality_score' in stat_obj else None

            embeddings_fixed = []
            good_idxs = []  # 保证数据同步
            for i in range(len(embeddings_rate)):
                # 默认用embedding_rate
                arr = np.array(embeddings_rate[i])
                # shape修正
                if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == 512:
                    arr = arr.squeeze(0)

                if arr.shape == (512,) and bqs[i] > 0.2:
                    embeddings_fixed.append(arr)
                    good_idxs.append(i)
                else:
                    print(f"Skip idx={i}，bqs={bqs[i]}, embedding shape: {arr.shape if arr is not None else 'None'}")

            # 只保留同步筛选后的times和scores
            times_fixed = [times[i] for i in good_idxs]
            scores_fixed = [scores[i] for i in good_idxs]

            # 拼成np.array，便于后续拼接
            embeddings_fixed = np.array(embeddings_fixed)
            vision_embeddings.append(embeddings_fixed)
            vision_times.append(times_fixed)
            active_scores.append(scores_fixed)
            
    # 合并所有pkl的内容
    vision_embeddings = np.concatenate([array for array in vision_embeddings if len(array)>0], axis=0)
    audio_embeddings = np.array(audio_embeddings)  
    
    my_dict = { 'text': texts, 
                'ch_texts': ch_texts,
                'start_time': start_time,
                'end_time': end_time,
                '说话人': labels,
                'files': slices
              }
    df = pd.DataFrame(my_dict)

    return vision_embeddings, audio_embeddings, vision_times, indexs, active_scores, df



def get_role_name(avatar_path,centers):
    # 将人脸和头像对比，得到实际的角色名称
    score_dict = {}
    spk_dict = {}
    role_dict = {}

    ## 头像初始化
    for role in os.listdir(avatar_path):
        if role.endswith('npy'):
            emb_file = os.path.join(avatar_path,role)
            emb = np.load(emb_file)
            role_name = role.split('_')[0]
            role_dict[role_name] = emb
        
    ## 计算相似度
    for i in range(len(centers)):
        score_dict[i] = {}
        emb1 = centers[i].reshape(1,-1)
        for role,emb2 in role_dict.items():
            score = compute_similarity(emb1,emb2)
            score_dict[i][role]=score

    ## 重置角色名
    for spk, roles in score_dict.items():      
        max_role = max(roles.items(), key=lambda x: x[1])
        if max_role[1] > 0.5:  # 如果需要考虑分数大于0.5
            spk_dict[spk] = max_role[0]
    return spk_dict

def get_vision_pred_all(times, indexs, scores, vision_embedding_all):
    """
    times:      有效帧的全局索引列表（如 [5,20,41,...]）
    indexs:     每个视频段的总帧数（如 [100, 120,...]）
    scores:     与times的每一帧一一对应
    vision_embedding_all: 与times一一对应的embedding
    返回: scores_all, vision_embeddings（均为total_length长的列表，"无"帧为[]，有的帧为[score]或[embedding]）
    """
    total_length = sum(indexs)
    scores_all = [[] for _ in range(total_length)]
    vision_embeddings = [[] for _ in range(total_length)]

    for j in range(len(times)):
        global_idx = times[j]

        # 这里append是对当前帧
        scores_all[global_idx].append(scores[j])
        vision_embeddings[global_idx].append(vision_embedding_all[j])

    return scores_all, vision_embeddings



def get_vision_pred(vision_times, indexs, vision_pred, spk_dict, active_scores,vision_embeddings):
    """
    处理视觉模型输出的说话人预测结果，将标签映射为统一格式，
    并过滤掉出现次数少于2次的标签（即孤立项）。

    参数:
        vision_times (List[List[int]]): 每个视频段中有效帧的相对索引。
        indexs (List[int]): 每个视频段的总帧数。
        vision_pred (List): 原始说话人预测标签列表。
        spk_dict (Dict): 标签映射字典，如 {0: 'speaker_A'}
        active_scores (List[List[float]]): 每帧的活跃度得分。

    返回:
        Tuple[List, List]: 处理后的说话人标签列表和对应活跃度得分列表。
    """
    
    # 初始化最终结果容器
    total_length = sum(indexs)
    vision_pred_all = [None] * total_length
    active_scores_all = [0.0] * total_length
    active_vision_embeddings = [None] * total_length
    pred_index = 0  # 用于遍历 vision_pred 的指针

    # 遍历每个视频段
    offset = 0
    for i in range(len(vision_times)):
        times = vision_times[i]
        scores = active_scores[i]
        vision_embedding_all = vision_embeddings[i]
        
        # 将当前段的预测结果映射到全局位置
        for j in range(len(times)):
            global_idx = times[j] + offset
            old_label = vision_pred[pred_index]
            new_label = spk_dict.get(old_label, f'speaker_{old_label}')
            if global_idx >= len(vision_pred_all):
                print(f'global_idx over:{global_idx}')
                continue
                # print(f"ERROR: global_idx {global_idx} out of range (max={len(vision_pred_all)-1})")
                # print(f"  offset={offset}, times[{j}]={times[j]}, pred_index={pred_index}, len(times)={len(times)}")
                # print(f"  old_label={old_label}, new_label={new_label}")
                # print(f"  indexs[i]={indexs[i]}, i={i}")
                # print(f"  len(vision_pred)={len(vision_pred)}, len(scores)={len(scores)}, len(vision_embedding_all)={len(vision_embedding_all)}")
                # continue  # 跳过这次写入，避免程序崩溃
            vision_pred_all[global_idx] = new_label
            active_scores_all[global_idx] = scores[j]
            
            active_vision_embeddings[global_idx] = vision_embedding_all[j]
            
            pred_index += 1

        offset += indexs[i]  # 更新偏移量

    # 统计每个说话人标签的出现次数
    label_counts = defaultdict(int)
    for label in vision_pred_all:
        if label is not None:
            label_counts[label] += 1

    # 过滤掉只出现一次的说话人标签（孤立项）
    filtered_pred = [
        label if label_counts.get(label, 0) > 1 else None
        for label in vision_pred_all
    ]

    return filtered_pred, active_scores_all,active_vision_embeddings


def face_filter(df, audio_pred, vision_pred_all, active_scores_all):
    """
    根据音频预测和视觉预测，使用 embedding 和计数信息对人脸进行过滤。

    参数:
        df (pd.DataFrame): 原始数据 DataFrame。
        audio_pred (List): 音频预测结果。
        vision_pred_all (List): 视觉预测结果。
        active_scores_all (List): 每帧的活跃度分数。
        compute_embedding (Callable): 用于计算音频 embedding 的函数。

    返回:
        Tuple[List, Dict]: 过滤后的人脸预测结果和修正映射表。
    """

    # Step 1: 统计每个视觉标签对应的音频标签出现次数及得分总和
    counts = defaultdict(lambda: defaultdict(int))
    score_sums = defaultdict(lambda: defaultdict(float))

    for i, (l1, l2) in enumerate(zip(audio_pred, vision_pred_all)):
        if l2 is not None:
            counts[l2][l1] += 1
            score_sums[l2][l1] += active_scores_all[i]

    # Step 2: 构建校正规则 corrections
    corrections = {}
    for l2 in counts:
        corrections[l2] = get_most_common_or_by_score(counts[l2], score_sums[l2])

    # Step 3: 计算每个视觉标签对应的平均 embedding
    def make_condition_func(corrections):
        def condition_factory(label):
            a_val = corrections[label]
            return lambda row, v=label, a=a_val: row['vision_pred'] == v and row['audio_pred'] == a
        return condition_factory

    condition_func = make_condition_func(corrections)

    affinity_vision,_, _ = compute_class_embeddings_and_similarity(
        df=df,
        label_col='vision_pred',
        condition_func=condition_func
    )

    # Step 4: 统计每一集中的 audio-vision 分布
    index_flag = {}
    audio_pred_sub = df['audio_pred']
    vision_pred_sub = df['vision_pred']

    # 初始化统计数据
    count_dict = defaultdict(lambda: defaultdict(int))
    audio_vision_set = defaultdict(set)

    for a, v in zip(audio_pred_sub, vision_pred_sub):
        if pd.notna(v):
            count_dict[v][a] += 1
            audio_vision_set[a].add(v)

    # 标记 flag：该人脸在当前集中是否保留
    for v in count_dict:
        index_flag[v] = {}
        for a in count_dict[v]:
            vision_count = len(audio_vision_set.get(a, set()))
            #如果音频数量大于10/音频只对应一个人脸，则进行保留
            # if (vision_count == 1 and count_dict[v][a] > 1) or count_dict[v][a] >= 10:
            if count_dict[v][a] >= 10:

                index_flag[v][a] = 1
            else:
                index_flag[v][a] = 0

    # Step 5: 对视觉预测进行最终过滤
    vision_pred_all_new = []
    scores = []
    for _, row in df.iterrows():
        vpred = row['vision_pred']
        a_pred = row['audio_pred']
        wavfile = row['files']
        zh_text = row['text']
        active_score = row['active_scores']
        if pd.isnull(vpred):
            vision_pred_all_new.append(None)
            continue

        emb = load_embedding(wavfile)
        centers = affinity_vision[vpred].reshape(1, -1)
        score_tmp = cosine_similarity(centers, emb.reshape(1, -1))[0][0]

        # 判断是否保留该视觉预测
        try:
            flag = index_flag[vpred].get(a_pred, 0)
        except KeyError:
            flag = 0

        if score_tmp > 0.5 or a_pred == corrections[vpred] or flag == 1 or (len(zh_text) <= 2 and score_tmp > 0.35):
        # if score_tmp > 0.5 or a_pred == corrections[vpred] or flag == 1 or (len(zh_text) <= 2 and score_tmp > 0.35) or active_score>2:

            vision_pred_all_new.append(vpred)
            scores.append(score_tmp)
        else:
            vision_pred_all_new.append(None)
            scores.append(score_tmp)


    return vision_pred_all_new, corrections



def merge_sim_face_by_audio(df,vision_pred_all,vision_pred_all_new):
  
    def condition_factory(v_val):
        return lambda row: row['vision_pred_new'] == v_val
        
    affinity_vision,cnt_dict,sim_v2v = compute_class_embeddings_and_similarity(
        df=df,
        label_col='vision_pred_new',
        condition_func=condition_factory
    )

    vision_pred_all_new = np.array(vision_pred_all_new)
    vision_pred_all = np.array(vision_pred_all)

    for k1,v1 in sim_v2v.items():
        cnt1 = cnt_dict[k1]
        sim_sort = list(sim_v2v[k1].items())
        sim_sort.sort(key=lambda x:x[1])
        k2 = sim_sort[-1][0]
        sim = sim_sort[-1][1]
        cnt2 = cnt_dict[k2]
        if ((cnt1<3 and cnt2>10) or (cnt1>10 and cnt2<3)) and  sim>0.6:
            if cnt1>cnt2:
                vision_pred_all_new[vision_pred_all_new==k2]=k1
                vision_pred_all[vision_pred_all==k2]=k1
            else:
                vision_pred_all_new[vision_pred_all_new==k1]=k2
                vision_pred_all[vision_pred_all==k1]=k2
   
    return vision_pred_all,vision_pred_all_new

def get_face_audio(df, corrections, cluster_n=cn):
    sample_single = {}
    sample_single_old = []
    
    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']
        vpred_old = row['vision_pred']
        if not pd.isnull(vpred_old):
            sample_single_old.append(vpred_old)
        if not pd.isnull(vpred):
            emb_file = wavfile.replace('wav', 'npy')
            if os.path.exists(emb_file):
                tmp_emb = np.load(emb_file)
            else:
                print(f'{emb_file} not exist')
            if not sample_single.__contains__(vpred):
                sample_single[vpred] = []
                sample_single[vpred].append(tmp_emb)
            else:
                sample_single[vpred].append(tmp_emb)
                
    counts = Counter(sample_single_old)
    sample_single_old = [item for item in sample_single_old if counts[item] >= 2]
 
    sample_singel_emb = {}
    sample_multi_emb = {}     # 聚类音库
    for speaker in sample_single:
        arrays = sample_single[speaker]
        stacked_arrays = np.stack(arrays)
        embedding = np.mean(stacked_arrays, axis=0)
        if not sample_singel_emb.__contains__(speaker):
            sample_singel_emb[speaker] = embedding
        if len(arrays) >= cluster_n:   # 足够样本才聚类
            kmeans = KMeans(n_clusters=cluster_n, random_state=0)
            kmeans.fit(stacked_arrays)
            centers = kmeans.cluster_centers_
            # centers = np.vstack([embedding for _ in range(cluster_n)])
        else:
            # centers = np.vstack(arrays)
            centers = np.vstack([embedding for _ in range(cluster_n)])
        # print(f"speaker: {speaker}, centers.shape: {centers.shape}")
        sample_multi_emb[speaker] = centers  # shape: (3, emb_dim)

    
    # 对于出现两次以上但不在本集的角色，从剧集全局注册音补全
    for speaker in sample_single_old:
        if not sample_singel_emb.__contains__(speaker):
            sample_singel_emb[speaker] = sample_all_emb[speaker]
            print('add sample_all_emb')
        if sample_multi_emb.get(speaker) is None and sample_all_emb:
            sample_multi_emb[speaker] = np.vstack([sample_all_emb[speaker] for _ in range(cluster_n)])
            print('add sample_all_emb (cluster centers by mean)')
    if cluster_flag:
        return sample_multi_emb
    else:
        return sample_singel_emb



def get_face_emb_dict(df):
    # 收集同一说话人的全部人脸embedding
    sample_vis_emb = {}
    # sample_single_old = []
    
    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']

        if not pd.isnull(vpred):
            tmp_emb = row['vis_emb_active']
            # 由视觉预测聚合到对应说话人
            if not sample_vis_emb.__contains__(vpred):
                sample_vis_emb[vpred] = []
                sample_vis_emb[vpred].append(tmp_emb)
            else:
                sample_vis_emb[vpred].append(tmp_emb)
 
    sample_singel_vis_emb = {}
    for speaker in sample_vis_emb:
        arrays = sample_vis_emb[speaker]
        stacked_arrays = np.stack(arrays)
        embedding = np.mean(stacked_arrays, axis=0)
        if not sample_singel_vis_emb.__contains__(speaker):
            sample_singel_vis_emb[speaker] = embedding  

    return sample_singel_vis_emb





def pred_spk_step1(df, sample_singel_emb, index_off1=60, index_off2=300):
    pred_spk = []      # 存储最终预测标签
    scores = []        # 存储每帧的得分字典
    max_scores = []    # 存储每帧最大得分
    sim_list = []      # 存储相邻帧embedding相似度

    # 统计是否开启新场景，用于分段聚类/分析
    time_diff = (df['start_time'] - df['end_time'].shift(1))
    new_conversation = (time_diff > 5000) | time_diff.isnull()
    df['scene'] = new_conversation.cumsum()

    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']

        embedding2 = load_embedding(wavfile)
        next_file = df['files'].iloc[idx+1] if idx < len(df)-1 else wavfile
        embedding3 = load_embedding(next_file)

        sim_list.append(compute_similarity(embedding3, embedding2))

        if pd.isnull(vpred):
            local_df = get_window_df(df, idx, index_off1)
            local_speakers = get_unique_speakers(local_df)
            local_weights = calculate_distance_weights(local_speakers, idx, local_df)
            if cluster_flag:
                local_scores, local_weighted = calculate_weighted_scores_by_clusteremb(
                sample_singel_emb, embedding2, local_weights
            )
            else:
                local_scores, local_weighted = calculate_weighted_scores(
                    sample_singel_emb, embedding2, local_weights
                )

            # 全局窗口
            global_df = get_window_df(df, idx, index_off2)
            global_speakers = get_unique_speakers(global_df)
            global_weights = calculate_distance_weights(global_speakers, idx, global_df)
            if cluster_flag:
                global_scores, global_weighted = calculate_weighted_scores_by_clusteremb(
                sample_singel_emb, embedding2, global_weights
            )
            else:
                global_scores, global_weighted = calculate_weighted_scores(
                    sample_singel_emb, embedding2, global_weights
                )
            

            # 根据得分优先级和窗口结果决策最终标签
            if local_weighted:
                max_local_key = max(local_weighted, key=local_weighted.get)
                max_local_score = local_scores[max_local_key]

                max_global_key = max(global_weighted, key=global_weighted.get)
                max_global_score = global_scores[max_global_key]

                if max_local_score > 0.5:
                    pred_spk.append(max_local_key)
                    scores.append(local_scores)
                    max_scores.append(max_local_score)
                elif max_global_score < 0.6:
                    pred_spk.append(max_local_key)
                    scores.append(local_scores)
                    max_scores.append(max_local_score)
                else:
                    pred_spk.append(max_global_key)
                    scores.append(global_scores)
                    max_scores.append(max_global_score)
            else:
                # 全局静态库兜底
                fallback_scores, _ = calculate_weighted_scores(sample_all_emb, embedding2)
                max_key = max(fallback_scores, key=fallback_scores.get)
                pred_spk.append(max_key)
                scores.append(fallback_scores)
                max_scores.append(fallback_scores[max_key])

        else:
            # 有视觉信息直接用视觉标签
            pred_spk.append(vpred)
            scores.append("")
            max_scores.append(0)

    return pred_spk, scores, max_scores, sim_list

def pred_spk_window_byface(df, sample_singel_emb, index_off1=60, index_off2=300):
    pred_spk = []      # 存储最终预测标签
    scores = []        # 存储每帧的得分字典
    max_scores = []    # 存储每帧最大得分
    faces = []   # 存储每帧出现过的人脸

    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']
        embedding2 = load_embedding(wavfile)

        if pd.isnull(vpred):
            # 无视觉信息，进行滑窗聚类推断

            # 局部窗口
            local_df = get_window_df(df, idx, index_off1)
            local_speakers = get_unique_speakers_byface(local_df)
            local_weights = calculate_distance_weights_global(local_speakers, idx, df)
            if cluster_flag:
                local_scores, local_weighted = calculate_weighted_scores_by_clusteremb(
                sample_singel_emb, embedding2,local_weights
            )
            else:
                local_scores, local_weighted = calculate_weighted_scores(
                    sample_singel_emb, embedding2,local_weights
                )

            # 全局窗口
            global_df = get_window_df(df, idx, index_off2)
            global_speakers = get_unique_speakers_byface(global_df)
            global_weights = calculate_distance_weights_global(global_speakers, idx, df)
            if cluster_flag:
                global_scores, global_weighted = calculate_weighted_scores_by_clusteremb(
                    sample_singel_emb, embedding2,global_weights
                )
            else:
                global_scores, global_weighted = calculate_weighted_scores(
                    sample_singel_emb, embedding2,global_weights
                )

            # 根据得分优先级和窗口结果决策最终标签
            if local_weighted:
                max_local_key = max(local_weighted, key=local_weighted.get)
                max_local_score = local_scores[max_local_key]

                max_global_key = max(global_weighted, key=global_weighted.get)
                max_global_score = global_scores[max_global_key]

                if max_local_score > 0.5:
                    pred_spk.append(max_local_key)
                    scores.append(local_scores)
                    max_scores.append(max_local_score)
                elif max_global_score < 0.6:
                    pred_spk.append(max_local_key)
                    scores.append(local_scores)
                    max_scores.append(max_local_score)
                else:
                    pred_spk.append(max_global_key)
                    scores.append(global_scores)
                    max_scores.append(max_global_score)
            else:
                # 全局静态库兜底
                fallback_scores, _ = calculate_weighted_scores(sample_all_emb, embedding2)
                max_key = max(fallback_scores, key=fallback_scores.get)
                pred_spk.append(max_key)
                scores.append(fallback_scores)
                max_scores.append(fallback_scores[max_key])

        else:
            # 有视觉信息直接用视觉标签
            pred_spk.append(vpred)
            scores.append("")
            max_scores.append(0)

    return pred_spk, scores, max_scores

def pred_spk_step2(
    df: pd.DataFrame,
    sample_singel_emb: dict,
    window_offset: int = 15,
    fallback_offset: int = 25
):
    """
    根据音频 embedding 和局部上下文信息，为无视觉预测的帧分配说话人标签。

    参数:
        df (pd.DataFrame): 包含音频路径、时间戳、预测等信息的数据框。
        sample_singel_emb (Dict): 每个 episode 中各 speaker 的代表 embedding。
        window_offset (int): 默认窗口大小。
        fallback_offset (int): 回退窗口大小。

    返回:
        Tuple[List, List, List, List]: 预测的说话人、相似度字典、最大相似度值、相邻帧相似度列表
    """

    pred_spk = []
    scores = []
    max_scores = []

    # 添加辅助列：vision_pred_high（新的一列 vision_pred_high，每行优先填视觉说话人标签；没有视觉时，如果声纹识别分数较高（>0.4），则用声纹推理的标签；否则为空。）
    df['vision_pred_high'] = np.where(
        df['vision_pred'].notna(),
        df['vision_pred'],
        np.where(df['max_scores'] > 0.4, df['step1_spk'], np.nan)
    )

    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']

        # 加载当前帧的 embedding
        embedding2 = load_embedding(wavfile)

        if pd.isnull(vpred):
            # 尝试获取局部窗口内说话人
            local_df = get_window_df(df, idx, window_offset)
            unique_speakers = get_unique_speakers(local_df, 'vision_pred_high')
        
            # 如果说话人太少，扩展窗口
            if len(unique_speakers) < 2:
                local_df = get_window_df(df, idx, fallback_offset)
                unique_speakers = get_unique_speakers(local_df, 'vision_pred_high')

            # 计算距离权重
            weights = calculate_distance_weights(unique_speakers, idx, local_df, 'vision_pred_high')

            # 使用当前 episode 的样本 embedding 进行匹配
            if cluster_flag:
                score_dict, weighted_scores = calculate_weighted_scores_by_clusteremb(
                sample_singel_emb, embedding2, weights
            )
            else:
                score_dict, weighted_scores = calculate_weighted_scores(
                    sample_singel_emb, embedding2, weights
                )


            max_key = max(weighted_scores, key=weighted_scores.get)
            pred_spk.append(max_key)
            scores.append(score_dict)
            max_scores.append(score_dict[max_key])

        else:
            pred_spk.append(vpred)
            scores.append('')
            max_scores.append(0)


    return pred_spk, scores, max_scores



def update_label(df,thold=0.3):
    # 改为通过音频相似度来分组
    group_id = 0
    groups = []
    
    for i in range(len(df)):
        groups.append(group_id)

        if df.loc[i, 'sim'] <= 0.5:
            group_id += 1
    
    # 添加 group 列
    df['group'] = groups
    # 分组结束

    # 筛选出 vision_pred 为空的行
    df_filtered = df.groupby('group').filter(lambda x: (x['vision_pred_new'].isnull().all()) and (len(x) >= 1))
    # 计算每个组的 max_scores 均值
    group_means = df_filtered.groupby('group')['max_scores'].mean()
    
    # 进行标签的统一，对于均值小于0.3的分组，认为无法检测人脸。
    def rename_group_labels(row): # v3
        high_thod = 100
        if vflag:
            # thod = 0.3
            high_thod = 1
        if row['group'] in group_means.index and group_means[row['group']] < thold and len(row['text'].replace(' ',''))>2 and row['active_scores']<high_thod: # v8
            return f"audio_{row['group']}"
        else:
            return row['step2_spk']
    # 更新原 DataFrame 中相应行的 updated_label 列，再次根据新的条件
    df['updated_label1'] = df.apply(rename_group_labels, axis=1)
    

    def get_most_common_label(group):
        labels = group['updated_label1']
        scores = group['max_scores']
        # 找到出现频率最高的标签
        mode_labels = labels.mode()
    
        if len(mode_labels) > 1:
            # 如果有多个标签的频率相同，找到评分最高的标签
            max_score = -1
            max_label = None
            for label in mode_labels:
                label_scores = scores[labels == label]
                if label_scores.max() > max_score:
                    max_score = label_scores.max()
                    max_label = label
            return max_label
        else:
            return mode_labels.iloc[0]
    
    most_common_labels = df.groupby('group').apply(get_most_common_label)

    #进行标签的统一，对于声纹标签连续相同且声纹分数小于0.5的，转换为次数最多的标签
    def update_label_with_most_common(row):
        if row['group'] in most_common_labels.index and float(row['max_scores'])!=0: # v6
        # if row['group'] in most_common_labels.index and float(row['max_scores']) < 0.5 and float(row['max_scores'])!=0:
            return most_common_labels[row['group']]
        else:
            return row['updated_label1']

    df['updated_label2'] = df.apply(update_label_with_most_common, axis=1)
    return df

def merge_label_dif(df):
    left = 0.2
    t = 0.2
    # df = pd.read_csv(path)
    score_dict_col = 'scores'
    label_col = 'temp_final'
    sim_col = 'muti_sim' if 'muti_sim' in df.columns else 'sim'
    new_col = 'new_final'
    
    df[new_col] = df[label_col]
    cor_num = 0
    for i in range(len(df)):
        current_row = df.iloc[i]
        # 检查 max_scores_final 的条件
        if current_row['max_scores'] < t:
            if i > 0:  
                prev_row = df.iloc[i - 1]
                if prev_row[sim_col] <= left and current_row[label_col] == prev_row[label_col]:
                    # 检查 scores_final 是否为空
                    if pd.isna(current_row[score_dict_col]) or current_row[score_dict_col] == '':
                        df.at[i, new_col] = current_row[label_col]
                    else:
                        # 解析 scores_final 为字典
                        try:
                            # scores_final_dict = eval(current_row[score_dict_col])  # 将字符串解析为字典
                            data = current_row[score_dict_col]
                            try:
                                # 保证 data 是字符串，再解析为字典
                                if isinstance(data, str):
                                    scores_final_dict = ast.literal_eval(data)
                                elif isinstance(data, dict):
                                    scores_final_dict = data
                                else:
                                    # 其它格式尝试直接转换，或者raise
                                    raise ValueError("scores_final为非预期格式")
                            except Exception as e:
                                print(f"Error parsing scores_final at row {i}: {e}")
                                scores_final_dict = {}


                            
                            # 筛除上一行的 final_label，找到 value 最大的 key
                            filtered_scores = {key: value for key, value in scores_final_dict.items() if key != prev_row[label_col]}
                            max_key = max(filtered_scores, key=filtered_scores.get) if filtered_scores else current_row[label_col]
                            df.at[i, new_col] = max_key
                            cor_num += 1
                        except Exception as e:
                            print(f"Error parsing scores_final at row {i}: {e}")
                            df.at[i, new_col] = current_row[label_col]
    return df

def merge_label_by_sim(df):
    label_col = 'updated_label3' # updated_label3，final_label
    score_col = 'max_scores' # max_scores_final，max_scores
    sim_col = 'muti_sim' if 'muti_sim' in df.columns else 'sim'
    numflag = 0
    if 'final' in score_col:
        numflag = 1
    group_id = 0
    groups = []
    
    for i in range(len(df)):
        groups.append(group_id)

        if df.loc[i, sim_col] <= 0.6:
            group_id += 1
    
    # 添加 group 列
    df['new_group'] = groups
    def get_most_common_label(group):
        labels = group[label_col]
        scores = group[score_col]
        
        # 找到出现频率最高的标签
        mode_labels = labels.mode()
    
        if len(mode_labels) > 1:
            # 如果有多个标签的频率相同，找到评分最高的标签
            max_score = -1
            max_label = None
            for label in mode_labels:
                label_scores = scores[labels == label]
                if numflag == 1:
                    label_scores = label_scores[label_scores != 1]
                if label_scores.max() > max_score:
                    max_score = label_scores.max()
                    max_label = label
            return max_label
        else:
            return mode_labels.iloc[0]
    
    most_common_labels = df.groupby('new_group').apply(get_most_common_label)
    # most_common_labels = df.groupby('new_group').apply(get_most_common_label, include_group=False)

    #进行标签的统一，对于声纹标签连续相同且声纹分数小于0.5的，转换为次数最多的标签
    def update_label_with_most_common(row):
        # if row['new_group'] in most_common_labels.index and float(row[score_col])!=numflag:
        if row['group'] in most_common_labels.index and float(row[score_col]) < 0.4 and float(row[score_col])!=0:
            return most_common_labels[row['new_group']]
        else:
            return row[label_col]


    df['temp_final'] = df.apply(update_label_with_most_common, axis=1)
    df = merge_label_dif(df)
    return df


def get_audio_emb(
    df: pd.DataFrame,
    sample_singel_emb: dict
) -> Tuple[dict]:
    """
    根据 updated_label2 列中以 'a' 开头的标签，计算其平均 embedding，
    并更新到 sample_singel_emb 和 sample_all_emb 字典中。

    参数:
        df (pd.DataFrame): 数据框，包含 files、updated_label2 等列。
        sample_singel_emb (dict): 每个 episode 的 speaker embedding 字典。
        sample_all_emb (dict): 所有 speaker 的全局 embedding 字典。
        episode_idx (int): 当前 episode 编号，用于 sample_singel_emb 更新。

    返回:
        Tuple[dict, dict]: 更新后的 sample_singel_emb 和 sample_all_emb
    """

    def condition_factory(label):
        return lambda row: row['updated_label2'] == label

    unique_labels = set(df['updated_label2'].dropna())
    for label in unique_labels:
        if label.startswith('a'):
                # 构建当前 label 的筛选条件
            cond_func = condition_factory(label)
    
            # 计算平均 embedding
            mean_emb, count = compute_mean_embedding(
                df=df,
                condition_func=cond_func
            )
    
            
            sample_singel_emb[label] = mean_emb

    return sample_singel_emb
def get_audio_emb_cluster(
    df: pd.DataFrame,
    sample_multi_emb: dict,
    cluster_n: int = cn
) -> dict:
    """
    按照 updated_label2（以'a'开头），对每类聚合所有embedding，进行聚类，得到每类3个中心。
    返回 sample_multi_emb: {label: np.ndarray(3, emb_dim)}
    """
    unique_labels = set(df['updated_label2'].dropna())
    
    for label in unique_labels:
        if label.startswith('a'):
            # 选此label的所有样本行
            rows = df[df['updated_label2'] == label]
            emb_list = []
            for _, row in rows.iterrows():
                emb_file = row['files'].replace('wav', 'npy')
                if os.path.exists(emb_file):
                    emb = np.load(emb_file)
                    emb_list.append(emb)
            if emb_list:
                # 聚类
                stacked_emb = np.stack(emb_list)
                if len(emb_list) >= cluster_n:
                    
                    kmeans = KMeans(n_clusters=cluster_n, random_state=0)
                    kmeans.fit(stacked_emb)
                    centers = kmeans.cluster_centers_
                else:
                    # 样本数量不足，直接补齐（如用均值/重复）
                    mean_emb = np.mean(stacked_emb, axis=0)
                    centers = np.vstack([mean_emb for _ in range(cluster_n)])
                sample_multi_emb[label] = centers  # (3, emb_dim)
    return sample_multi_emb
    
def merge_sim_audio(
    df: pd.DataFrame,
    sample_singel_emb: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    合并音频类中相似度过高的说话人，更新标签并重新计算平均 embedding。

    参数:
        df (pd.DataFrame): 包含 updated_label2 和 files 等列的数据框。
        sample_singel_emb (Dict[str, np.ndarray]): 当前各说话人的代表 embedding。

    返回:
        Tuple[pd.DataFrame, Dict[str, np.ndarray]]: 更新后的 DataFrame 和 embedding 字典
    """

    # Step 1: 构建相似度矩阵
    if cluster_flag:
        sim_v2v = build_similarity_matrix_clustered(sample_singel_emb)
    else:
         sim_v2v = build_similarity_matrix(sample_singel_emb)
    label_mapping = build_label_mapping(sim_v2v)

    df['updated_label3'] = df['updated_label2'].replace(label_mapping)

    if cluster_flag:
        updated_sample_singel_emb = update_embeddings_with_new_labels_clustered(
            df=df,
            label_col='updated_label3',
            files_col='files',
            cluster_n = cn
        )
    else:
        updated_sample_singel_emb = update_embeddings_with_new_labels(
        df=df,
        label_col='updated_label3',
        files_col='files',
        sample_singel_emb=sample_singel_emb
    )

    return df, updated_sample_singel_emb



def get_final_spk(
    df: pd.DataFrame,
    sample_singel_emb: dict,
    window_offset: int = 10,
    fallback_offset: int = 20
):
    """
    基于多种条件生成最终说话人预测。

    参数:
        df (pd.DataFrame): 包含预测信息的数据框。
        sample_singel_emb (Dict): 当前 episode 的 speaker embedding 字典。
        window_offset (int): 默认邻域窗口大小。
        fallback_offset (int): 回退时的邻域窗口大小。

    返回:
        Tuple[List, List, List]: 预测的说话人、分数字典、最大分数列表
    """

    pred_spk = []     # 最终预测的说话人列表
    scores = []       # 每帧的得分字典
    max_scores = []   # 每帧的最大得分

    # 生成 vision_pred_high_new 列，根据不同条件选择优先标签
    condition1 = df['updated_label3'].str.startswith('a')  # 条件1：updated_label3 以 'a' 开头
    condition2 = df['vision_pred'].notna()                # 条件2：vision_pred 非空
    condition3 = df['max_scores'] > 0.4                   # 条件3：max_scores 大于 0.4

    # 用 np.select 实现多条件优先标签选择
    df['vision_pred_high_new'] = np.select(
        [condition1, condition2, condition3],
        [df['updated_label3'], df['vision_pred'], df['step2_spk']],
        default=np.nan
    )

    # 遍历每一帧，生成最终预测
    for idx, row in df.iterrows():
        wavfile = row['files']            # 当前音频文件
        vpred = row['vision_pred_new']    # 视觉预测结果
        old_max_score = row['max_scores'] # 原有最大得分

        embedding2 = load_embedding(wavfile)   # 加载音频的特征 embedding

        if pd.isnull(vpred) and old_max_score < 0.45:
            # 若无视觉预测且声纹得分低，滑窗在邻域内分析
            local_df = get_window_df(df, idx, window_offset)
            unique_speakers = get_unique_speakers(local_df, 'vision_pred_high_new')

            # 若候选说话人过少，扩大窗口
            if len(unique_speakers) < 3:
                local_df = get_window_df(df, idx, fallback_offset)
                unique_speakers = get_unique_speakers(local_df, 'vision_pred_high_new')

            # 计算邻域说话人权重
            weights = calculate_distance_weights(unique_speakers, idx, local_df, 'vision_pred_high_new')

            # 用当前 episode 的 embedding 与候选说话人加权匹配
            if cluster_flag:
                score_dict, weighted_scores = calculate_weighted_scores_by_clusteremb(
                sample_singel_emb, embedding2, weights
            )
            else:
                score_dict, weighted_scores = calculate_weighted_scores(
                    sample_singel_emb, embedding2, weights
                )

            if weighted_scores:
                # 若滑窗有有效分数，选得分最高的说话人
                max_key = max(weighted_scores, key=weighted_scores.get)
                pred_spk.append(max_key)
                scores.append(score_dict)
                max_scores.append(score_dict[max_key])
            else:
                # 若无，则回退用全局 embedding 匹配
                if cluster_flag:
                    score_dict, weighted_scores = calculate_weighted_scores_by_clusteremb(sample_singel_emb, embedding2)
                else:
                    score_dict, weighted_scores = calculate_weighted_scores(sample_singel_emb, embedding2)
                max_key = max(score_dict, key=score_dict.get)
                pred_spk.append(max_key)
                scores.append(score_dict)
                max_scores.append(score_dict[max_key])

        else:
            # 有视觉预测结果，直接采用
            if not pd.isnull(vpred):
                pred_spk.append(vpred)
                scores.append('')
                max_scores.append(1)   # 置信度最高
            else:
                # 没有视觉预测但声纹分数较高，采用 updated_label3
                pred_spk.append(row['updated_label3'])
                scores.append('')
                max_scores.append(old_max_score)

    return pred_spk, scores, max_scores



def update_label_final(df):
    """
    对声纹标签做进一步的后处理与统一，使标签更加平滑可靠。

    参数:
        df (pd.DataFrame): 包含中间标签和分数的数据框。
    """

    df['final_label'] = df['step3_spk']   # 初始化最后标签
    def get_most_common_label(group):
        labels = group['step3_spk']
        scores = group['max_scores_final']

        mode_labels = labels.mode()   # 众数标签
        if len(mode_labels) > 1:
            # 众数有多个，取分数最高的
            max_score = -1
            max_label = None
            for label in mode_labels:
                label_scores = scores[labels == label]
                if label_scores.max() > max_score:
                    max_score = label_scores.max()
                    max_label = label
            return max_label
        else:
            return mode_labels.iloc[0]

    most_common_labels = df.groupby('group').apply(get_most_common_label)

    def update_label_with_most_common(row):
        if row['group'] in most_common_labels.index and (
            (float(row['max_scores_final']) < 0.5 and float(row['max_scores_final'])!=0) or
            (float(row['max_scores']) < 0.5 and float(row['max_scores']) !=0 and float(row['max_scores_final'])==0)
        ):
            return most_common_labels[row['group']]
        else:
            return row['step3_spk']


    df['final_label'] = df.apply(update_label_with_most_common, axis=1)

    for i in range(len(df)-1):
        current_label = df.loc[i, 'final_label']
        current_score = df.loc[i, 'sim']
        next_max_scores =  df.loc[i+1, 'max_scores_final']
        now_max_scores = df.loc[i, 'max_scores_final']

        if i == 0:
            if current_score > 0.4  and now_max_scores < 0.5 and (next_max_scores > now_max_scores and  next_max_scores > 0.3):
                df.loc[i, 'final_label'] = df.loc[i+1, 'final_label']
        else:
            previous_score = df.loc[i-1, 'sim']
            previous_max_scores = df.loc[i-1, 'max_scores_final']

            if now_max_scores < 0.5:
                if previous_score > 0.4 and previous_score > current_score and (previous_max_scores > now_max_scores and previous_max_scores > 0.3):
                    df.loc[i, 'final_label'] = df.loc[i-1, 'final_label']
                if current_score > 0.4 and current_score > previous_score and (next_max_scores > now_max_scores and  next_max_scores > 0.3):
                    df.loc[i, 'final_label'] =  df.loc[i+1, 'final_label']
    return df