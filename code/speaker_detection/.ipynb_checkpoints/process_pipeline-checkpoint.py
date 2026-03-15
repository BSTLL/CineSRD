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


def get_raw_emb(csvpaths,pklpaths,train_or_pred=False):
    """
    Input:
    - csvpaths: list of csv files, each containing the columns files, text, start_time, end_time
    - pklpaths: results of active speaker recognition
    - trai
"""

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
                labels.append(df['说话人'].iloc[i])
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



def get_role_name(avatar_path,centers):
    score_dict = {}
    spk_dict = {}
    role_dict = {}

    for role in os.listdir(avatar_path):
        if role.endswith('npy'):
            emb_file = os.path.join(avatar_path,role)
            emb = np.load(emb_file)
            role_name = role.split('_')[0]
            role_dict[role_name] = emb

    for i in range(len(centers)):
        score_dict[i] = {}
        emb1 = centers[i].reshape(1,-1)
        for role,emb2 in role_dict.items():
            score = compute_similarity(emb1,emb2)
            score_dict[i][role]=score

    for spk, roles in score_dict.items():      
        max_role = max(roles.items(), key=lambda x: x[1])
        if max_role[1] > 0.5: 
            spk_dict[spk] = max_role[0]
    return spk_dict


def get_vision_pred(vision_times, indexs, vision_pred, spk_dict, active_scores):
   """
    Processes the speaker prediction results from the visual model, maps labels to a unified format,
    and filters out labels that appear less than 2 times (i.e., outliers).

    Parameters:
        vision_times (List[List[int]]): Relative indices of valid frames in each video segment.
        indexs (List[int]): Total frame count for each video segment.
        vision_pred (List): Original list of speaker prediction labels.
        spk_dict (Dict): Label mapping dictionary, e.g. {0: 'speaker_A'}
        active_scores (List[List[float]]): Activity score for each frame.

    Returns:
        Tuple[List, List]: List of processed speaker labels and corresponding activity score list.
"""

    
    # 初始化最终结果容器
    total_length = sum(indexs)
    vision_pred_all = [None] * total_length
    active_scores_all = [0.0] * total_length

    pred_index = 0  # 用于遍历 vision_pred 的指针

    # 遍历每个视频段
    offset = 0
    for i in range(len(vision_times)):
        times = vision_times[i]
        scores = active_scores[i]
        
        # 将当前段的预测结果映射到全局位置
        for j in range(len(times)):
            global_idx = times[j] + offset
            old_label = vision_pred[pred_index]
            new_label = spk_dict.get(old_label, f'speaker_{old_label}')
            
            vision_pred_all[global_idx] = new_label
            active_scores_all[global_idx] = scores[j]
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

    return filtered_pred, active_scores_all


def face_filter(df, audio_pred, vision_pred_all, active_scores_all):
    """
    Filters face predictions using embedding and count information, based on audio and visual predictions.

    Parameters:
        df (pd.DataFrame): Original data DataFrame.
        audio_pred (List): Audio prediction results.
        vision_pred_all (List): Visual prediction results.
        active_scores_all (List): Activity scores for each frame.

    Returns:
        Tuple[List, Dict]: Filtered face prediction results and correction mapping.
"""


    counts = defaultdict(lambda: defaultdict(int))
    score_sums = defaultdict(lambda: defaultdict(float))

    for i, (l1, l2) in enumerate(zip(audio_pred, vision_pred_all)):
        if l2 is not None:
            counts[l2][l1] += 1
            score_sums[l2][l1] += active_scores_all[i]

    corrections = {}
    for l2 in counts:
        corrections[l2] = get_most_common_or_by_score(counts[l2], score_sums[l2])

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


    index_flag = {}
    audio_pred_sub = df['audio_pred']
    vision_pred_sub = df['vision_pred']

    count_dict = defaultdict(lambda: defaultdict(int))
    audio_vision_set = defaultdict(set)

    for a, v in zip(audio_pred_sub, vision_pred_sub):
        if pd.notna(v):
            count_dict[v][a] += 1
            audio_vision_set[a].add(v)

    for v in count_dict:
        index_flag[v] = {}
        for a in count_dict[v]:
            vision_count = len(audio_vision_set.get(a, set()))
            if count_dict[v][a] >= 10:
                index_flag[v][a] = 1
            else:
                index_flag[v][a] = 0

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

        try:
            flag = index_flag[vpred].get(a_pred, 0)
        except KeyError:
            flag = 0

        if score_tmp > 0.5 or a_pred == corrections[vpred] or flag == 1 or (len(zh_text) <= 2 and score_tmp > 0.35):

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
        if ((cnt1<3 and cnt2>5) or (cnt1>5 and cnt2<3)) and  sim>0.6:
            if cnt1>cnt2:
                vision_pred_all_new[vision_pred_all_new==k2]=k1
                vision_pred_all[vision_pred_all==k2]=k1
            else:
                vision_pred_all_new[vision_pred_all_new==k1]=k2
                vision_pred_all[vision_pred_all==k1]=k2
   
    return vision_pred_all,vision_pred_all_new



def get_face_audio(df,corrections):
   
    sample_single = {}
    sample_single_old = []
       
    for idx,row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']
        vpred_old = row['vision_pred']
        
        if not pd.isnull(vpred_old):
            sample_single_old.append(vpred_old)

        if not pd.isnull(vpred):
            emb_file = wavfile.replace('wav','npy')
            if os.path.exists(emb_file):
                tmp_emb = np.load(emb_file)
            if not sample_single.__contains__(vpred):
                sample_single[vpred] = []
                sample_single[vpred].append(tmp_emb)
            else:
                sample_single[vpred].append(tmp_emb)
                

    counts = Counter(sample_single_old)
    sample_single_old= [item for item in sample_single_old if counts[item] >= 2]
 
    sample_singel_emb = {}
    for speaker in sample_single:
       arrays = sample_single[speaker]
       stacked_arrays = np.stack(arrays)
       embedding = np.mean(stacked_arrays, axis=0)
       if not sample_singel_emb.__contains__(speaker):
            sample_singel_emb[speaker] = embedding  

    for speaker in sample_single_old:
        if not sample_singel_emb.__contains__(speaker):
            sample_singel_emb[speaker] = sample_all_emb[speaker]  
                
    return sample_singel_emb



def pred_spk_step1(df, sample_singel_emb, index_off1=60, index_off2=300):
    pred_spk = []
    scores = []
    max_scores = []
    sim_list = []

    time_diff = (df['start_time'] - df['end_time'].shift(1))
    new_conversation = (time_diff > 5000) | time_diff.isnull()
    df['scene'] = new_conversation.cumsum()

    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']

        embedding2 = load_embedding(wavfile)
        next_file = df['files'].iloc[idx+1] if idx < len(df)-1 else wavfile
        embedding3 = load_embedding(next_file)

        tmp_sim = compute_similarity(embedding3, embedding2)
        sim_list.append(tmp_sim)

        if pd.isnull(vpred):
            local_df = get_window_df(df, idx, index_off1)
            local_speakers = get_unique_speakers(local_df)
            local_weights = calculate_distance_weights(local_speakers, idx, local_df)
            local_scores, local_weighted = calculate_weighted_scores(
                sample_singel_emb, embedding2, local_weights
            )

            global_df = get_window_df(df, idx, index_off2)
            global_speakers = get_unique_speakers(global_df)
            global_weights = calculate_distance_weights(global_speakers, idx, global_df)
            global_scores, global_weighted = calculate_weighted_scores(
                sample_singel_emb, embedding2, global_weights
            )


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
                fallback_scores, _ = calculate_weighted_scores(sample_all_emb, embedding2)
                max_key = max(fallback_scores, key=fallback_scores.get)
                pred_spk.append(max_key)
                scores.append(fallback_scores)
                max_scores.append(fallback_scores[max_key])

        else:
            pred_spk.append(vpred)
            scores.append("")
            max_scores.append(0)

    return pred_spk, scores, max_scores, sim_list



def pred_spk_step2(
    df: pd.DataFrame,
    sample_singel_emb: dict,
    window_offset: int = 15,
    fallback_offset: int = 25
):
   """
    Assigns speaker labels to frames without visual predictions based on audio embeddings and local context information.

    Parameters:
        df (pd.DataFrame): DataFrame containing audio paths, timestamps, predictions, etc.
        sample_singel_emb (Dict): Dictionary of representative embeddings for each speaker in the episode.
        window_offset (int): Default window size.
        fallback_offset (int): Fallback window size.

    Returns:
        Tuple[List, List, List, List]: Predicted speakers, similarity score dictionaries, maximum similarity scores, and adjacent frame similarity list.
"""


    pred_spk = []
    scores = []
    max_scores = []

    df['vision_pred_high'] = np.where(
        df['vision_pred'].notna(),
        df['vision_pred'],
        np.where(df['max_scores'] > 0.4, df['step1_spk'], np.nan)
    )

    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']

        embedding2 = load_embedding(wavfile)

        if pd.isnull(vpred):
            local_df = get_window_df(df, idx, window_offset)
            unique_speakers = get_unique_speakers(local_df, 'vision_pred_high')
            if len(unique_speakers) < 2:
                local_df = get_window_df(df, idx, fallback_offset)
                unique_speakers = get_unique_speakers(local_df, 'vision_pred_high')

            weights = calculate_distance_weights(unique_speakers, idx, local_df, 'vision_pred_high')

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



def update_label(df):
    group_id = 0
    groups = []
    
    for i in range(len(df)):
        groups.append(group_id)

        if df.loc[i, 'sim'] <= 0.5:
            group_id += 1

    df['group'] = groups

    df_filtered = df.groupby('group').filter(lambda x: (x['vision_pred_new'].isnull().all()) and (len(x) >= 1))

    group_means = df_filtered.groupby('group')['max_scores'].mean()

    def rename_group_labels(row):
        if row['group'] in group_means.index and group_means[row['group']] < 0.3 and len(row['text'].replace(' ',''))>=2:
            return f"audio_{row['group']}"
        else:
            return row['step2_spk']

    df['updated_label1'] = df.apply(rename_group_labels, axis=1)

    def get_most_common_label(group):
        labels = group['updated_label1']
        scores = group['max_scores']

        mode_labels = labels.mode()
    
        if len(mode_labels) > 1:

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
        if row['group'] in most_common_labels.index and float(row['max_scores']) < 0.5 and float(row['max_scores'])!=0:
            return most_common_labels[row['group']]
        else:
            return row['updated_label1']


    df['updated_label2'] = df.apply(update_label_with_most_common, axis=1)
    return df 



def get_audio_emb(
    df: pd.DataFrame,
    sample_singel_emb: dict
) -> Tuple[dict]:
    """
    Calculates the average embedding for labels in the updated_label2 column that start with 'a',
    and updates the sample_singel_emb and sample_all_emb dictionaries.

    Parameters:
        df (pd.DataFrame): DataFrame containing files, updated_label2, etc.
        sample_singel_emb (dict): Speaker embedding dictionary for each episode.
        sample_all_emb (dict): Global embedding dictionary for all speakers.
        episode_idx (int): Current episode index, used for updating sample_singel_emb.

    Returns:
        Tuple[dict, dict]: Updated sample_singel_emb and sample_all_emb dictionaries.
"""


    def condition_factory(label):
        return lambda row: row['updated_label2'] == label

    unique_labels = set(df['updated_label2'].dropna())
    for label in unique_labels:
        if label.startswith('a'):
            cond_func = condition_factory(label)
            mean_emb, count = compute_mean_embedding(
                df=df,
                condition_func=cond_func
            )
            sample_singel_emb[label] = mean_emb

    return sample_singel_emb

    
def merge_sim_audio(
    df: pd.DataFrame,
    sample_singel_emb: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Merges speakers in the audio category who have high similarity, updates labels, and recalculates the average embedding.

    Parameters:
        df (pd.DataFrame): DataFrame containing updated_label2 and files columns.
        sample_singel_emb (Dict[str, np.ndarray]): Dictionary of representative embeddings for current speakers.

    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray]]: Updated DataFrame and embedding dictionary.
"""


    sim_v2v = build_similarity_matrix(sample_singel_emb)

    label_mapping = build_label_mapping(sim_v2v)

    df['updated_label3'] = df['updated_label2'].replace(label_mapping)

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
    Generates final speaker predictions based on multiple criteria.

    Parameters:
        df (pd.DataFrame): DataFrame containing prediction information.
        indexs (List[int]): List of frame counts for each video segment.
        sample_singel_emb (Dict): Speaker embedding dictionary for each episode.
        sample_all_emb (Dict): Global embedding dictionary for all speakers.
        window_offset (int): Default window size.
        fallback_offset (int): Fallback window size.

    Returns:
        Tuple[List, List, List, List]: Predicted speakers, similarity score dictionaries, maximum similarity scores, and adjacent frame similarity list.
"""


    pred_spk = []
    scores = []
    max_scores = []

    condition1 = df['updated_label3'].str.startswith('a')
    condition2 = df['vision_pred'].notna()
    condition3 = df['max_scores'] > 0.5

    df['vision_pred_high_new'] = np.select(
        [condition1, condition2, condition3],
        [df['updated_label3'], df['vision_pred'], df['step2_spk']],
        default=np.nan
    )


    for idx, row in df.iterrows():
        wavfile = row['files']
        vpred = row['vision_pred_new']
        old_max_score = row['max_scores']

        embedding2 = load_embedding(wavfile)

        if pd.isnull(vpred) and old_max_score < 0.5:

            local_df = get_window_df(df, idx, window_offset)
            unique_speakers = get_unique_speakers(local_df, 'vision_pred_high_new')

            if len(unique_speakers) < 3:
                local_df = get_window_df(df, idx, fallback_offset)
                unique_speakers = get_unique_speakers(local_df, 'vision_pred_high_new')

            weights = calculate_distance_weights(unique_speakers, idx, local_df, 'vision_pred_high_new')

            score_dict, weighted_scores = calculate_weighted_scores(
                sample_singel_emb, embedding2, weights
            )

            if weighted_scores:
                max_key = max(weighted_scores, key=weighted_scores.get)
                pred_spk.append(max_key)
                scores.append(score_dict)
                max_scores.append(score_dict[max_key])
            else:
                score_dict, weighted_scores = calculate_weighted_scores(sample_singel_emb, embedding2)
                max_key = max(score_dict, key=score_dict.get)
                pred_spk.append(max_key)
                scores.append(score_dict)
                max_scores.append(score_dict[max_key])


        else:
            if not pd.isnull(vpred):
                pred_spk.append(vpred)
                scores.append('')
                max_scores.append(1)
            else:
                pred_spk.append(row['updated_label3'])
                scores.append('')
                max_scores.append(old_max_score)


    return pred_spk, scores, max_scores



def update_label_final(df):

    df['final_label'] = df['step3_spk']

    def get_most_common_label(group):
        labels = group['step3_spk']
        scores = group['max_scores_final']
        
        mode_labels = labels.mode()
    
        if len(mode_labels) > 1:

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
        if row['group'] in most_common_labels.index and( (float(row['max_scores_final']) < 0.5 and float(row['max_scores_final'])!=0) or (float(row['max_scores']) < 0.5 and float(row['max_scores']) !=0 and float(row['max_scores_final'])==0)):
            return most_common_labels[row['group']]
        else:
            return row['step3_spk']


    df['final_label'] = df.apply(update_label_with_most_common, axis=1)

    for i in range(len(df)-1):
        current_label = df.loc[i, 'final_label']
        current_score = df.loc[i, 'sim']
        next_max_scores =  df.loc[i+1, 'max_scores_final']
        now_max_scores = df.loc[i, 'max_scores_final']

        if i==0:
            if current_score > 0.45  and (next_max_scores>now_max_scores and  next_max_scores>0.3):
                df.loc[i, 'final_label'] = df.loc[i+1, 'final_label']
        else:
            previous_score = df.loc[i-1, 'sim']
            previous_max_scores = df.loc[i-1, 'max_scores_final']

            # if now_max_scores<0.5:
            if  previous_score > 0.45 and previous_score>current_score and (previous_max_scores>now_max_scores and previous_max_scores>0.3):
                df.loc[i, 'final_label'] = df.loc[i-1, 'final_label']
            if current_score > 0.45 and current_score>previous_score  and (next_max_scores>now_max_scores and  next_max_scores>0.3):
                df.loc[i, 'final_label'] =  df.loc[i+1, 'final_label']
    return df