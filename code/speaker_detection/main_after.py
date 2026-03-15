import os
import json
import yaml
import warnings
import torch
import pandas as pd
from collections import defaultdict

from cluster import CommonClustering
from utils import *
from process import *
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config('config.yaml')
show_list = config['common']['show_list']

# Required fields, can be extended if needed
SAVE_COLUMNS = [
    'start_time', 'end_time', 'speaker', 'text', 'final_label','new_final',
    'vision_pred_new', 'audio_pred'
]

cluster_v = CommonClustering(
    cluster_type='AHC',
    cluster_line=2,
    mer_cos=0.9,
    min_cluster_size=1,
    fix_cos_thr=0.3
)

cluster_a = CommonClustering(
    cluster_type='spectral',
    min_num_spks=1,
    max_num_spks=20,
    min_cluster_size=5,
    pval=0.032,
    mer_cos=0.8,
    oracle_num=17
)

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def calc_major_label(df):
    vision_col = 'vision_pred_new'
    col_list = []
    if vision_col in df.columns:
        col_list.append(vision_col)
    col_list.append('final_label')
    def choose_label(group):
        if vision_col in group.columns:
            vals = group[vision_col]
            vals = vals[vals.notna() & (vals != '')]
            if not vals.empty:
                return vals.mode().iloc[0]
        vals = group['final_label']
        vals = vals[vals.notna() & (vals != '')]
        if not vals.empty:
            return vals.mode().iloc[0]
        return ''
    major_label_dict = (
        df.groupby('speaker')[col_list]
        .apply(choose_label)
        .to_dict()
    )
    df['spk_major_label'] = df['speaker'].map(major_label_dict)
    return df

def calculate_all_metrics(
    df,
    pred_col,
    ref_col='speaker', 
    start_col='start_time', end_col='end_time', text_col='text'
):
    df = calc_major_label(df)
    if not isinstance(pred_col, list):
        pred_col_list = [pred_col]
    else:
        pred_col_list = pred_col

    all_metrics = {}
    for this_pred_col in pred_col_list:
        if this_pred_col not in df.columns or ref_col not in df.columns:
            continue

        has_timestamps = start_col in df.columns and end_col in df.columns
        s_col, e_col = start_col, end_col
        if not has_timestamps:
            df = df.copy()
            df['temp_start'] = range(len(df))
            df['temp_end'] = range(1, len(df) + 1)
            s_col, e_col = 'temp_start', 'temp_end'

        reference = Annotation()
        hypothesis = Annotation()

        for _, row in df.iterrows():
            if pd.isnull(row[ref_col]):
                continue
            start, end = row[s_col], row[e_col]
            if pd.isnull(start) or pd.isnull(end) or end <= start:
                continue
            ref_speaker = str(row[ref_col])
            segment = Segment(start, end)
            reference[segment] = ref_speaker
            if not pd.isnull(row[this_pred_col]):
                pred_speaker = str(row[this_pred_col])
                hypothesis[segment] = pred_speaker

        if not reference:
            continue

        metric_collar = 0.25 if has_timestamps else 0.0
        der_metric = DiarizationErrorRate(collar=metric_collar, skip_overlap=False)
        jer_metric = JaccardErrorRate(collar=metric_collar, skip_overlap=False)
        der_components = der_metric(reference, hypothesis, detailed=True)
        jer_components = jer_metric(reference, hypothesis, detailed=True)
        mapping_res = jer_metric.optimal_mapping(reference, hypothesis)

        text_der = None
        if text_col in df.columns:
            char_error_count = 0
            total_char_count = 0
            for _, row in df.iterrows():
                if pd.isnull(row[ref_col]):
                    continue
                text = str(row[text_col]) if not pd.isnull(row[text_col]) else ""
                chinese_chars = len([ch for ch in text if '\u4e00' <= ch <= '\u9fff'])
                english_chars = len([ch for ch in text if ch.isalpha()])
                num_chars = len(text) if chinese_chars > english_chars else len(text.split())
                total_char_count += num_chars
                ref_speaker = str(row[ref_col])
                ref_speaker_map = str(row['spk_major_label'])
                pred_speaker = str(row[this_pred_col]) if not pd.isnull(row[this_pred_col]) else ""
                mapped_pred_speaker = mapping_res.get(pred_speaker, None)
                if mapped_pred_speaker:
                    if mapped_pred_speaker != ref_speaker:
                        char_error_count += num_chars
                else:
                    if pred_speaker != ref_speaker_map:
                        char_error_count += num_chars
            text_der = char_error_count / total_char_count if total_char_count > 0 else 0.0

        all_metrics[this_pred_col] = {
            'DER': der_components.get('diarization error rate', 0.0),
            'JER': jer_components.get('jaccard error rate', 0.0),
            'TextDER': text_der
        }

    return all_metrics

def get_mean_metrics_muti(results_list):
    all_colnames = set()
    for result in results_list:
        if isinstance(result, dict):
            all_colnames.update(result.keys())
    avg_metrics = {}
    for col in all_colnames:
        col_sum_metrics = defaultdict(float)
        col_count_metrics = defaultdict(int)
        for res in results_list:
            if isinstance(res, dict) and col in res:
                metrics_dict = res[col]
                for key, value in metrics_dict.items():
                    if value is not None:
                        col_sum_metrics[key] += value
                        col_count_metrics[key] += 1
        metrics_avg = {}
        for name in ['DER', 'JER', 'TextDER']:
            if col_count_metrics[name] > 0:
                metrics_avg[name] = col_sum_metrics[name] / col_count_metrics[name]
        avg_metrics[col] = metrics_avg
    return avg_metrics

def predict(show, config):
    s_thold = config['common']['s_thold']
    source_data_root = config['common']['source_data_root']
    data_root = config['common']['data_root']
    result_save_root = config['common']['result_save_root']
    print(f'============= predict {show} ===============')
    csv_dir =  f"{source_data_root}/{show}/csv"
    prefix = f"{data_root}/{show}_exp"
    pkl_dir =  f"{prefix}/pkl"
    all_pkl_dir = f"{prefix}/all_pkl"
    avatar_dir = f"{source_data_root}/{show}/avatar/"
    file_path = f"{source_data_root}/{show}/all_fusion_prob.csv"
    savepath = f"{result_save_root}/{show}"
    os.makedirs(savepath, exist_ok=True)
    save_metric = f'{savepath}/test_metric.json'

    if os.path.exists(file_path):
        sim_df = pd.read_csv(file_path)
        print("sim file loaded")
    else:
        sim_df = None
        print("sim file does not exist, not loaded")
    res_all = []

    for i in range(1, 21):
        ep = str(i).zfill(2)
        csvpaths = [f'{csv_dir}/{ep}.csv']
        pklpaths = [f'{pkl_dir}/{ep}.pkl']
        all_pkl_path = f'{all_pkl_dir}/{ep}.pkl'
        if not os.path.exists(pklpaths[0]) or not os.path.exists(csvpaths[0]):
            continue
        for csv_path in csvpaths:
            df = pd.read_csv(csv_path)
            if 'speaker' in df.columns:
                df = df.rename(columns={'speaker': 'speaker'})
                df.to_csv(csv_path, index=False)
        save_file = f'{savepath}/{ep}.csv'
        # Get features and prediction labels
        vision_embeddings, audio_embeddings, vision_times, indexs, active_scores, df = get_raw_emb(csvpaths, pklpaths, True)
        all_vision_embeddings, all_vision_times, all_active_scores = get_all_pkl(all_pkl_path)
        vision_pred, vision_mat, centers = cluster_v(vision_embeddings)
        spk_dict = get_role_name(avatar_dir, centers)
        vision_pred_all, active_scores_all, vision_embeddings_all = get_vision_pred(
            vision_times, indexs, vision_pred, spk_dict, active_scores, [vision_embeddings])
        audio_pred, audio_mat, _ = cluster_a(audio_embeddings)
        df['vision_pred'] = vision_pred_all
        df['active_scores'] = active_scores_all
        df['audio_pred'] = audio_pred
        vision_pred_all_new, corrections = face_filter(df, audio_pred, vision_pred_all, active_scores_all)
        df['vision_pred_new'] = vision_pred_all_new
        vision_pred_all, vision_pred_all_new = merge_sim_face_by_audio(df, vision_pred_all, vision_pred_all_new)
        df['vision_pred_new'] = vision_pred_all_new
        df['vision_pred'] = vision_pred_all
        sample_emb = get_face_audio(df, corrections)
        df['vis_emb_active'] = all_vision_embeddings
        sample_vis_emb = get_face_emb_dict(df)
        # Clear intermediate embeddings
        if 'vis_emb_active' in df.columns:
            df.drop(columns=['vis_emb_active'], inplace=True)
        scores_all_role, vision_embeddings_all_role = get_vision_pred_all(all_vision_times, indexs, all_active_scores, all_vision_embeddings)
        df['scores_all_role'] = scores_all_role
        df['vision_embeddings_all_role'] = vision_embeddings_all_role
        roles, role_scores = get_speakers_by_emb(df, sample_vis_emb)
        df['roles'] = roles
        df['role_scores'] = role_scores
        if 'vision_embeddings_all_role' in df.columns:
            df.drop(columns=['vision_embeddings_all_role'], inplace=True)
        pred_spk, scores, max_scores, sim = pred_spk_step1(df, sample_emb, index_off1=100, index_off2=1000)
        df['sim'] = sim
        if sim_df is not None:
            sim_cleaned = list(sim_df[ep][1:len(df)]) + [0.99]
            assert len(sim_cleaned) == len(df), "Length mismatch, cannot assign"
            df['muti_sim'] = sim_cleaned
            df['sim'] = sim_cleaned
        df['scores'] = scores
        df['max_scores'] = max_scores
        df['step1_spk'] = pred_spk
        pred_spk, scores, max_scores = pred_spk_window_byface(df, sample_emb, index_off1=100, index_off2=1000)
        df['face_scores'] = scores
        df['face_max_scores'] = max_scores
        df['face_spk'] = pred_spk
        pred_spk, scores, max_scores = pred_spk_step2(df, sample_emb, window_offset=15, fallback_offset=25)
        df['scores'] = scores
        df['max_scores'] = max_scores
        df['step2_spk'] = pred_spk
        df = update_label(df, s_thold)
        sample_emb = get_audio_emb(df, sample_emb)
        df, sample_emb1 = merge_sim_audio(df, sample_emb)
        pred_spk, scores, max_scores = get_final_spk(df, sample_emb)
        df['scores_final'] = scores
        df['max_scores_final'] = max_scores
        df['step3_spk'] = pred_spk
        df = update_label_final(df)
        df = merge_label_by_sim(df)
        save_cols = [col for col in SAVE_COLUMNS if col in df.columns]
        df_save = df[save_cols].copy()
        df_save.to_csv(save_file, index=False, encoding='utf-8-sig')

        pred_col = ['final_label', 'vision_pred_new', 'audio_pred']
        metrics = calculate_all_metrics(df, pred_col=pred_col)
        if metrics:
            res_all.append(metrics)
            print(f"{ep}-Evaluation result:")
            for col, m_dict in metrics.items():
                print(f"  [Prediction column: {col}]", m_dict)

    mean_metrics = get_mean_metrics_muti(res_all)
    for col, m_dict in mean_metrics.items():
        print(f"[Prediction column: {col}]", m_dict)
    res_all.append(mean_metrics)
    with open(save_metric, "w", encoding="utf-8") as f:
        json.dump(res_all, f, ensure_ascii=False, indent=2)
    print(f"\nres_all has been saved to {save_metric}\n")
    return mean_metrics

if __name__ == "__main__":
    for show in show_list:
        predict(show, config)
