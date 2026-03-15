from video_preprocess import run_preprocessing
from video_extract import get_face
from audio_extract import get_audio
from avatar_extract import get_avatar
from ass2csv import zimu_analysis
from infer import predict
import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_file_paths(base_folder, txt_file):
    video_list, csv_list = [], []
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            csv_path, video_path = line.strip().split('\t')
            video_file  = os.path.join(base_folder, 'video', video_path)
            csv_file = os.path.join(base_folder, 'subtitle', csv_path)
            video_list.append(video_file)
            csv_list.append(csv_file)
    return csv_list, video_list

def reformat_subtitle(subtitle_list):
    return [zimu_analysis(i) for i in subtitle_list]

def main():
    config = load_config('config.yaml')
    video_cfg = config['video']
    for show_name in video_cfg['show_list']:
        base_folder   = os.path.join(video_cfg['input_root'], show_name)
        avatar_folder = os.path.join(base_folder, video_cfg['avatar_subfolder'])
        output_folder = os.path.join(video_cfg['output_root'], show_name)
        os.makedirs(output_folder, exist_ok=True)
        txt_file      = os.path.join(base_folder, f'{show_name}.txt')

        subtitle_list, video_list = read_file_paths(base_folder, txt_file)
        logger.info("Subtitle analysis")
        csv_output_list = reformat_subtitle(subtitle_list)

        logger.info("Video preprocessing")
        video_output_list = run_preprocessing(
            csv_output_list, video_list, output_folder,
            num_processes=video_cfg['num_processes']
        )

        logger.info("Active speaker detection, extracting video features")
        pkl_list = get_face(
            csv_output_list, video_output_list, output_folder,
            num_gpus=video_cfg['num_gpus']
        )

        logger.info("Extracting audio speaker features")
        get_audio(csv_output_list)

        logger.info("Extracting character avatars") 
        get_avatar(avatar_folder)

        logger.info("Predicting labels")
        predict(csv_output_list, video_output_list, avatar_folder, output_folder)

if __name__ == "__main__":
    main()
