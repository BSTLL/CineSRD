import subprocess
import os
import logging
from multiprocessing import Pool, current_process
import demucs.separate
import pandas as pd
from pydub import AudioSegment
import cv2
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)

def slice_wav(csv_file, wav_file, output_path):
    df = pd.read_csv(csv_file)
    audio = AudioSegment.from_wav(wav_file)
    output_files = []
    for i in range(len(df)):
        start = df['start_time'].iloc[i]
        end = df['end_time'].iloc[i]
        slice = audio[start:end]
        output_file = os.path.join(output_path, str(i).zfill(3) + '.wav')
        output_files.append(output_file)
        slice.export(output_file, format="wav")
        
    df['files'] = output_files
    df.to_csv(csv_file, index=False)

def preprocess(args):
    # Unpack input arguments
    input_csv, input_video, output_folder, gpu_id = args
    # Get current process ID
    process_id = current_process().pid
    try:
        # Set GPU environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Get video filename and extension
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        _, extension = os.path.splitext(input_video)

        # Create output directory for videos
        video_output = os.path.join(output_folder, 'video')
        os.makedirs(video_output, exist_ok=True)
        
        # Construct output video and audio paths
        out_video = os.path.join(video_output, video_name + '.mp4')
        out_wav = out_video.replace('.mp4', '.wav')       

        logging.info("Start converting video format to mp4")

        # Obtain the original video frame rate
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # If it's a high framerate mp4, copy it directly
        if extension == '.mp4' and fps > 24.9 and not os.path.exists(out_video):
            print('copy video')
            shutil.copy(input_video, out_video)
            
        # For non-mp4 or low framerate videos, convert to 25fps mp4
        if (extension != '.mp4' or fps < 24.9) and not os.path.exists(out_video):
            print('The raw video fps is:', fps)
            subprocess.run([
                'ffmpeg', '-nostdin', '-y', '-i', input_video,
                '-qscale:v', '2', '-threads', '16', '-async', '1',
                '-r', '25', out_video, '-loglevel', 'panic'
            ])
            cap = cv2.VideoCapture(out_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print('The new video fps is:', fps)

        logging.info("Start extracting audio from video")

        # Extract audio to 16kHz mono wav file
        if not os.path.exists(out_wav):
            subprocess.run([
                'ffmpeg', '-nostdin', '-y', '-i', input_video,
                '-qscale:a', '0', '-ac', '1', '-vn', '-threads', '16',
                '-ar', '16000', out_wav, '-loglevel', 'panic'
            ])
        
        logging.info("Start slicing audio")

        slice_output = os.path.join(output_folder, 'slice', video_name)
        os.makedirs(slice_output, exist_ok=True)

        slice_wav(input_csv, out_wav, slice_output)
        return out_video
        
    except Exception as e:
        logging.error(f"Process {process_id} - Error processing {input_video} on GPU {gpu_id}: {str(e)}")

def run_preprocessing(csv_list, video_list, output_folder, num_processes=4):
    if len(csv_list) != len(video_list):
        raise ValueError("The lengths of CSV and video lists are inconsistent!")
    task_list = [
        (csv_list[i], video_list[i], output_folder, i % num_processes)
        for i in range(len(csv_list))
    ]
    try:
        with Pool(processes=num_processes) as pool:
            new_video_list = pool.map(preprocess, task_list)
            pool.close()
            pool.join()
        logging.info("All preprocessing tasks completed successfully.")
        return new_video_list
    except Exception as e:
        logging.error(f"Task execution failed: {str(e)}")
