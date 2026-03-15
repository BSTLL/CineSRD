import os
import sys
import argparse
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from speakerlab.utils.builder import build
from speakerlab.utils.utils import get_logger
from speakerlab.utils.config import build_config
from speakerlab.utils.fileio import load_wav_scp
import pandas as pd 

lang = 'zh'
if lang=='en':
    exp_dir = 'pretrained_models/audio_ext_model/en/'
    config_file = 'conf/eres2net_en.yaml'
    print('Load english audio model')

elif lang == 'zh':
    exp_dir = 'pretrained_models/audio_ext_model/zh/'
    config_file = 'conf/eres2netv2_lm.yaml'
    print('Load chinese audio model')


if torch.cuda.is_available():
    gpu = 1
    device = torch.device('cuda', gpu)
else:
    device = 'cpu'

config = build_config(config_file)
embedding_model = build('embedding_model', config)

# Recover the embedding params of last epoch
config.checkpointer['args']['checkpoints_dir'] = os.path.join(exp_dir, 'models')
config.checkpointer['args']['recoverables'] = {'embedding_model':embedding_model}
checkpointer = build('checkpointer', config)
checkpointer.recover_if_possible(epoch=4, device=device)
embedding_model.to(device)
embedding_model.eval()
feature_extractor = build('feature_extractor', config)


def compute_embedding(wavfile, save=True):
    # load wav
    wav, fs = torchaudio.load(wavfile)
    if fs != 16000:
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[['rate','16000']]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)

    #短音频优化
    data_len = wav.shape[0]
    chunk_len = int(fs)*1
        
    if data_len <= chunk_len:
        wav = F.pad(wav, (0, chunk_len - data_len))
        
    # compute feat
    feat = feature_extractor(wav).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = embedding_model(feat).detach().squeeze(0).cpu().numpy()

    if save:
        emb_file =  wavfile.replace('.wav','_src.npy')
        np.save(emb_file, embedding)
    return embedding

def get_audio(csv_list):
    for csvpath in csv_list:
        df = pd.read_csv(csvpath) 
        for i in range(len(df)):
            wavfile = df['files'].iloc[i]
            emb_file = wavfile.replace('wav','npy')
            emb = compute_embedding(wavfile)
import glob
import os

def get_audio_from_folder(folder):
    csvpaths = glob.glob(os.path.join(folder, "*.csv"))
    for csvpath in csvpaths:
        df = pd.read_csv(csvpath) 
        for i in range(len(df)):
            wavfile = df['files'].iloc[i]
            # emb_file = wavfile.replace('wav','npy')
            emb = compute_embedding(wavfile)
