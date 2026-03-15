import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os, time, torch, cv2, pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import vision_tools.face_recognition as face_recognition
import cv2
import numpy as np
from skimage import transform as trans
import pandas as pd
import requests
import os
import shutil

onnx_dir = 'pretrained_models'
device_id = 1
face_embs_extractor = face_recognition.FaceRecIR101(onnx_dir, 'cpu', device_id)
face_detector = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface',device=f'cpu')

    
def align_face(image, size, lmks):
    dst_w = size[1]
    dst_h = size[0]
    # landmark calculation of dst images
    base_w = 96
    base_h = 112
    assert (dst_w >= base_w)
    assert (dst_h >= base_h)
    base_lmk = [
        30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655,
        62.7299, 92.2041
    ]

    dst_lmk = np.array(base_lmk).reshape((5, 2)).astype(np.float32)
    if dst_w != base_w:
        slide = (dst_w - base_w) / 2
        dst_lmk[:, 0] += slide

    if dst_h != base_h:
        slide = (dst_h - base_h) / 2
        dst_lmk[:, 1] += slide

    src_lmk = lmks
    # using skimage method
    tform = trans.SimilarityTransform()
    tform.estimate(src_lmk, dst_lmk)
    t = tform.params[0:2, :]

    assert (image.shape[2] == 3)

    dst_image = cv2.warpAffine(image.copy(), t, (dst_w, dst_h))
    return dst_image


def get_emb(img,extension):
    emb_file = img.replace(extension,'_emb.npy')
    emb = None
    try:
        face_det = face_detector(img)
        lmks = np.array(face_det['keypoints'][0]).reshape(5,2)
        image = cv2.imread(img)
        align_img = align_face(image, (112,112), lmks)
        emb = face_embs_extractor(align_img)
        np.save(emb_file, emb)
    except:
        print(f'not face:   {emb_file}')
    return emb
    
def get_avatar(avatar_folder):
    for jpg in os.listdir(avatar_folder):
        filename = os.path.join(avatar_folder,jpg)
        _, extension = os.path.splitext(filename)
        if extension in['.jpg','.png','.webp','.jpeg']:
        # try:
            print(filename)
            get_emb(filename,extension)
        # except:
