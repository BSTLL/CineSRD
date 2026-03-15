import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os, time, torch, cv2, pickle, python_speech_features
from tqdm import tqdm
import vision_tools.face_detection as face_detection
import vision_tools.active_speaker_detection as active_speaker_detection
import vision_tools.face_recognition as face_recognition
import vision_tools.face_quality_assessment as face_quality_assessment
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from skimage import transform as trans
from scipy import signal

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import yaml_config_loader, Config
from concurrent.futures import ProcessPoolExecutor

class VisionProcesser():
    def __init__(
        self, 
        video_file_path, 
        audio_file_path, 
        csv, 
        out_feat_path, 
        out_all_feat_path,
        out_img_path,
        onnx_dir, 
        conf, 
        face_det_stride,
        face_score,
        device='cpu', 
        device_id=0, 
        debug=False,
        out_video_path=None
        ):
        # Read audio data and check the sample rate.
        fs, self.audio = wavfile.read(audio_file_path)
        assert fs == 16000, '[ERROR]: Samplerate of wav must be 16000'
        self.video_id = os.path.basename(video_file_path).rsplit('.', 1)[0]
        self.debug = debug
        # Read video data
        self.cap = cv2.VideoCapture(video_file_path)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('video %s info: w: {}, h: {}, count: {}, fps: {}'.format(w, h, self.count, self.fps) % self.video_id)

        # Initial vision models
        # self.face_detector = face_detection.Predictor(onnx_dir, device, device_id)
        self.face_detector = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface',device=f'gpu:{device_id}')

        self.speaker_detector = active_speaker_detection.ASDTalknet(onnx_dir, device, device_id)
        self.face_quality_evaluator = face_quality_assessment.FaceQualityAssess(onnx_dir, device, device_id)
        self.face_embs_extractor = face_recognition.FaceRecIR101(onnx_dir, device, device_id)

        # Store facial features and related info.
        self.active_facial_embs = {'index_zimu':np.empty((0,), dtype=int), 
        'feat':np.empty((0, 512), dtype=np.float32), 
        'score':np.empty((0,), dtype=np.float32),
        'path':np.empty((0,), dtype=str),
        'best_quality_score':np.empty((0,), dtype=np.float32),
        }
        
        self.all_facial_embs = {'index_zimu':np.empty((0,), dtype=int),
                                   'score':np.empty((0,), dtype=np.float32),
                                   'feat':np.empty((0, 512), dtype=np.float32)
                                  }
        self.csv_file = csv
        self.out_video_path = out_video_path
        self.out_feat_path = out_feat_path
        self.out_all_feat_path = out_all_feat_path
        self.out_img_path = out_img_path
        self.face_score = face_score
        self.min_track = conf['min_track']
        self.num_failed_det = conf['num_failed_det']
        self.crop_scale = conf['crop_scale']
        self.min_face_size = conf['min_face_size']
        # self.face_det_stride = conf['face_det_stride']
        self.face_det_stride = face_det_stride
        self.shot_stride = conf['shot_stride']

        if self.out_video_path is not None:
            # Save active face detection results video (for debugging).
            self.v_out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(w), int(h)))

        # Record the time spent by each module.
        self.elapsed_time = {'faceTime':[], 'trackTime':[], 'cropTime':[],'asdTime':[], 'visTime':[], 'featTime':[]}

    def run(self):
        frames, face_det_frames = [], []
        df = pd.read_csv(self.csv_file)
        scores = []
        for i in tqdm(range(len(df))):
            audio_sample_st = df['start_time'].iloc[i]/1000*16000
            audio_sample_ed = df['end_time'].iloc[i]/1000*16000

            ratio = 640
            
            frame_st, frame_ed = int(audio_sample_st/ratio), int(audio_sample_ed/ratio)
            num_frames = frame_ed - frame_st + 1
            # Go to frame 'frame_st'.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_st)
            index = 0
            for _ in range(num_frames):
                ret, frame = self.cap.read() # ret is a boolean, indicating if the frame was read successfully; frame is the image read.
                if not ret:
                    break
                if index % self.face_det_stride==0: # If index is a multiple of self.face_det_stride add frame for face detection
                    face_det_frames.append(frame)
                    save_path = f"debug/frame/{index}.jpg"  # Define image save path and filename
                    if self.debug:
                        cv2.imwrite(save_path, frame)  # Save image
                frames.append(frame)
                index += 1
                
            if len(frames) != 0:
                audio = self.audio[(frame_st)*ratio:(frame_st + index)*ratio]
                self.process_one_shot(i,frames, face_det_frames, audio, frame_st)
                frames, face_det_frames = [], []
            if self.debug:
                break
        self.cap.release()

        active_facial_embs = {
            'embeddings':self.active_facial_embs['feat'], 
            'score':self.active_facial_embs['score'],
            'times': self.active_facial_embs['index_zimu'],
            'path': self.active_facial_embs['path'],
            'best_quality_score': self.active_facial_embs['best_quality_score'],
        }
        pickle.dump(active_facial_embs, open(self.out_feat_path, 'wb'))
        
        all_elapsed_time = 0
        for k in self.elapsed_time:
            all_elapsed_time += sum(self.elapsed_time[k])
            self.elapsed_time[k] = sum(self.elapsed_time[k])
        elapsed_time_msg = 'The total processing time for %s is %.2fs, including' % (self.video_id, all_elapsed_time)
        for k in self.elapsed_time:
            elapsed_time_msg += ' %s %.2fs,'%(k, self.elapsed_time[k])
        print(elapsed_time_msg[:-1]+'.')
      
    def GetAffinePoints(self,pts_in, trans):
        pts_out = pts_in.copy()
        assert (pts_in.shape[1] == 2)
    
        for k in range(pts_in.shape[0]):
            pts_out[k, 0] = pts_in[k, 0] * trans[0, 0] + pts_in[k, 1] * trans[
                0, 1] + trans[0, 2]
            pts_out[k, 1] = pts_in[k, 0] * trans[1, 0] + pts_in[k, 1] * trans[
                1, 1] + trans[1, 2]
        return pts_out
  

    def align_face(self,image, size, lmks):
        dst_w = size[1]
        dst_h = size[0]
        # Landmark calculation for the dst images
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
        dst_pts = self.GetAffinePoints(src_lmk, t)
        return dst_image, dst_pts

    def pad_to_square(self,im):
        h, w, _ = im.shape
        ns = int(max(h, w) * 1.5)
        im = cv2.copyMakeBorder(im, int((ns - h) / 2), (ns - h) - int((ns - h) / 2), int((ns - w) / 2),
                                (ns - w) - int((ns - w) / 2), cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return im
    


    def transformation_from_points(self,points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        if s1 < 1.0e-4:
            s1 = 1.0e-4
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    
    def rotate(self,im, keypoints):
        h, w, _ = im.shape
        points_array = np.zeros((5, 2))
        dst_mean_face_size = 160
        dst_mean_face = np.asarray([0.31074522411511746, 0.2798131190011913,
                                    0.6892073313037804, 0.2797830232679366,
                                    0.49997367716346774, 0.5099309118810921,
                                    0.35811903020866753, 0.7233174007629063,
                                    0.6418878095835022, 0.7232890570786875])
        dst_mean_face = np.reshape(dst_mean_face, (5, 2)) * dst_mean_face_size
    
        for k in range(5):
            points_array[k, 0] = keypoints[2 * k]
            points_array[k, 1] = keypoints[2 * k + 1]
    
        pts1 = np.float64(np.matrix([[point[0], point[1]] for point in points_array]))
        pts2 = np.float64(np.matrix([[point[0], point[1]] for point in dst_mean_face]))
        trans_mat = self.transformation_from_points(pts1, pts2)
        if trans_mat[1, 1] > 1.0e-4:
            angle = math.atan(trans_mat[1, 0] / trans_mat[1, 1])
        else:
            angle = math.atan(trans_mat[0, 1] / trans_mat[0, 2])
            
        # im = self.pad_to_square(im)
        ns = int(1.5 * max(h, w))
        
        # Calculate new size to fit the rotated image
        cos_angle = np.abs(np.cos(angle))
        sin_angle = np.abs(np.sin(angle))
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Calculate rotation matrix and center rotation
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=-angle / np.pi * 180, scale=1.0)
        
        # Adjust rotation matrix to fit new size
        M[0, 2] += (new_w / 2) - w / 2
        M[1, 2] += (new_h / 2) - h / 2
        
        # Rotate image
        im = cv2.warpAffine(im, M=M, dsize=(new_w, new_h), borderValue=(255, 255, 255))
        
        return im

    def process_one_shot(self, index_zimu,frames, face_det_frames, audio, frame_st=None):
        curTime = time.time()

        dets = self.face_detection(face_det_frames) # Return frame index and face bounding boxes
        faceTime = time.time()

        allTracks, vidTracks = [], []
        allTracks.extend(self.track_shot(dets))

        if self.debug:
            print('dets:',dets)
            print('allTracks:',allTracks)

        trackTime = time.time()

        for ii, track in enumerate(allTracks):
            vidTracks.append(self.crop_video(ii,track, frames, audio))

        cropTime = time.time()

        scores = self.evaluate_asd(vidTracks) # Return speaker activity scores per track
        if self.debug:
            print(scores)
        asdTime = time.time()

        active_facial_embs = self.evaluate_fr(index_zimu,frames, vidTracks, scores)
        
        self.active_facial_embs['index_zimu'] = np.append(self.active_facial_embs['index_zimu'], active_facial_embs['index_zimu'])
        self.active_facial_embs['score'] = np.append(self.active_facial_embs['score'], active_facial_embs['score'])
        self.active_facial_embs['feat'] = np.append(self.active_facial_embs['feat'], active_facial_embs['feat'], axis=0)
        self.active_facial_embs['path'] = np.append(self.active_facial_embs['path'], active_facial_embs['path'], axis=0)
        self.active_facial_embs['best_quality_score'] = np.append(self.active_facial_embs['best_quality_score'], active_facial_embs['best_quality_score'], axis=0)

        featTime =  time.time()
        self.elapsed_time['faceTime'].append(faceTime-curTime)
        self.elapsed_time['trackTime'].append(trackTime-faceTime)
        self.elapsed_time['cropTime'].append(cropTime-trackTime)
        self.elapsed_time['asdTime'].append(asdTime-cropTime)
        self.elapsed_time['featTime'].append(featTime-asdTime)
        
    def face_detection(self, frames):
        dets = []
        for fidx, image in enumerate(frames):
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets.append([])

            face_det =  self.face_detector(image_input)
            face_bboxes = face_det['boxes']
            face_keypoints = face_det['keypoints']
            
            for i in range(len(face_det['scores'])):
                frame_idex = fidx * self.face_det_stride
                dets[-1].append({'frame':frame_idex, 'bbox':(face_bboxes[i]),'keypoints':(face_keypoints[i]) ,'conf':face_det['scores'][i]})  # dets has frame info, bbox info, conf info

        return dets

    def bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # IOU function to calculate overlap between two images
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, scene_faces):
        # Input: scene_faces is a list of lists, each element in the outer list corresponds to a frame, inner list contains bounding boxes for all faces detected in that frame.
        # Output: tracks is a list, each element represents a continuous face track. Each track is a dict with:
        #   'frame': array of frame indices for the track.
        #   'bbox': 2D array, each row is bounding box coordinates.
        # Face tracking
        tracks = []
        while True:   # continuously search for consecutive faces
            track = []
            prev_iou = None  # previous IOU

            for frame_faces in scene_faces:
                for face in frame_faces: # face: includes frame, bbox, conf 
                    if track == []:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:  # face doesn't interrupt for 'num_failed_det' frames
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        # minimum IOU between consecutive faces
                        # compare current iou with previous iou
                        if prev_iou is not None:
                            iou_change = abs(iou - prev_iou)
                        else:
                            iou_change = 0
                        # if iou > 0.5 and iou_change < 0.15:
                        if iou > 0.5:
                            track.append(face)
                            frame_faces.remove(face)
                            prev_iou = iou
                            break
                    else:
                        break
            if track == []:
                break
            elif len(track) > 1 and track[-1]['frame'] - track[0]['frame'] + 1 >= self.min_track:
                frame_num = np.array([ f['frame'] for f in track ])
                
                bboxes = np.array([np.array(f['bbox']) for f in track])
                keypoints = np.array([np.array(f['keypoints']) for f in track])
                
                frameI = np.arange(frame_num[0], frame_num[-1]+1)
                bboxesI = []
                keypointsI = []
                for ij in range(0, 4):
                    interpfn  = interp1d(frame_num, bboxes[:,ij]) # Fill missing boxes by interpolation
                    bboxesI.append(interpfn(frameI))
                    
                for ij in range(0, 10):
                    interpfn_k = interp1d(frame_num, keypoints[:,ij])
                    keypointsI.append(interpfn_k(frameI))

                bboxesI  = np.stack(bboxesI, axis=1)
                keypointsI = np.stack(keypointsI, axis=1)

                tracks.append({'frame':frameI,'bbox':bboxesI,'keypoints':keypointsI})
        return tracks

    def crop_video(self, ii,track, frames, audio):
        # Crop the face clips
        # x: x coordinate of face bbox center
        # y: y coordinate of face bbox center
        # s: face bbox size (usually average width/height)
        crop_frames = []
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y

        if len(dets['s'])>13:
            dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections     
            dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
            dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
                    
        for fidx, frame_idx in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]   # detection box size
            bsi = int(bs * (1 + 2 * cs))  # pad videos by this amount 
            image = frames[frame_idx]          
            frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][fidx] + bsi  # BBox center Y
            mx  = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            cropped_face = cv2.resize(face, (224, 224))
            crop_frames.append(cropped_face)

            if self.debug:
                save_path = f"debug/track_new/{ii}_face_{fidx}_{frame_idx}.jpg"  # Define image save path and filename
                cv2.imwrite(save_path, cropped_face)  # Save image

        cropaudio = audio[track['frame'][0]*640:(track['frame'][-1]+1)*640]
        return {'track':track, 'proc_track':dets, 'data':[crop_frames, cropaudio]}

    def evaluate_asd(self, tracks):
        # Active speaker detection by pretrained TalkNet
        all_scores = []
        for ins in tracks:
            video, audio = ins['data']
            audio_feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
            video_feature = []
            for frame in video:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_feature.append(face)
            video_feature = np.array(video_feature)
            if self.debug:
                print(video_feature.shape)
                print(audio_feature.shape)

            # Remove tail, use minimal audio/video length
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
            audio_feature = audio_feature[:int(round(length * 100)),:]
            video_feature = video_feature[:int(round(length * 25)),:,:]

            audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)
            video_feature = np.expand_dims(video_feature, axis=0).astype(np.float32)

            score = self.speaker_detector(audio_feature, video_feature)
            all_score = np.round(score, 1).astype(float)
            all_scores.append(all_score)	
        return all_scores

    def evaluate_fr(self,index_zimu, frames, tracks, scores):
        # Initialize max speaker score
        max_score = -np.inf
        # Traverse all tracks to find the track with highest score
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            current_max_score = np.max(score)
            
            if current_max_score > max_score:
                max_score = current_max_score
                tidx_with_max_score = tidx
    
        # Initialize return struct: face emb, score, index
        active_facial_embs={'index_zimu':np.empty((0,), dtype=int), 
        'feat':np.empty((0, 512), dtype=np.float32), 
        'score':np.empty((0,), dtype=np.float32),
        'path':np.empty((0,), dtype=str),
        'best_quality_score':np.empty((0,), dtype=np.float32),
        }
        # If valid best score track found, proceed
        if max_score>=0:
            track = tracks[tidx_with_max_score]
            best_quality_score = 0
            # Look for highest quality face frame in best track
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                # Skip frames not at face detection stride
                if fidx % self.face_det_stride != 0:
                    continue
                bbox = track['track']['bbox'][fidx]
                keypoints  = track['track']['keypoints'][fidx]
                # Crop face region
                face_old = frames[frame][max(int(bbox[1]), 0):min(int(bbox[3]), frames[frame].shape[0]), max(int(bbox[0]), 0):min(int(bbox[2]), frames[frame].shape[1])]
                # Face alignment/rotation correction
                face = self.rotate(face_old, keypoints)
                
                # Evaluate face quality (aligned and unaligned)
                face_quality_score = self.face_quality_evaluator(face)
                face_quality_score_old = self.face_quality_evaluator(face_old)
    
                # Debug mode, save debugging image
                if self.debug:
                    save_path = f"debug/evaluate_fr/{face_quality_score_old}_{face_quality_score}_face_{fidx}_active.jpg"
                    cv2.imwrite(save_path, face_old)
                    
                # If better quality and above threshold and better than current best, keep
                if face_quality_score>face_quality_score_old  and face_quality_score>self.face_score and face_quality_score>best_quality_score:
                        best_quality_score = face_quality_score
                        face_save_old = face_old
    
                if face_quality_score_old>face_quality_score and face_quality_score_old>self.face_score and face_quality_score_old>best_quality_score:                   
                        best_quality_score = face_quality_score_old
                        face_save_old = face_old

            if best_quality_score>0:
                try:
                    # Debug: save best face image
                    if self.debug:
                        save_path = f"debug/save_img/{index_zimu}_{best_quality_score}_{max_score}.jpg"
                        cv2.imwrite(save_path, face_old)
                    
                    # Redetect face to get alignment keypoints
                    face_det =  self.face_detector(face_save_old)
                    lmks = np.array(face_det['keypoints'][0]).reshape(5,2)
                    # Face alignment
                    face_align,_ = self.align_face(face_save_old,(112,112), lmks)
                    # Feature extraction
                    best_feature = self.face_embs_extractor(face_align)
        
                    # Store results
                    active_facial_embs['feat'] = np.append(active_facial_embs['feat'], best_feature, axis=0)
                    active_facial_embs['index_zimu'] = np.append(active_facial_embs['index_zimu'], index_zimu)
                    active_facial_embs['score'] = np.append(active_facial_embs['score'], max_score)
                    
                    active_facial_embs['path'] = np.append(active_facial_embs['path'], f"{self.out_img_path}/{index_zimu}_{best_quality_score}.jpg")
                    active_facial_embs['best_quality_score'] = np.append(active_facial_embs['best_quality_score'], best_quality_score)
    
                    # Save aligned/un-aligned face image
                    cv2.imwrite(f"{self.out_img_path}/{index_zimu}_{best_quality_score}.jpg",face_align)
                    cv2.imwrite(f"{self.out_img_path}/{index_zimu}_{best_quality_score}_old.jpg",face_save_old)

                except:
                    # Error handling, avoid crashes due to low quality or other errors
                    print('error')
        # Return structure containing face features, scores, indices
        return active_facial_embs
    def evaluate_fr_all(self, index_zimu, frames, tracks, scores):
        # Initialize return structure
        active_facial_embs = {
            'index_zimu': np.empty((0,), dtype=int),
            'feat': np.empty((0, 512), dtype=np.float32),
            'score': np.empty((0,), dtype=np.float32)
        }
        # Traverse all tracks
        for tidx, track in enumerate(tracks):
            best_quality_score = 0
            face_save_old = None
            # Search for best quality face frame in current track
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                if fidx % self.face_det_stride != 0:
                    continue
                bbox = track['track']['bbox'][fidx]
                keypoints = track['track']['keypoints'][fidx]
                face_old = frames[frame][max(int(bbox[1]), 0):min(int(bbox[3]), frames[frame].shape[0]),
                                         max(int(bbox[0]), 0):min(int(bbox[2]), frames[frame].shape[1])]
                face = self.rotate(face_old, keypoints)
                
                face_quality_score = self.face_quality_evaluator(face)
                face_quality_score_old = self.face_quality_evaluator(face_old)

                if self.debug:
                    save_path = f"debug/evaluate_fr/{face_quality_score_old}_{face_quality_score}_face_{tidx}_{fidx}_active.jpg"
                    cv2.imwrite(save_path, face_old)
                
                if face_quality_score > face_quality_score_old and face_quality_score > self.face_score and face_quality_score > best_quality_score:
                    best_quality_score = face_quality_score
                    face_save_old = face_old
                if face_quality_score_old > face_quality_score and face_quality_score_old > self.face_score and face_quality_score_old > best_quality_score:
                    best_quality_score = face_quality_score_old
                    face_save_old = face_old

            # If best quality frame found for track, extract feature and save
            if best_quality_score > 0:
                try:
                    if self.debug:
                        save_path = f"debug/save_img/{index_zimu}_track{tidx}_{best_quality_score}.jpg"
                        cv2.imwrite(save_path, face_save_old)
                    face_det = self.face_detector(face_save_old)
                    lmks = np.array(face_det['keypoints'][0]).reshape(5,2)
                    face_align, _ = self.align_face(face_save_old, (112,112), lmks)
                    best_feature = self.face_embs_extractor(face_align)
                    # Store results
                    active_facial_embs['feat'] = np.append(active_facial_embs['feat'], best_feature, axis=0)
                    active_facial_embs['index_zimu'] = np.append(active_facial_embs['index_zimu'], index_zimu)
                    active_facial_embs['score'] = np.append(active_facial_embs['score'], best_quality_score)
                except:
                    print('error')
        return active_facial_embs


def process_single_video(csv_file,video_file,output_folder,device_id):
    
    out_feat_path = os.path.join(output_folder,'pkl')
    out_all_feat_path = os.path.join(output_folder,'all_pkl')

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    audio_file = os.path.join(output_folder,'video',video_name+'.wav')
    out_feat_file = f'{out_feat_path}/{video_name}.pkl'
    out_all_feat_file = f'{out_all_feat_path}/{video_name}.pkl'
    out_img_path = os.path.join(output_folder,'img',video_name)
    
    os.makedirs(out_feat_path, exist_ok=True)
    os.makedirs(out_all_feat_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)
    onnx_dir = 'pretrained_models'
    conf_path = 'conf/diar_video.yaml'
    conf = yaml_config_loader(conf_path)

    vprocesser = VisionProcesser(
        video_file, 
        audio_file, 
        csv_file, 
        out_feat_file,
        out_all_feat_file,
        out_img_path, 
        onnx_dir, 
        conf, 
        5,
        0,
        device='gpu', 
        device_id=device_id, 
        debug=False,
        out_video_path=None)
    
    vprocesser.run()
  
def get_face(csv_list,video_list,output_folder,num_gpus=4):
    num_videos = len(video_list)
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        available_gpus = list(range(num_gpus))  # List of available GPU indices

        for i in range(min(num_gpus, num_videos)): 
            if available_gpus:
                device_id = available_gpus.pop(0)  # Get an available GPU
                future = executor.submit(process_single_video,csv_list[i],video_list[i],output_folder,device_id)
                futures[future] = (i, device_id)

        for i in range(num_gpus, num_videos):
            for future in as_completed(futures):
                ep_index, device_id = futures.pop(future)
                result = future.result()
                print(f"Episode {ep_index} processed successfully on GPU {device_id}.")
                available_gpus.append(device_id)  # Release GPU
                # Submit next task
                if available_gpus:
                    device_id = available_gpus.pop(0)
                    future = executor.submit(process_single_video,csv_list[i],video_list[i],output_folder,device_id)
                    futures[future] = (i, device_id)
                    break  # Break out the as_completed loop to avoid duplicate submissions
        for future in as_completed(futures):
            ep_index, device_id = futures[future]
            result = future.result()
            print(f"Episode {ep_index} processed successfully on GPU {device_id}.")
