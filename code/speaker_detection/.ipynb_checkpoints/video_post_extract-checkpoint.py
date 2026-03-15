def get_pkl(pkl_path):
    """
    Function: Read the specified pkl file and return its object (such as dictionary stat_obj).
    Input:
        pkl_path (str): Full path of the pkl file
        show (str, optional): Show name, default is '藏海传'
    Output:
        stat_obj (dict/others): The object deserialized from the pkl file, returns None if the file does not exist
    """
    if not os.path.exists(pkl_path):
        print(f"pkl {pkl_path} not exists!")
        return None
    with open(pkl_path, 'rb') as f:
        stat_obj = pickle.load(f)
    return stat_obj

def get_emb(img,q_flag = False):
    """
    Function: Read the input image, detect the face and get its 5 keypoints for alignment, extract the face embedding and quality score.
    Input:
        img (str): Path to the image file
    Output:
        new_emb (np.ndarray): Extracted face embedding features
        new_score (float): Face quality score
    """
    
    image = cv2.imread(img)
    new_score = 0
    if q_flag:
        new_score = face_quality_evaluator(image)
    face_det = face_detector(image)
    if ('keypoints' not in face_det) or (not face_det['keypoints']) or (len(face_det['keypoints'][0]) < 10):
        print(f"No face/keypoints detected in {img}")
        return None, None

    lmks = np.array(face_det['keypoints'][0]).reshape(5,2)
    new_score = face_quality_evaluator(image)
    align_img = align_face(image, (112,112), lmks)
    new_emb = face_embs_extractor(align_img)
    return new_emb,new_score

def set_rate_pkl_emb(root_prefix, show='藏海传'):
    """
    Function: Iterate over pkl files of the show, and for each path in the pkl, extract the image filename, find the corresponding image in the img folder, extract the embedding, update the new_embedding column, and save the new pkl.
    Input:
        root_prefix (str): Path prefix
        show (str): Show name
    Output:
        None, but the new version of the pkl will be saved in new_pkl_dir
    """
    # img_root = f'{root_prefix}/{show}/resgan_img_0'
    img_source = f'{root_prefix}/{show}/img'
    pkl_dir = f'{root_prefix}/{show}/pkl'
    new_pkl_dir = f'{root_prefix}/{show}/new_pkl'

    for folder_name in sorted(os.listdir(img_source)):
        print(f'================= process {folder_name} =================')
        input_img_dir = os.path.join(img_source, folder_name)
        if not os.path.isdir(input_img_dir):
            continue

        # Load pkl
        pkl_path = f'{pkl_dir}/{str(folder_name).zfill(2)}.pkl'
        stat_obj = get_pkl(pkl_path)
        if stat_obj is None or 'path' not in stat_obj:
            continue

        # Initialize new_embedding column
        num_items = len(stat_obj['path'])
        if 'embedding_rate' not in stat_obj or len(stat_obj['embedding_rate']) != num_items:
            stat_obj['embedding_rate'] = [None] * num_items
        # print(stat_obj['path'])
        for idx, img_rel_path in enumerate(stat_obj['path']):
            img_rate = img_rel_path.split('/')[-1].split('.jp')[0] + '.jpg'     # Only the filename part

            img_full_path2 = os.path.join(img_source, folder_name, img_rate)
            
            
            # if not os.path.exists(img_full_path1):
            #     print(f"Image {img_full_path1} not found for {folder_name}")
            #     continue

            if not os.path.exists(img_full_path2):
                print(f"Image {img_full_path2} not found for {folder_name}")
                continue

            
            # if str(img_source_hq.split('_')[0]) != str(stat_obj['times'][idx]):
            #     print(f"index {img_source_hq.split('_')[0]} not equal for {stat_obj['times'][idx]}")
            #     continue
            

            new_emb2,_ = get_emb(img_full_path2)
            if new_emb2 is not None:
                stat_obj['embedding_rate'][idx] = new_emb2   
            else:
                print(f"idx={idx}: Skip because new_emb2({new_emb2 is not None})")


        os.makedirs(new_pkl_dir, exist_ok=True)
        save_path = f'{new_pkl_dir}/{str(folder_name).zfill(2)}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(stat_obj, f)
        print(f"Updated pkl saved: {save_path}")

    return None
def set_pkl_emb(root_prefix, show='藏海传'):
    """
    Function: Iterate over pkl files of the show, and for each path in the pkl, extract the image filename, find the corresponding image in the img folder, extract the embedding, update the new_embedding column, and save the new pkl.
    Input:
        root_prefix (str): Path prefix
        show (str): Show name
    Output:
        None, but the new version of the pkl will be saved in new_pkl_dir
    """
    img_root = f'{root_prefix}/{show}/resgan_img_0'
    img_source = f'{root_prefix}/{show}/img'
    pkl_dir = f'{root_prefix}/{show}/pkl'
    new_pkl_dir = f'{root_prefix}/{show}/new_pkl'

    for folder_name in sorted(os.listdir(img_root)):
        print(f'================= process {folder_name} =================')
        input_img_dir = os.path.join(img_root, folder_name)
        if not os.path.isdir(input_img_dir):
            continue

        # Load pkl
        pkl_path = f'{pkl_dir}/{str(folder_name).zfill(2)}.pkl'
        stat_obj = get_pkl(pkl_path)
        if stat_obj is None or 'path' not in stat_obj:
            continue

        # Initialize new_embedding column
        num_items = len(stat_obj['path'])
        if 'embedding_hq' not in stat_obj or len(stat_obj['embedding_hq']) != num_items:
            stat_obj['embedding_hq'] = [None] * num_items
            stat_obj['score_hq'] = [None] * num_items
            stat_obj['embedding_rate'] = [None] * num_items
        # print(stat_obj['path'])
        for idx, img_rel_path in enumerate(stat_obj['path']):
            # img_rel_path may be "19/807_0.756_out.jpg" or similar format
            img_source_hq = img_rel_path.split('/')[-1].split('.jp')[0] + '_old_out.jpg'     # Only the filename part
            img_rate = img_rel_path.split('/')[-1].split('.jp')[0] + '.jpg'     # Only the filename part

            
            img_full_path1 = os.path.join(img_root, folder_name, img_source_hq)
            img_full_path2 = os.path.join(img_source, folder_name, img_rate)
            
            
            if not os.path.exists(img_full_path1):
                print(f"Image {img_full_path1} not found for {folder_name}")
                continue

            if not os.path.exists(img_full_path2):
                print(f"Image {img_full_path2} not found for {folder_name}")
                continue

            
            if str(img_source_hq.split('_')[0]) != str(stat_obj['times'][idx]):
                print(f"index {img_source_hq.split('_')[0]} not equal for {stat_obj['times'][idx]}")
                continue
            
            
            new_emb1,newscore = get_emb(img_full_path1,True)
            new_emb2,_ = get_emb(img_full_path2)
            if new_emb1 is not None and new_emb2 is not None:
                stat_obj['embedding_hq'][idx] = new_emb1  
                stat_obj['score_hq'][idx] = newscore  
                stat_obj['embedding_rate'][idx] = new_emb2   
            else:
                print(f"idx={idx}: Skip because new_emb1({new_emb1 is not None}), new_emb2({new_emb2 is not None})")


        os.makedirs(new_pkl_dir, exist_ok=True)
        save_path = f'{new_pkl_dir}/{str(folder_name).zfill(2)}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(stat_obj, f)
        print(f"Updated pkl saved: {save_path}")

    return None

def get_score(img):

    old_img = img.replace('resgan_img_0','img').replace('_out','')

    image = cv2.imread(img)
    new_score = face_quality_evaluator(image)

    image = cv2.imread(old_img)
    old_score = face_quality_evaluator(image)

    return new_score,old_score
    
def statist_scores(root_prefix,show='藏海传'):
    input_root = f'{root_prefix}/{show}/resgan_img_0'

    all_old_resgan_scores =[]
    all_old_scores = []
    all_rate_resgan_scores =[]
    all_rate_scores = []
    for folder_name in sorted(os.listdir(input_root)):
        # if folder_name !='01':
        #     continue
        input_dir = os.path.join(input_root, folder_name)
        eff = 0
        old_resgan_scores =[]
        old_scores = []
        rate_resgan_scores =[]
        rate_scores = []
        for jpg in os.listdir(input_dir):
            filename = os.path.join(input_dir,jpg)
            
            prefix, extension = os.path.splitext(filename)
            # prefix = prefix.split('/')[-1]
            if '.ipy' in prefix:
                print(prefix.split('_'))
                continue
            # if len(prefix.split('_'))!=3:
            #     print(prefix.split('_'))
            #     continue
            # index,q_score,_ = prefix.split('_')
            # index = prefix.split('_')[0]
            if extension in['.jpg','.png','.webp','.jpeg']:
                new_score,old_score = get_score(filename)
                if '_old' in filename:
                    old_resgan_scores.append(float(new_score))
                    old_scores.append(float(old_score))
                else:
                    rate_resgan_scores.append(float(new_score))
                    rate_scores.append(float(old_score))
                # anew_scores += new_scores
                # ascores += scores
        
        if len(os.listdir(input_dir))!=0:
            # print(f'Average scores: {sum(scores)/len(scores)}')
            # print(f'Average new_scores: {sum(new_scores)/len(new_scores)}')
            print(f'source enhanced: {sum(old_resgan_scores)/len(old_resgan_scores)}')
            print(f'source: {sum(old_scores)/len(old_scores)}')
            print(f'rotated enhanced: {sum(rate_resgan_scores)/len(rate_resgan_scores)}')
            print(f'rotated: {sum(rate_scores)/len(rate_scores)}')
            all_old_resgan_scores += old_resgan_scores
            all_old_scores +=  old_scores
            all_rate_resgan_scores+= rate_resgan_scores
            all_rate_scores +=  rate_scores
        else:
            print(os.listdir(input_dir))

        
    print(f'source enhanced: {sum(all_old_resgan_scores)/len(all_old_resgan_scores)}')
    print(f'source: {sum(all_old_scores)/len(all_old_scores)}')
    print(f'rotated enhanced: {sum(all_rate_resgan_scores)/len(all_rate_resgan_scores)}')
    print(f'rotated: {sum(all_rate_scores)/len(all_rate_scores)}')
