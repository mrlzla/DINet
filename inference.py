from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptions
from models.DINet import DINet
from detection import init_retinaface_model

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
import face_alignment
import dlib
from numba import cuda 

from collections import OrderedDict
from tqdm.auto import tqdm

FACE_TEMPLATE = np.array([
    [89.12872023809523, 97.39955357142857], 
    [225.22361546499477, 96.19853709508881], 
    [157.5, 186.5], 
    [96.67748917748916, 258.69047619047615], 
    [211.5285285285285, 259.0330330]])
FACE_SIZE = (416, 320)

def extract_frames_from_video(video_path,save_dir, max_len):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    length = min(max_len, int(frames))
    for i in range(length):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.png')
        cv2.imwrite(result_path, frame)
    videoCapture.release()
    return (int(frame_width),int(frame_height))

def get_smoothened_landmarks(landmarks, T=2):
	for i in range(len(landmarks)):
		if i + T > len(landmarks):
			window = landmarks[len(landmarks) - T:]
		else:
			window = landmarks[i : i + T]
		landmarks[i] = np.mean(window, axis=0)
	return landmarks.astype(np.int32)

def get_smoothened_bboxes(landmarks, img_size, T=3, pad=[0.0, 0.1, 0.0, 0.18]):
    bboxes = []
    for landmark in landmarks:
        left_top = np.min(landmark, axis=0)
        right_bottom = np.max(landmark, axis=0)

        y1 = max(0, left_top[1] - pad[1]*(right_bottom[1] - left_top[1]))
        y2 = min(img_size[1], right_bottom[1] + pad[3] * (right_bottom[1] - left_top[1]))
        x1 = max(0, left_top[0] - pad[0]*(right_bottom[0] - left_top[0]))
        x2 = min(img_size[0], right_bottom[0] + pad[2] * (right_bottom[0] - left_top[0]))

        bboxes.append([x1, y1, x2, y2])

    bboxes = np.array(bboxes)

    for i in range(len(bboxes)):
        if i + T > len(bboxes):
            window = bboxes[len(bboxes) - T:]
        else:
            window = bboxes[i : i + T]
        bboxes[i] = np.mean(window, axis=0)
    return bboxes.astype(np.int32)

def get_segmentation(points, shape):
    segmentation = np.zeros(shape, dtype=np.uint8)
    points = points.astype(np.int32)
    points = cv2.convexHull(points).astype(np.int32)
    segmentation = cv2.fillConvexPoly(segmentation, points, color=1)
    cv2.imwrite("segmentation.png", 255*segmentation)

    return segmentation

def get_edge(segmentation):
    total_face_area = np.sum(segmentation)
    return int(total_face_area**0.5) // 20 

def dilate_segmentation(segmentation, ratio=0.75):
    w_edge = get_edge(segmentation)
    dilate_radius = int(w_edge * ratio)

    segmentation = cv2.dilate(segmentation, np.ones((dilate_radius, dilate_radius), np.uint8))
    return segmentation

def smooth_segmentation(segmentation):
    blur_radius = get_edge(segmentation) * 2

    # compute the fusion edge based on the area of face
    segmentation = dilate_segmentation(segmentation)
    cv2.imwrite("dilate_segmentation.png", 255*segmentation)
    soft_mask = cv2.GaussianBlur(segmentation.astype(np.float32), (blur_radius + 1, blur_radius + 1), 11)
    cv2.imwrite("soft_mask.png", (255*soft_mask).astype(np.uint8))
    return soft_mask[..., None]

def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return [det_faces[largest_idx]], largest_idx

def get_face_landmarks_5(input_img,
                        detector,
                        only_keep_largest=True,
                        only_center_face=False,
                        resize=None,
                        blur_ratio=0.01,
                        eye_dist_threshold=None):
        orig_shape = input_img.shape
        if resize is None:
            scale = 1
        else:
            h, w = input_img.shape[0:2]
            scale = resize / min(h, w)
            scale = max(1, scale) # always scale up
            h, w = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            input_img = cv2.resize(input_img, (w, h), interpolation=interp)

        with torch.no_grad():
            bboxes = detector.detect_faces(input_img)

        if bboxes is None or bboxes.shape[0] == 0:
            raise ValueError("No face detected!!")
        else:
            bboxes = bboxes / scale
        
        all_landmarks_5 = []
        det_faces = []

        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            all_landmarks_5.append(landmark)
            det_faces.append(bbox[0:4].astype(np.int32))
            
        if len(det_faces) == 0:
            raise ValueError("No face detected!!")
        if only_keep_largest:
            h, w, _ = orig_shape
            det_faces, largest_idx = get_largest_face(det_faces, h, w)
            all_landmarks_5 = [all_landmarks_5[largest_idx]]
        return det_faces, all_landmarks_5

def align_warp_face(img, landmarks, border_mode='constant'):
    """Align and warp faces with face template.
    """
    affine_matrices = []
    cropped_faces = []
    for idx, landmark in enumerate(landmarks):
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(landmark, FACE_TEMPLATE, method=cv2.LMEDS)[0]
        affine_matrices.append(affine_matrix)
        # warp and crop faces
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        cropped_face = cv2.warpAffine(
            img, affine_matrix, FACE_SIZE, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
        cropped_faces.append(cropped_face)
    return cropped_faces, affine_matrices

def get_inverse_affine(affine_matrices):
    """Get inverse affine matrix."""
    inverse_affine_matrices = []
    for idx, affine_matrix in enumerate(affine_matrices):
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine_matrices.append(inverse_affine)
    return inverse_affine_matrices

if __name__ == '__main__':
    # load config
    opt = DINetInferenceOptions().parse_args()
    if not os.path.exists(opt.source_video_path):
        raise ('wrong video path : {}'.format(opt.source_video_path))
    ############################################## extract deep speech feature ##############################################
    print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
    if not os.path.exists(opt.deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    if not os.path.exists(opt.driving_audio_path):
        raise ('wrong audio path :{}'.format(opt.driving_audio_path))
    ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    del DSModel
    device = cuda.get_current_device()
    device.reset()
    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir, res_frame_length)
    # ############################################## load facial landmark ##############################################
    # print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    # if not os.path.exists(opt.source_openface_landmark_path):
    #     raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
    # video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int32)
    ############################################## extract facial landmarks ##############################################
    detector = init_retinaface_model("retinaface_resnet50")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.png'))
    video_frame_path_list.sort()
    video_frame_path_list = video_frame_path_list[:res_frame_length]
    #video_landmark_data, video_face_bbox_data, video_landmark5_data = [], [], []
    video_face_bbox_data, video_landmark5_data = [], []
    for video_frame_path in tqdm(video_frame_path_list):
        frame = cv2.imread(video_frame_path)[..., ::-1]
        video_face_bbox, video_landmarks5 = get_face_landmarks_5(
            frame, detector, resize=640, eye_dist_threshold=5)
        video_face_bbox_data.append(video_face_bbox)
        video_landmark5_data.append(video_landmarks5)

        # video_landmark = fa.get_landmarks_from_image(frame, 
        #     detected_faces=video_face_bbox)
        # if len(video_landmark) == 0:
        #     raise ValueError("Could not find face on the image")
        #video_landmark = video_landmark[0].astype(np.int32)
        #video_landmark_data.append(video_landmark)
    #video_landmark_data = np.array(video_landmark_data)
    video_landmark5_data = np.array(video_landmark5_data)

    #video_bbox_data = get_smoothened_bboxes(video_landmark_data, video_size)
    # ############################################## align frame with driving audio ##############################################
    # if len(video_frame_path_list) != video_landmark_data.shape[0]:
    #     raise ('video frames are misaligned with detected landmarks')
    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    #video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    #video_bbox_data_cycle = np.concatenate([video_bbox_data, np.flip(video_bbox_data, 0)], 0)
    video_landmark5_data_cycle = np.concatenate([video_landmark5_data, np.flip(video_landmark5_data, 0)], 0)
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
    if video_frame_path_list_cycle_length >= res_frame_length:
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        #res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        #res_video_bbox_data = video_bbox_data_cycle[:res_frame_length]
        res_video_landmark5_data = video_landmark5_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        #res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
        #res_video_bbox_data = np.concatenate([video_bbox_data_cycle]* divisor + [video_bbox_data_cycle[:remainder, :]],0)
        res_video_landmark5_data = np.concatenate([video_landmark5_data_cycle]* divisor + [video_landmark5_data_cycle[:remainder, :, :]],0)

    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                    + res_video_frame_path_list \
                                    + [video_frame_path_list_cycle[-1]] * 2
    #res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    #res_video_bbox_data_pad = np.pad(res_video_bbox_data, ((2, 2), (0, 0)), mode='edge')
    res_video_landmark5_data_pad = np.pad(res_video_landmark5_data, ((2, 2), (0, 0), (0, 0)), mode='edge')

    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0] == res_video_bbox_data_pad.shape[0] == res_video_landmark5_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    ############################################## randomly select 5 reference images ##############################################
    print('selecting five reference images')
    ref_img_list = []
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    for ref_index in ref_index_list:
        #crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
           raise ('our method can not handle videos with large change of facial size!!')
        #crop_radius = opt.mouth_region_size//2
        #crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
        #ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
        #bbox = res_video_bbox_data_pad[ref_index - 3]
        # ref_img_crop = ref_img[
        #           ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
        #           ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
        #           :]
        ref_img_crop = ref_img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        cv2.imwrite(f"logs/ref_{ref_index}_crop.png", ref_img_crop)
        
        ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
        ref_img_crop = ref_img_crop / 255.0
        ref_img_list.append(ref_img_crop)
    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

    ############################################## load pretrained model weight ##############################################
    print('loading pretrained model from: {}'.format(opt.pretrained_clip_DINet_path))
    model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DINet_path):
        raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DINet_path))
    state_dict = torch.load(opt.pretrained_clip_DINet_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    ############################################## inference frame by frame ##############################################
    if not os.path.exists(opt.res_video_dir):
        os.mkdir(opt.res_video_dir)
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    if os.path.exists(res_video_path):
        os.remove(res_video_path)
    res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
    if os.path.exists(res_face_path):
        os.remove(res_face_path)
    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (resize_w, resize_h))
    for clip_end_index in tqdm(range(5, pad_length, 1)):
        crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.2)
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
        bbox = res_video_bbox_data_pad[clip_end_index - 3]

        crop_frame_data = frame_data[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        # crop_frame_data = frame_data[
        #                     frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
        #                     frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
        #                     :]
        crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
        crop_frame_data = crop_frame_data / 255.0
        crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                        opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0

        cv2.imwrite(f"logs/clip_{clip_end_index}_crop_frame_data.png", (255*crop_frame_data).astype(np.uint8))

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
        old_frame = frame_data[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        cv2.imwrite("pre_frame_resize.png", pre_frame_resize)
        
        old_landmark = fa.get_landmarks(old_frame)
        new_landmark = fa.get_landmarks(pre_frame_resize)
        old_mask = get_segmentation(old_landmark[0], old_frame.shape[:2])
        new_mask = get_segmentation(new_landmark[0], pre_frame_resize.shape[:2])
        old_jaw = get_segmentation(old_landmark[0][6:11], old_frame.shape[:2])
        old_jaw = dilate_segmentation(old_jaw, ratio=4.0)
        cv2.imwrite("old_jaw.png", 255*old_jaw)
        mask = np.max([old_mask, new_mask, old_jaw], axis=0)
        mask = smooth_segmentation(mask)
        
        
        res_frame = pre_frame_resize*mask + old_frame*(1-mask)
        res_frame = res_frame.astype(np.uint8)
        
        frame_data[bbox[1]:bbox[3],bbox[0]:bbox[2],:] = res_frame
        #frame_data = blur_border(frame_data, bbox)

        # frame_data[
        # frame_landmark[29, 1] - crop_radius:
        # frame_landmark[29, 1] + crop_radius * 2,
        # frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        # frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        # :] = pre_frame_resize[:crop_radius * 3,:,:]
        videowriter.write(frame_data[:, :, ::-1])
    videowriter.release()
    videowriter_face.release()
    video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')
    if os.path.exists(video_add_audio_path):
        os.remove(video_add_audio_path)
    cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
        res_video_path,
        opt.driving_audio_path,
        video_add_audio_path)
    subprocess.call(cmd, shell=True)







