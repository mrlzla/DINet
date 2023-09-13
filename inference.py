from utils.deep_speech import DeepSpeech
from config.config import DINetInferenceOptions
from models.DINet import DINet
from detection import init_retinaface_model
from utils.face_utils import \
    get_face_landmarks_5, get_smoothened_landmarks, align_warp_face, get_edge

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
import shutil
from numba import cuda

from collections import OrderedDict
from tqdm.auto import tqdm

FACE_SIZE = (320, 416)
FACE_PAD = (152, 208)
FACE_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118]])
TEMPLATE_SIZE = 624
FACE_TEMPLATE = FACE_TEMPLATE_512 * TEMPLATE_SIZE/512

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
    ############################################## extract facial landmarks ##############################################
    detector = init_retinaface_model("retinaface_resnet50")
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.png'))
    video_frame_path_list.sort()
    video_frame_path_list = video_frame_path_list[:res_frame_length]
    video_frames = []
    video_cropped_face_data, video_face_bbox_data = [], []
    video_face_segm_data, video_jam_segm_data, video_landmark5_data = [], [], []
    video_affine_transform_data, video_inv_affine_transform_data = [], []

    for video_frame_path in tqdm(video_frame_path_list):
        frame = cv2.imread(video_frame_path)[..., ::-1]
        face_bbox, landmarks5 = get_face_landmarks_5(
            frame, detector, resize=640, eye_dist_threshold=5)
        try:
            video_frames.append(frame)
            video_face_bbox_data.append(face_bbox[0])
            video_landmark5_data.append(landmarks5[0])
        except Exception as e:
            raise ('Cound not find a face!')

    video_landmark5_data = get_smoothened_landmarks(video_landmark5_data)

    for i, (frame, landmarks5) in tqdm(enumerate(zip(video_frames, video_landmark5_data))):
        size = TEMPLATE_SIZE
        faces_aligned, affine_transforms, inv_transforms = align_warp_face(
            frame, [landmarks5], (size, size), FACE_TEMPLATE)
        video_cropped_face_data.append(faces_aligned[0])
        video_affine_transform_data.append(affine_transforms[0])
        video_inv_affine_transform_data.append(inv_transforms[0])

    # ############################################## align frame with driving audio ##############################################
    frames_count = len(video_frame_path_list)
    if frames_count != len(video_cropped_face_data):
        raise ('video frames are misaligned with detected landmarks')

    video_frame_index_cycle = list(range(frames_count))
    video_frame_index_cycle = video_frame_index_cycle + video_frame_index_cycle[::-1]
    video_frame_index_cycle_length = len(video_frame_index_cycle)
    if video_frame_index_cycle_length >= res_frame_length:
        res_video_index_list = video_frame_index_cycle[:res_frame_length]
    else:
        divisor = res_frame_length // video_frame_index_cycle_length
        remainder = res_frame_length % video_frame_index_cycle_length
        res_video_index_list = video_frame_index_cycle * divisor + video_frame_index_cycle[:remainder]

    res_video_index_list_pad = [res_video_index_list[0]] * 2 \
                                    + res_video_index_list \
                                    + [res_video_index_list[-1]] * 2

    assert ds_feature_padding.shape[0] == len(res_video_index_list_pad)
    pad_length = ds_feature_padding.shape[0]
    ############################################## randomly select 5 reference images ##############################################
    print('selecting five reference images')
    ref_img_list = []
    ref_index_list = random.sample(range(5, len(res_video_index_list_pad) - 2), 5)

    roi_bbox = [
        FACE_PAD[0],
        FACE_PAD[1],
        FACE_PAD[0]+FACE_SIZE[0],
        FACE_PAD[1]+FACE_SIZE[1]
    ]

    for ref_index in ref_index_list:
        index = res_video_index_list_pad[ref_index - 3]
        ref_img = video_cropped_face_data[index].copy()

        ref_img = ref_img[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2]] / 255.0
        ref_img_list.append(ref_img)
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
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, FACE_SIZE)

    mouth_bbox = [
        opt.mouth_region_size//8,
        opt.mouth_region_size//2,
        opt.mouth_region_size//8 + opt.mouth_region_size,
        opt.mouth_region_size//2 + opt.mouth_region_size
    ]

    mask_face_template = np.zeros((TEMPLATE_SIZE,  TEMPLATE_SIZE), dtype=np.float32)
    mask_face_template[
        mouth_bbox[1]+roi_bbox[1]:mouth_bbox[3]+roi_bbox[1],
        mouth_bbox[0]+roi_bbox[0]:mouth_bbox[2]+roi_bbox[0]] = 1

    radius = get_edge(mask_face_template) * 2
    mask_face_template = cv2.boxFilter(mask_face_template, 0, ksize=(2*radius+1, 2*radius+1)) * 255
    #mask = cv2.erode(mask, np.ones((radius, radius), np.uint8)) * 255
    mask_face_template = cv2.GaussianBlur(mask_face_template, (2*radius + 1, 2*radius + 1), 0)
    mask_face_template = cv2.boxFilter(mask_face_template, 0, (2*radius + 1, 2*radius + 1)) / 255.0

    for clip_end_index in tqdm(range(5, pad_length, 1)):
        index = res_video_index_list_pad[clip_end_index - 3]
        face = video_cropped_face_data[index].copy()

        crop_frame_data = face / 255.0
        crop_frame_data = crop_frame_data[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2]]
        crop_frame_data[mouth_bbox[1]:mouth_bbox[3], mouth_bbox[0]:mouth_bbox[2]] = 0

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))

        face[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2]] = pre_frame.astype(np.uint8)
        inverse_affine = video_inv_affine_transform_data[index]

        inv_restored = cv2.warpAffine(face, inverse_affine, video_size)

        # remove the black borders
        mask_border = np.ones((TEMPLATE_SIZE,  TEMPLATE_SIZE), dtype=np.float32)
        mask_border = cv2.warpAffine(mask_border, inverse_affine, video_size)
        mask_border[mask_border < 1.0] = 0
        mask_border = cv2.erode(mask_border, np.ones((4,4), np.uint8))
        inv_restored = mask_border[:, :, None] * inv_restored

        mask = cv2.warpAffine(mask_face_template, inverse_affine, video_size)

        mask = np.min([mask, mask_border], axis=0)
        mask = mask[..., None]

        orig_frame = video_frames[index]
        res_frame = mask * inv_restored + (1 - mask) * orig_frame
        res_frame = res_frame.astype(np.uint8)

        videowriter.write(res_frame[:, :, ::-1])
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







