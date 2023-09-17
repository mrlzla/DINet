from utils.deep_speech import DeepSpeech
from config.config import DINetInferenceOptions
from models.DINet import DINet
from detection import init_retinaface_model
from parsing import init_parsing_model
from utils.face_utils import \
    get_face_landmarks_5, get_smoothened_landmarks, align_warp_face, get_edge, get_parsed_mask

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms.functional import normalize


pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

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
FACE_PAD = (200, 220)
FACE_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118]])
TEMPLATE_SIZE = (512, 512)
DINET_PRE_FACE_SIZE = (712, 712)
DY = 72
FACE_TEMPLATE = FACE_TEMPLATE_512.copy()
#FACE_TEMPLATE[..., 0] *= TEMPLATE_SIZE[0]/512
#FACE_TEMPLATE[..., 1] *= TEMPLATE_SIZE[1]/512

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
    device='cuda'
    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir, res_frame_length)
    ############################################## extract facial landmarks ##############################################
    detector = init_retinaface_model("retinaface_resnet50")
    parser = init_parsing_model(model_name='parsenet')
    codeformer = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    codeformer.load_state_dict(checkpoint)
    codeformer.eval()

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
        w, h = TEMPLATE_SIZE
        faces_aligned, affine_transforms, inv_transforms = align_warp_face(
            frame, [landmarks5], (w, h+DY), FACE_TEMPLATE)
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
        cv2.imwrite('ref.png', ref_img)

        ref_img = cv2.resize(ref_img, DINET_PRE_FACE_SIZE)

        cv2.imwrite('ref_resize.png', ref_img)

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

    mask_face_template = np.zeros((DINET_PRE_FACE_SIZE[1], DINET_PRE_FACE_SIZE[0]) , dtype=np.float32)
    mask_face_template[
        mouth_bbox[1]+roi_bbox[1]:mouth_bbox[3]+roi_bbox[1],
        mouth_bbox[0]+roi_bbox[0]:mouth_bbox[2]+roi_bbox[0]] = 1
    mask_face_template = cv2.resize(
        mask_face_template, 
        (TEMPLATE_SIZE[0], TEMPLATE_SIZE[1] + DY), 
        interpolation=cv2.INTER_NEAREST_EXACT)

    y1 = (mouth_bbox[1]+roi_bbox[1]) * (TEMPLATE_SIZE[1] + DY) / DINET_PRE_FACE_SIZE[0]
    y2 = (mouth_bbox[3]+roi_bbox[1]) * (TEMPLATE_SIZE[1] + DY) / DINET_PRE_FACE_SIZE[0]
    y_mean = int(y1 + (y2-y1)*2/3)
    mask_face_template_copy = mask_face_template.copy()
    mask_face_template_copy = cv2.dilate(mask_face_template_copy, np.ones((4,4), dtype=np.uint8))

    radius = get_edge(mask_face_template) * 2
    mask_face_template = cv2.boxFilter(mask_face_template, 0, ksize=(121, 121))
    # mask_face_template[mouth_bbox[3]+roi_bbox[1]-1:-1] = mask_face_template[mouth_bbox[3]+roi_bbox[1]-1:0:-1][:TEMPLATE_SIZE-mouth_bbox[3]-roi_bbox[1]]
    mask_face_template = 255*mask_face_template
    #mask_face_template = cv2.dilate(mask_face_template, (121, 121), 0)
    mask_face_template = cv2.GaussianBlur(mask_face_template, (2*radius+1, 2*radius+1), 0)
    mask_face_template = cv2.boxFilter(mask_face_template, 0, (61, 61))
    mask_face_template = mask_face_template / 255.0
    #mask_face_template[y_mean:] = mask_face_template_copy[y_mean:]
    #mask = cv2.erode(mask, np.ones((radius, radius), np.uint8)) * 255

    crop_frame_path = "logs/crop_frames"
    shutil.rmtree(crop_frame_path, ignore_errors=True)
    os.makedirs(crop_frame_path, exist_ok=True)

    res_frame_path = "logs/res_frame"
    shutil.rmtree(res_frame_path, ignore_errors=True)
    os.makedirs(res_frame_path, exist_ok=True)

    face_mask_path = "logs/face_mask"
    shutil.rmtree(face_mask_path, ignore_errors=True)
    os.makedirs(face_mask_path, exist_ok=True)

    for clip_end_index in tqdm(range(5, pad_length, 1)):
        index = res_video_index_list_pad[clip_end_index - 3]
        face = video_cropped_face_data[index].copy()

        face = cv2.resize(face, DINET_PRE_FACE_SIZE)

        crop_frame_data = face / 255.0
        crop_frame_data = crop_frame_data[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2]]
        crop_frame_data[mouth_bbox[1]:mouth_bbox[3], mouth_bbox[0]:mouth_bbox[2]] = 0

        cv2.imwrite(f"{crop_frame_path}/{clip_end_index-3:06d}_{index:06d}.png", 255*crop_frame_data)

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))

        #cv2.imwrite(f"{res_frame_path}/{clip_end_index-3:06d}_{index:06d}.png", pre_frame)

        face[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2]] = pre_frame.astype(np.uint8)
        face = cv2.resize(face, (TEMPLATE_SIZE[0], TEMPLATE_SIZE[1] + DY))

        inverse_affine = video_inv_affine_transform_data[index]

        inv_restored = cv2.warpAffine(face, inverse_affine, video_size)

        # remove the black borders
        mask_border = np.ones((TEMPLATE_SIZE[1]+DY,  TEMPLATE_SIZE[0]), dtype=np.float32)
        mask_border = cv2.warpAffine(mask_border, inverse_affine, video_size)
        #mask_border[mask_border < 1.0] = 0
        mask_border = cv2.erode(mask_border, np.ones((4,4), np.uint8))
        inv_restored = mask_border[:, :, None] * inv_restored
                
        orig_frame = video_frames[index]
        res_frame = mask_border[..., None] * inv_restored + \
            (1 - mask_border[..., None]) * orig_frame
                
        mask_face = get_parsed_mask(face[:512], parser)
        mask_face = np.concatenate([mask_face, np.zeros((DY, TEMPLATE_SIZE[0]))], axis=0)

        mask_face = cv2.warpAffine(mask_face, inverse_affine, video_size)
        #mask_face[mask_face < 255] = 0
        #mask_face = cv2.dilate(mask_face, np.ones((6,6), np.uint8))
        mask_face /= 255.0

        inv_mask_face_template = cv2.warpAffine(mask_face_template, inverse_affine, video_size)
        mask = np.min([mask_face, inv_mask_face_template], axis=0)

                    #0.3*255*mask[..., None] + 0.7 * res_frame)

        cv2.imwrite(f"{res_frame_path}/{clip_end_index-3:06d}_{index:06d}.png", 
                    0.7*orig_frame[..., ::-1] + 0.3 * 255*mask[..., None])

        # mask = np.min([mask, mask_border], axis=0)[..., None]
        # cv2.imwrite('mask.png', 255*mask)
        res_frame = mask[..., None] * res_frame + (1 - mask[..., None]) * orig_frame
        
        res_frame = res_frame.astype(np.uint8)

        if opt.use_codeformer:
            landmark5 = video_landmark5_data[index]

            faces_aligned, affine_transforms, inv_transforms = align_warp_face(
                res_frame, [landmarks5], TEMPLATE_SIZE, FACE_TEMPLATE_512)

            cropped_face_t = img2tensor(faces_aligned[0] / 255., bgr2rgb=False, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            w=opt.codeformer_w
            with torch.no_grad():
                output = codeformer(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=False, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
            restored_face = restored_face.astype('uint8')
            
            inv_restored = cv2.warpAffine(restored_face, inv_transforms[0], video_size)

            mask_border = np.ones((TEMPLATE_SIZE[1],  TEMPLATE_SIZE[0]), dtype=np.float32)
            mask_border = cv2.warpAffine(mask_border, inv_transforms[0], video_size)
            mask_border = cv2.erode(mask_border, np.ones((4,4), np.uint8))
            pasted_face = mask_border[:, :, None] * inv_restored

            # compute the fusion edge based on the area of face
            w_edge = get_edge(mask_border)
            radius = w_edge * 2
            mask_border = cv2.erode(mask_border, np.ones((radius, radius), np.uint8))
            mask_border = cv2.GaussianBlur(mask_border, (radius + 1, radius + 1), 0)

            mask_border = np.min([mask_border, mask], axis=0)

            cv2.imwrite(f"{face_mask_path}/{clip_end_index-3:06d}_{index:06d}.png", 255*mask_border)

            res_frame = mask_border[..., None] * pasted_face + \
                (1 - mask_border[..., None]) * res_frame
            
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







