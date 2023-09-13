import cv2
import torch
import numpy as np

def get_smoothened_landmarks(landmarks, T=2):
    for i in range(len(landmarks)):
        if i + T > len(landmarks):
            window = landmarks[len(landmarks) - T:]
        else:
            window = landmarks[i : i + T]
        landmarks[i] = np.mean(window, axis=0)
    return landmarks

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

def get_edge(segmentation):
    total_face_area = np.sum(segmentation)
    return int(total_face_area**0.5) // 20

def dilate_segmentation(segmentation, ratio=0.75):
    w_edge = get_edge(segmentation)
    dilate_radius = int(w_edge * ratio)

    print(dilate_radius)

    segmentation = cv2.dilate(segmentation, np.ones((dilate_radius, dilate_radius), np.uint8))
    return segmentation

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

def align_warp_face(img, landmarks, face_size, reference_points, border_mode='constant'):
    """Align and warp faces with face template.
    """
    cropped_faces = []
    affine_matrices = []
    inv_affine_matrices = []
    for idx, landmark in enumerate(landmarks):
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(landmark, reference_points, method=cv2.LMEDS)[0]
        affine_matrices.append(affine_matrix)
        # warp and crop faces
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        cropped_face = cv2.warpAffine(
            img, affine_matrix, face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
        cropped_faces.append(cropped_face)

        inv_affine_matrix = get_inverse_affine(affine_matrix)
        inv_affine_matrices.append(inv_affine_matrix)

    return cropped_faces, affine_matrices, inv_affine_matrices

def get_inverse_affine(affine_matrix):
    """Get inverse affine matrix."""
    return cv2.invertAffineTransform(affine_matrix)
