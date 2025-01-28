import face_alignment
import cv2
import os
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device='cuda', flip_input=False)

image_folder = '/home/vasu/Desktop/smartphone/crp4_light'
left_eye_folder = '/home/vasu/Desktop/smartphone/left_eye_light'
right_eye_folder = '/home/vasu/Desktop/smartphone/right_eye_light'
eye_landmarked_folder = '/home/vasu/Desktop/smartphone/eye_landmarked_faces'

for folder in [left_eye_folder, right_eye_folder, eye_landmarked_folder]:
    os.makedirs(folder, exist_ok=True)

left_eye_indices = list(range(42, 48))
right_eye_indices = list(range(36, 42))

def crop_eyes(image, points, padding=10):
    x_min = max(0, int(min(points[:, 0]) - padding))
    y_min = max(0, int(min(points[:, 1]) - padding))
    x_max = min(image.shape[1], int(max(points[:, 0]) + padding))
    y_max = min(image.shape[0], int(max(points[:, 1]) + padding))
    return image[y_min:y_max, x_min:x_max]

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.png', '.jpg')):
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping: {image_path} (Invalid Image)")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = fa.get_landmarks(rgb_image)

    if not landmarks:
        print(f"No face detected in {image_path}, skipping.")
        continue

    landmark_set = np.array(landmarks[0]) 
    left_eye_lndmrks = landmark_set[left_eye_indices]
    right_eye_lndmrks = landmark_set[right_eye_indices]

    left_eye_crop = crop_eyes(image, left_eye_lndmrks)
    right_eye_crop = crop_eyes(image, right_eye_lndmrks)

    eye_landmarked_image = image.copy()
    for x, y in np.vstack([left_eye_lndmrks, right_eye_lndmrks]):
        cv2.circle(eye_landmarked_image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    target_h = max(left_eye_crop.shape[0], right_eye_crop.shape[0])
    blank_space = np.ones((target_h, 10, 3), dtype=np.uint8) * 255
    
    left_eye_resized = cv2.resize(left_eye_crop, (left_eye_crop.shape[1], target_h)) if left_eye_crop.size != 0 else blank_space
    right_eye_resized = cv2.resize(right_eye_crop, (right_eye_crop.shape[1], target_h)) if right_eye_crop.size != 0 else blank_space
    eye_landmarked_resized = cv2.resize(eye_landmarked_image, (eye_landmarked_image.shape[1], target_h))
    
    combined_image = np.hstack([eye_landmarked_resized, blank_space, left_eye_resized, blank_space, right_eye_resized])

    left_eye_output = os.path.join(left_eye_folder, f"left_eye_{filename}")
    right_eye_output = os.path.join(right_eye_folder, f"right_eye_{filename}")
    eye_landmarked_output = os.path.join(eye_landmarked_folder, f"eye_landmarked_{filename}")

    cv2.imwrite(left_eye_output, left_eye_crop)
    cv2.imwrite(right_eye_output, right_eye_crop)
    cv2.imwrite(eye_landmarked_output, combined_image)

    print(f" Saved: {filename} → Eye Landmarked, Left Eye, and Right Eye Image")

print(" All images processed successfully!")