import face_alignment
import cv2
import matplotlib.pyplot as plt
import os

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd',device='cuda', flip_input=False)

image_folder = '/home/vasu/Desktop/smartphone/crp4_light'
left_eye_folder = '/home/vasu/Desktop/smartphone/left_eye_light'
right_eye_folder = '/home/vasu/Desktop/smartphone/right_eye_light'
# jaw_indices = list(range(0,17))

left_eye_indices = list(range(42,48))
right_eye_indices = list(range(36,42))

def crop_eyes(image,points):
    x_min = int(min(points[:,0]))
    x_max = int(max(points[:,0]))
    y_min = int(min(points[:,1]))
    y_max = int(max(points[:,1]))

    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)

    return image[y_min:y_max, x_min:x_max]

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png",".jpg")):
        image_path = os.path.join(image_folder,filename)
        image = cv2.imread(image_path)

        if image is None:
            print("no images")
            continue

        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        landmarks = fa.get_landmarks(rgb_image)
        if not landmarks:
            print(f"no face detected in{image_path}")
            continue

        landmark_set = landmarks[0]
        left_eye_lndmrks = landmark_set[left_eye_indices]
        right_eye_lndmrks = landmark_set[right_eye_indices]

        left_eye_crop = crop_eyes(image,left_eye_lndmrks)
        right_eye_crop = crop_eyes(image,right_eye_lndmrks)

        left_eye_output = os.path.join(left_eye_folder,f"left_eye_{filename}")
        right_eye_output = os.path.join(right_eye_folder,f"right eye_{filename}")

        cv2.imwrite(left_eye_output, left_eye_crop)
        cv2.imwrite(right_eye_output, right_eye_crop)

        print(f"saved left eye to {left_eye_output} and right eye to {right_eye_output}")

print("all images saved with success")