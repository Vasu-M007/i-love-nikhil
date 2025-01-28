# import face_alignment
# import cv2
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd',device='cuda', flip_input=False)

# lk_params = dict(winSize = (15,15),maxlevel=2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))


# image_folder = '/home/vasu/Desktop/smartphone/crp4_light'
# output_folder = "/home/vasu/Desktop/smartphone/landmark_normalized_crp4"


# left_eye_indices = list(range(42,48))
# right_eye_indices = list(range(36,42))

# for filename in os.listdir(image_folder):
#     if filename.lower().endswith((".png",".jpg")):
#          #for first cropped face
#         first_img_path = os.path.join(image_folder,filename)
#         frame1 = cv2.imread(first_img_path)
#         gray_frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#         rgb_frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)

#         MIN_DIM = 256
#         h, w = frame1.shape[:2]
#         if h < MIN_DIM or w < MIN_DIM:
#             scale_factor = MIN_DIM / min(h, w)
#             new_w = int(w * scale_factor)
#             new_h = int(h * scale_factor)
#             frame1 = cv2.resize(frame1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#         landmarks = fa.get_landmarks(rgb_frame1)
#         if not landmarks:
#             print(f"no face detected in{first_img_path}")
#             continue

#         eye_landmarks = np.vstack([landmarks[0][left_eye_indices], landmarks[0][right_eye_indices]])
#         p0 = np.array(eye_landmarks, dtype=np.float32).reshape(-1, 1, 2)


#         images_path = os.path.join(image_folder,filename)
#         frame2 = cv2.imread(images_path)
#         gray_frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#         rgb_frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)

#         p1,st,err = cv2.calcOpticalFlowPyrLK(gray_frame1,gray_frame2,p0,None,**lk_params)

#         good_new = p1[st == 1]
#         good_old = p0[st == 1]

#         for pt in good_new:
#             x,y = pt.ravel()
#             cv2.circle(frame2,int(x),int(y),1,(0,0,255),-1)

#         if frame1 is None:
#             print("no images")
#             continue   

#         MIN_DIM = 256
#         h, w = frame2.shape[:2]
#         if h < MIN_DIM or w < MIN_DIM:
#             scale_factor = MIN_DIM / min(h, w)
#             new_w = int(w * scale_factor)
#             new_h = int(h * scale_factor)
#             frame2 = cv2.resize(frame2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#         # for landmark_sets in landmarks:
#         #     # for x,y in landmark_sets[jaw_indices]:
#         #     #     cv2.circle(image,(int(x),int(y)),2,(0,255,0),-1)
#         #     for x,y in landmark_sets[left_eye_indices]:
#         #         cv2.circle(image,(int(x),int(y)),2,(255,0,0),-1)
#         #     for x,y in landmark_sets[right_eye_indices]:
#         #         cv2.circle(image,(int(x),int(y)),2,(255,0,0),-1)
        
#         output_path = os.path.join(output_folder,filename)
#         cv2.imwrite(output_path,frame2)

#         gray_frame1 = gray_frame2.copy()
#         p0 = good_new.reshape(-1, 1, 2)
        
#         print(f"Processed: {filename} -> Saved to {output_path}")

# print(f"all annotated images saved in : {output_folder}")

import face_alignment
import cv2
import os
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                  face_detector='sfd', device='cuda', flip_input=False)


lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


image_folder = "/home/vasu/Desktop/smartphone/crp4_light"
output_folder = "/home/vasu/Desktop/smartphone/landmark_normalized_crp4"
eye_crop_folder = "/home/vasu/Desktop/smartphone/eye_crops"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(eye_crop_folder, exist_ok=True)


left_eye_indices = list(range(42, 48))
right_eye_indices = list(range(36, 42))

image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg"))])

if not image_files:
    print("No images found in folder!")
    exit()

first_img_path = os.path.join(image_folder, image_files[0])
frame1 = cv2.imread(first_img_path)

if frame1 is None:
    print(f"Error loading first image: {first_img_path}")
    exit()

rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


landmarks = fa.get_landmarks(rgb_frame1)

if landmarks is None:
    print(f"No face detected in {first_img_path}")
    exit()

eye_landmarks = np.vstack([landmarks[0][left_eye_indices], landmarks[0][right_eye_indices]])
p0 = np.array(eye_landmarks, dtype=np.float32).reshape(-1, 1, 2)


for x, y in eye_landmarks:
    cv2.circle(frame1, (int(x), int(y)), 2, (0, 255, 0), -1)

cv2.imwrite(os.path.join(output_folder, "first_image_landmarks.png"), frame1)
print(f"Saved first image with landmarks: first_image_landmarks.png")


for filename in image_files[1:]:
    img_path = os.path.join(image_folder, filename)
    frame2 = cv2.imread(img_path)

    if frame2 is None:
        print(f"Error loading image: {img_path}")
        continue

    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if gray_frame1.shape != gray_frame2.shape:
        gray_frame2 = cv2.resize(gray_frame2, (gray_frame1.shape[1], gray_frame1.shape[0]), interpolation=cv2.INTER_LINEAR)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frame1, gray_frame2, p0, None, **lk_params)

    if p1 is not None and st.sum() >= len(p0) // 2:  
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        
        for pt in good_new:
            x, y = pt.ravel()
            cv2.circle(frame2, (int(x), int(y)), 2, (255, 0, 0), -1)

       
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame2)

        x_min, y_min = np.min(good_new, axis=0).astype(int)
        x_max, y_max = np.max(good_new, axis=0).astype(int)
        
        
        pad = 10
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(frame2.shape[1], x_max + pad), min(frame2.shape[0], y_max + pad)

    
        eye_crop = frame2[y_min:y_max, x_min:x_max]
        eye_crop_path = os.path.join(eye_crop_folder, f"eye_{filename}")

        if eye_crop.size > 0:
            cv2.imwrite(eye_crop_path, eye_crop)

        gray_frame1 = gray_frame2.copy()
        p0 = good_new.reshape(-1, 1, 2)

        print(f"Processed: {filename} -> Saved to {output_path} & Eye Crop to {eye_crop_path}")
    else:
        print(f"Skipping {filename}: Tracking failed.")

print(f"All images processed. Annotated images saved in {output_folder} and cropped eyes saved in {eye_crop_folder}.")
