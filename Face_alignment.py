import face_alignment
import cv2
import matplotlib.pyplot as plt
import os

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd',device='cuda', flip_input=False)

image_folder = '/home/vasu/Desktop/smartphone/crp3_dark'
output_folder = "/home/vasu/Desktop/smartphone/lndmrk_crp3_sfd"

# jaw_indices = list(range(0,17))
left_eye_indices = list(range(42,48))
right_eye_indices = list(range(36,42))

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png",".jpg")):
        image_path = os.path.join(image_folder,filename)
        image = cv2.imread(image_path)

        if image is None:
            print("no images")
            continue

        MIN_DIM = 256
        print(image.shape)
        h, w = image.shape[:2]
        if h < MIN_DIM or w < MIN_DIM:
            scale_factor = MIN_DIM / min(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        landmarks = fa.get_landmarks(rgb_image)
        print(landmarks)
        if not landmarks:
            print(f"no face detected in{image_path}")
            continue

        for landmark_sets in landmarks:
            # for x,y in landmark_sets[jaw_indices]:
            #     cv2.circle(image,(int(x),int(y)),2,(0,255,0),-1)
            for x,y in landmark_sets[left_eye_indices]:
                cv2.circle(image,(int(x),int(y)),2,(255,0,0),-1)
            for x,y in landmark_sets[right_eye_indices]:
                cv2.circle(image,(int(x),int(y)),2,(255,0,0),-1)
        
        output_path = os.path.join(output_folder,filename)
        cv2.imwrite(output_path,image)
        print(f"annotated image saved in : {output_path}")

print(f"all annotated images saved in : {output_folder}")