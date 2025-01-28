from ultralytics import YOLO
import cv2
import os

video = '/dev/video0'
cap = cv2.VideoCapture(video)

facemodel = YOLO('yolov8n-face.pt')

crop_faces_dir = '/home/vasu/Desktop/smartphone/crp5                                                                                                                                                                                                                                                                                '
if not os.path.exists(crop_faces_dir):
    os.mkdir(crop_faces_dir)

output_path = "face_cropped_vid9.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
idx = 0
while cap.isOpened():
    rt, frame = cap.read()

    face_result = facemodel.predict(frame,conf = 0.40)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            idx += 1
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop_obj = frame[y1:y2,x1:x2]
            cv2.imwrite(os.path.join(crop_faces_dir,str(idx)+'.png'),crop_obj)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)


    cv2.imshow('frame', frame)
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()