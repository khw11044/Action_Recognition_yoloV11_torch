import cv2
from ultralytics import YOLO
import numpy as np
from utils.tools import h36m_coco_format, show2Dpose, normalize2dhp
import numpy as np
import os

# YOLO Pose 모델 로드
model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능


def extract_keypoints(results):
    results = results.reshape(17,3)
    pose = np.array([[res[0], res[1], res[2]] for idx, res in enumerate(results)]).flatten()
    return pose



# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame = cv2.flip(frame, 1)
    frame_size = frame.shape
    
    try:
        results = model(frame, verbose=False)

        # 결과에서 keypoints 가져오기
        for result in results:
            jointsdata = result.keypoints.data.cpu().numpy() 
            keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
            if scores.any():
                h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
                kps = np.concatenate([h36m_kpts, h36m_scores], axis=2)
                frame = show2Dpose(h36m_kpts, frame)
        

    except:
        print('something error')

    # 프레임 출력
    cv2.imshow('Pose Estimation', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                
cap.release()
cv2.destroyAllWindows()