import cv2
from ultralytics import YOLO
import numpy as np
from utils.tools import h36m_coco_format, show2Dpose, normalize2dhp, prob_viz
import numpy as np
import os
import torch
# PyTorch 모델 불러오기
from utils.network1 import ActionBiLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능

# Actions 설정
actions = np.array(['nothing', 'ready', 'stop', 'emergency'])  # 구분할 동작
num_classes = len(actions)
sequence_length = 15  # 프레임 길이
input_size = 34
threshold = 0.6      # 예측 임계값 


# LSTM 모델 불러오기
model = ActionBiLSTM(num_classes, sequence_length, input_size).to(device)
model.load_state_dict(torch.load("./models/best_kp.pth", map_location=device))
model.eval()

# Keypoints 추출 함수
def extract_keypoints(results):
    results = results.reshape(17, 3)  # (17, 3) -> x, y, confidence
    pose = np.array([[res[0], res[1]] for res in results]).flatten()  # x, y만 사용
    return pose

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# 변수 초기화
sequence = []  # 15 프레임씩 쌓을 리스트
sentence = []  # 예측된 동작 저장 리스트


# 실시간 웹캠 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # 좌우 반전
    frame_size = frame.shape

    # YOLO Pose 모델로 키포인트 추출
    results = pose_model(frame, verbose=False)
    for result in results:
        jointsdata = result.keypoints.data.cpu().numpy()  # (1, 17, 3)
        keypoints, scores = jointsdata[:1, :, :2], jointsdata[:1, :, 2:]

        if scores.any():
            # 키포인트 처리 및 정규화
            h36m_kpts, h36m_scores, _ = h36m_coco_format(keypoints, scores)
            frame = show2Dpose(h36m_kpts, frame)  # 2D Pose 그리기

            norm2dhp = normalize2dhp(h36m_kpts, w=frame_size[1], h=frame_size[0])
            keypoints = extract_keypoints(np.concatenate([norm2dhp, h36m_scores], axis=2))

            # 시퀀스에 키포인트 추가
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                try:
                    input_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 15, 34)
                    predictions = model(input_seq)
                    res = predictions.detach().cpu().numpy()[0]
                    # select = np.argmax(res)  # 가장 높은 확률의 클래스
                    select = np.argmax(res)
                except:
                    sequence = []
                    continue
                
                
                #3. Viz logic
                if res[select] > threshold: 
                    if len(sentence) > 0: 
                        if actions[select] != sentence[-1]:
                            sentence.append(actions[select])
                    else:
                        sentence.append(actions[select])
                
                else:
                    sentence = []
                

                if len(sentence) > 1: 
                    sentence = sentence[-1:]

                # Viz probabilities
                frame = prob_viz(res, actions, frame)
                
                
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
                
    # 화면에 프레임 표시
    cv2.imshow('Action Recognition', frame)

    # 종료 조건
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


