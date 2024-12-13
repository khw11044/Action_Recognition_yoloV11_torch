import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from utils.network2 import LSTMWithResBlock
import matplotlib.pyplot as plt

# Random Seed 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

##################################### 데이터 로드 #####################################

# Custom Dataset 클래스
class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Dataset')
actions = np.array(['nothing', 'ready', 'stop', 'emergency'])
num_classes = len(actions)
no_sequences = 100
sequence_length = 15

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)                     # X.shape -> (400, 15, 51)
y = np.eye(num_classes)[labels]             # y.shape -> (400, 4)
print()

# Dataset 및 DataLoader
dataset = ActionDataset(X, y)
train_size = int(0.8 * len(dataset))  # Train 80%, Validation 20%
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

################################# 모델 로드 ####################################

# 모델 초기화
input_size = 51     # (17*3)
hidden_size = 128
num_layers = 2
num_classes = 4
num_epochs = 500

model = LSTMWithResBlock(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

################################### 모델 학습 ####################################

# 훈련 및 검증 함수
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0
    train_loss_list, train_acc_list, val_acc_list = [], [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, correct = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == torch.max(y_batch, 1)[1]).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == torch.max(y_val, 1)[1]).sum().item()

        val_acc = correct / len(val_loader.dataset)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/best_kpcl.pth')

        # 기록 저장
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_loss_list, train_acc_list, val_acc_list

# 모델 훈련
train_loss, train_acc, val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 학습 곡선 그리기
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_curve.png')
plt.show()

# 테스트 데이터셋 평가 함수
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('./models/best_kpcl.pth'))
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == torch.max(y_test, 1)[1]).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

# 테스트 데이터셋 로드 및 평가
test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # 예시로 Validation 데이터 재사용
evaluate_model(model, test_loader)