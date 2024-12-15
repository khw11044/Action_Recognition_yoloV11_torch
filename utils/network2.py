import torch
import torch.nn as nn
from torch.autograd import Variable

# ResBlock 클래스
class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x  # Skip connection
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x += residual  # Residual connection
        return x

# LSTM 네트워크 클래스
class LSTMWithResBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, lstm_layers=2, bidirectional=True, dropout=0.5):
        super(LSTMWithResBlock, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # ResBlock을 통해 업스케일링
        self.upscale = nn.Linear(input_size, 1024)
        self.res_block = ResBlock(1024, 1024)

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True, 
                            bidirectional=bidirectional)

        # Fully Connected Layer (Output)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

        # Dropout 설정
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        

    def init_hidden(self, batch_size):
        # LSTM의 초기 hidden state와 cell state 설정
        num_directions = 2 if self.bidirectional else 1
        return (Variable(torch.zeros(self.lstm_layers * num_directions, batch_size, self.hidden_size).cuda()),
                Variable(torch.zeros(self.lstm_layers * num_directions, batch_size, self.hidden_size).cuda()))

    def forward(self, x):
        B, T, F = x.shape  # Batch, Time, Features
        
        # ResBlock을 통과하기 위해 각 타임스텝의 피처를 업스케일링
        x = x.view(B * T, F)  # (B*T, F)
        x = self.upscale(x)   # (B*T, 1024)
        x = self.res_block(x) # ResBlock 통과
        x = x.view(B, T, -1)  # 다시 (B, T, 1024)로 reshape

        # LSTM Forward
        hidden = self.init_hidden(B)
        x, _ = self.lstm(x, hidden)

        # LSTM의 마지막 타임스텝만 사용
        x = x[:, -1, :]  # 마지막 타임스텝 출력 (B, hidden_size * num_directions)

        # Fully Connected Layer를 통과
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)

        return self.softmax(x)
