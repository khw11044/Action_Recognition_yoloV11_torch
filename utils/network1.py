import torch
import torch.nn as nn

class ActionBiLSTM(nn.Module):
    def __init__(self, num_classes, sequence_length, input_size):
        super(ActionBiLSTM, self).__init__()
        
        # Bidirectional LSTM Layer 1
        self.bilstm1 = nn.LSTM(input_size=input_size, hidden_size=128, 
                               num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(sequence_length)  # Normalize over sequence length

        # Bidirectional LSTM Layer 2
        self.bilstm2 = nn.LSTM(input_size=256, hidden_size=128, 
                               num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.batchnorm2 = nn.BatchNorm1d(sequence_length)

        # Bidirectional LSTM Layer 3
        self.bilstm3 = nn.LSTM(input_size=256, hidden_size=64, 
                               num_layers=1, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout_fc1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc2 = nn.Dropout(0.3)

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)

        # Activation Function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Bidirectional LSTM 1
        x, _ = self.bilstm1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        # Bidirectional LSTM 2
        x, _ = self.bilstm2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        # Bidirectional LSTM 3
        x, _ = self.bilstm3(x)
        x = self.dropout3(x)

        # 마지막 타임스텝의 출력을 사용
        x = x[:, -1, :]  # 마지막 타임 스텝에서의 결과만 사용

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)

        # Output Layer
        x = self.fc_out(x)
        return self.softmax(x)
