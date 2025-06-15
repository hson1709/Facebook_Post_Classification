import torch.nn as nn
import torch.nn.functional as F
import torch

class BiLSTMwithCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, embed_model, freeze_embeddings=True):
        super(BiLSTMwithCNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        if embed_model is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors.copy()))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Conv1D + BiLSTM layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(0.6)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)

        # First Conv1D and BiLSTM
        x = F.relu(self.conv1(x))  # (batch_size, 256, seq_length)
        #x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, 256)
        x, _ = self.lstm1(x)  # (batch_size, seq_length, 256)

        # Second Conv1D and BiLSTM
        x = x.permute(0, 2, 1)  # (batch_size, 256, seq_length)
        x = F.relu(self.conv2(x))  # (batch_size, 512, seq_length)
        #x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, 512)
        x, _ = self.lstm2(x)  # (batch_size, seq_length, 256)

        # GlobalAveragePooling
        x = torch.mean(x, dim=1)  # GlobalAveragePooling
        #avg_pool = torch.mean(x, dim=1) # GlobalAveragePooling
        #max_pool = torch.max(x, dim=1).values  # GlobalMaxPooling
        #x = torch.cat((avg_pool, max_pool), dim=1)  # Concatenate avg_pool and max_pool

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x