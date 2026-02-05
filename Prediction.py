import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import muspy
from sklearn.preprocessing import MinMaxScaler

# --- READ MIDI FILE AND EXTRACT MONOPHONIC DATA ---

# music = muspy.read_midi("rondo_alla_turca.mid")
track = music.tracks[0]
notes = sorted(track.notes, key=lambda n: n.time)

monophonic_notes = []
current_time = -1
for note in notes:
    if note.time != current_time:
        monophonic_notes.append(note.pitch)
        current_time = note.time

# --- DATA PREPROCESSING: NORMALIZATION ---

scaler = MinMaxScaler()
scaled_notes = scaler.fit_transform(np.array(monophonic_notes).reshape(-1, 1)).flatten()

# --- DATASET CLASS ---

class MelodyDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- HYPERPARAMETERS ---

sequence_length = 20
batch_size = 64
hidden_size = 128
num_layers = 2
epochs = 50
learning_rate = 0.001

# --- LOAD DATA ---

dataset = MelodyDataset(scaled_notes, sequence_length)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- LSTM MODEL DEFINITION ---

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # Output shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :]            # Take the output of the final time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# --- INITIALIZE AND TRAIN MODEL ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
