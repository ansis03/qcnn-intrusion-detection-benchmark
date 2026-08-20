"""Classical 비교 모델: CNN (1D), SVM"""

import torch
import torch.nn as nn
from sklearn.svm import SVC


class CNN1D(nn.Module):
    """1D CNN — QCNN과 비슷한 파라미터 수로 맞춤."""

    def __init__(self, input_dim: int = 6, n_filters: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(n_filters * (input_dim // 2), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, input_dim) → unsqueeze → (batch, 1, input_dim)
        return self.net(x.unsqueeze(1)).squeeze(-1)


def build_svm():
    return SVC(kernel='rbf', C=1.0, random_state=42)
