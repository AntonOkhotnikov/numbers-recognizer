import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = torch.nn.BatchNorm2d(n_feats)
        self.layer_norm2 = torch.nn.BatchNorm2d(n_feats)

    def forward(self, x):
        residual = x  # (batch, filters, time, mfcc_dim)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, filters, time, mfcc_dim)


class DeepSpeechClone(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(DeepSpeechClone, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Sequential(*[
            nn.Conv2d(1, n_feats, kernel_size=(32, 1), dilation=(1, 1), stride=stride),
            nn.BatchNorm2d(n_feats), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv2d(n_feats, n_feats, kernel_size=(32, 1), dilation=(2, 1), stride=stride),
            nn.BatchNorm2d(n_feats), nn.ReLU(), nn.Dropout(dropout)
        ])

        # n residual cnn layers
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(n_feats, n_feats, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(13*n_feats, rnn_dim)
        self.birnn_layers = nn.LSTM(input_size=rnn_dim,  # number of expected features in the input x
                                    hidden_size=rnn_dim,  # number of features in the hidden state h
                                    num_layers=n_rnn_layers,
                                    dropout=dropout, 
                                    batch_first=True,
                                    bidirectional=True)
        self.classifier = nn.Sequential(
            # nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.Linear(rnn_dim, rnn_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim//2, n_class)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[3], sizes[2])
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)

        x, (hidden, cell) = self.birnn_layers(x)  # [batch, seq_len, 128] -> [batch, seq_len, 256]
        # since output is of dim 2, we can sum it
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)

        x = self.classifier(x)
        return x
