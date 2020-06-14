import torch
import torch.nn as nn
import torch.nn.functional as F


class NumbersRecognizer(nn.Module):

    def __init__(self, num_classes, enrich_target=True):
        """
        num_classes - number of recognized digits     
        """
        super(NumbersRecognizer, self).__init__()

        # add to output layer 1 dimesnsion for a blank symbol
        self.__num_classes = num_classes + 1

        # if * is used in labeling, add one more output neuron
        if enrich_target:
            self.__num_classes = num_classes + 1

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # # cnn5
            # nn.ZeroPad2d((2, 2, 8, 8)),
            # nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
            # nn.BatchNorm2d(64), nn.ReLU(),

            # # cnn6
            # nn.ZeroPad2d((2, 2, 16, 16)),
            # nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
            # nn.BatchNorm2d(64), nn.ReLU(),

            # # cnn7
            # nn.ZeroPad2d((2, 2, 32, 32)),
            # nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
            # nn.BatchNorm2d(64), nn.ReLU(),

            # # cnn8
            # nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
            # nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8*hp.audio.num_freq + hp.embedder.emb_dim,
            hp.model.lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2*hp.model.lstm_dim, )
        self.fc2 = nn.Linear(self.__num_classes)

    def forward(self, x):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # # dvec: [B, emb_dim]
        # dvec = dvec.unsqueeze(1)
        # dvec = dvec.repeat(1, x.size(1), 1)
        # # dvec: [B, T, emb_dim]

        # x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.nn.Sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
        residual = x  # (batch, filters, time, mfcc_dim) = [64, 32, 59, 13]
        x = self.layer_norm1(x)  # [64, 32, 59, 13] -> [64, 32, 59, 13]
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)  # [64, 32, 59, 13] -> [64, 32, 59, 13]
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, filters, time, mfcc_dim)


class DeepSpeechClone(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(DeepSpeechClone, self).__init__()
        n_feats = n_feats//2
        # self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # in_channels, out_channels, kernel_size=(3,3), stride=(2,2)
        self.cnn = nn.Sequential(*[
            nn.Conv2d(1, 32, kernel_size=(32, 1), dilation=(1, 1), stride=stride),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv2d(32, 32, kernel_size=(32, 1), dilation=(2, 1), stride=stride),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout)
        ])

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(13*32, rnn_dim)
        self.birnn_layers = nn.LSTM(input_size=rnn_dim,  # number of expected features in the input x
                                    hidden_size=rnn_dim,  # number of features in the hidden state h
                                    num_layers=n_rnn_layers,
                                    dropout=dropout, 
                                    batch_first=True,
                                    bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # [64, 388, 13] -> [64, 1, 388, 13]
        x = self.cnn(x)  # [64, 1, 388, 13] -> [64, 32, 59, 13]
        x = self.rescnn_layers(x)  # [64, 32, 59, 13]
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[3], sizes[2])  # (batch, feature, time) -> [64, 416, 59]
        x = x.transpose(1, 2)  # (batch, time, feature) -> [64, 59, 416]
        x = self.fully_connected(x)  # [64, 59, 416] -> [64, 59, 128]
        x, (hidden, cell) = self.birnn_layers(x)  # [64, 59, 128] -> [64, 59, 256]
        x = self.classifier(x)  # [64, 59, 256] -> [64, 59, 12]
        return x
