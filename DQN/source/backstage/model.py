

import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()

        filters = 64
        ch = [1, 2, 4, 4]

        self.conv = nn.Sequential(
            self.convBlock(c,             filters*ch[0], k=5, s=2, p=2),
            self.convBlock(filters*ch[0], filters*ch[1], k=5, s=2, p=2),
            self.convBlock(filters*ch[1], filters*ch[2], k=3, s=2, p=1),
            self.convBlock(filters*ch[2], filters*ch[3], k=3, s=2, p=1)
        )

        features = (w // 16) * (h // 16) * filters*ch[3]
        hidden = 256

        self.head = nn.Sequential(
            nn.Linear(features, hidden, bias=False),
            nn.BatchNorm1d(hidden, momentum=0.1),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, outputs)
        )

    def convBlock(self, inch, outch, k, s, p):
        layers = []
        layers.append(nn.Conv2d(inch, outch, kernel_size=k, stride=s, padding=p, bias=False))
        layers.append(nn.BatchNorm2d(outch, momentum=0.1))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def forward(self, x):

        # convert to floats!
        x = x.float() / 255.0

        x = self.conv(x)
        x = x.view( x.size(0), -1) 
        x = self.head(x)
        return x


