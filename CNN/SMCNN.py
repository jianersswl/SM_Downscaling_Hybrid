# Pytorch
import torch.nn as nn

class SMCNN(nn.Module):
    def __init__(self, input_channel):
        super(SMCNN, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [18, 11, 11]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 1, 1),  # [32, 11, 11]
            nn.BatchNorm2d(32),
            nn.ReLU(),
#             nn.MaxPool2d(2, 1, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*11*11, 4),
            nn.ReLU(),
#             nn.Linear(4, 2),
#             nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = x.permute([0, 3, 1, 2])
        out = self.cnn(x)
        
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc(out)
        out = out.squeeze(1)
        return out