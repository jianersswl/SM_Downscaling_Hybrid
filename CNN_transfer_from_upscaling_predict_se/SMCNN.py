# Pytorch
import torch
import torch.nn as nn
import torch.nn.init as init

class SMCNN(nn.Module):
    def __init__(self, input_channel):
        super(SMCNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # input 維度 [20, 11, 11]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1),  # [32, 11, 11]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),  # [32, 11, 11]
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*11*11, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
#             nn.ReLU()
        )
     
    def forward(self, x):
        x = x.permute([0, 3, 1, 2])
        out = self.cnn(x)
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc(out)
#         print(out)
#         nn.ReLU(out[:, 0])
        # 将a的值转换为非负值
#         out[:, 0] = torch.clamp(out[:, 0], min=0)
#         print(out)
        return out