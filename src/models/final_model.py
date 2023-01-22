from src.logger import logging
from src.exceptions import CustomException
from src.models.base_model import ImageClassificationBase
import torch.nn as nn

class CNNNetwork(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 16, pool=True) # 16 x 30 x 20
        self.conv2 = self.conv_block(16, 32, pool=True) # 32 x 15 x 10
        self.conv3 = self.conv_block(32, 64, pool=True) # 64 x 7 x 5
        self.conv4 = self.conv_block(64, 128, pool=True) # 128 x 4 x 3
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(128 * 5 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)
  
    @staticmethod
    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2), 
                  nn.BatchNorm2d(out_channels), 
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers) 

    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten(out)
        out = self.dropout(out)
        logits = self.linear(out)
        predictions = self.softmax(logits)
        return predictions