import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # first conv layer
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x6 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)

        # dropout with p=0.1
        self.drop1 = nn.Dropout(p=0.1)

        # second conv layer: 32 inputs, 64 outputs, 4x4 conv
        ## output size = (W-F)/S +1 = (110-4)/1 +1 = 107
        # the output tensor will have dimensions: (64, 107, 107)
        # after another pool layer this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 4)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)

        # dropout with p=0.2
        self.drop2 = nn.Dropout(p=0.2)

        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output tensor will have dimensions: (128, 51, 51)
        # after another pool layer this becomes (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)

        # dropout with p=0.3
        self.drop3 = nn.Dropout(p=0.3)

        # 4th conv layer: 128 inputs, 256 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (25-2)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 2)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2)

        # dropout with p=0.4
        self.drop4 = nn.Dropout(p=0.4)

        # 5th conv layer: 256 inputs, 512 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (12-1)/1 +1 = 12
        # the output tensor will have dimensions: (512, 12, 12)
        # after another pool layer this becomes (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, 1)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool5 = nn.MaxPool2d(2, 2)

        # dropout with p=0.5
        self.drop5 = nn.Dropout(p=0.5)

        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1000)

        # dropout with p=0.6
        self.fc1_drop = nn.Dropout(p=0.6)

        # 1000 inputs, 1000 outputs
        self.fc2 = nn.Linear(1000, 1000)

        # dropout with p=0.7
        self.fc2_drop = nn.Dropout(p=0.7)

        # 1000 inputs, 136 outputs
        self.fc3 = nn.Linear(1000, 136)


    def forward(self, x):
        ## Define the feedforward behavior of this model
        # Convolutional layers
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        x = self.drop5(self.pool5(F.elu(self.conv5(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.elu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        # final output
        return x
