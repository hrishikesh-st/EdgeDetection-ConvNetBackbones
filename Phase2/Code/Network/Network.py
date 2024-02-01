"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)          # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        acc = accuracy(out, labels)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)            # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)   # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))


# Baseline Model
class BaselineNet(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        print("Initializing BaselineNet")

        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)     # output --> 16 @ 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)    # output --> 32 @ 32 x 32
        self.pool = nn.MaxPool2d(2, 2)                  # output --> 32 @ 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    # output --> 64 @ 16 x 16
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)   # output --> 128 @ 16 x 16
        self.pool = nn.MaxPool2d(2, 2)                  # output --> 128 @ 8 x 8

        self.fc1 = nn.Linear(8 * 8 * 128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, OutputSize)



    def forward(self, out):
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool(out)

        out = out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


# BatchNorm Model
class BatchNormNet(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        print("Initializing BatchNormNet")

        super(BatchNormNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Norm for conv1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Norm for conv2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch Norm for conv3
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) # Batch Norm for conv4

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 8 * 128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, OutputSize)


    def forward(self, out):
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)

        out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


# ResNet Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.match_dimensions = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.match_dimensions = None

    def forward(self, x):
        residual = x
        if self.match_dimensions:
            residual = self.match_dimensions(residual)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        print("Initializing ResNet")

        super(ResNet, self).__init__()
        self.input_size = InputSize
        self.output_size = OutputSize

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.resblock3 = ResidualBlock(128, 128)
        self.resblock4 = ResidualBlock(128, 256, stride=2)
        self.resblock5 = ResidualBlock(256, 256)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, OutputSize)
    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

# ResNeXt Model
class ResNextResidualBlock(nn.Module):
    def __init__(self, in_channles, out_channels, stride, first=False, cardinality=16):
        super(ResNextResidualBlock, self).__init__()

        C = cardinality
        intermediate_channels = out_channels // 2
        self.downsample = stride == 2 or first

        self.conv1 = nn.Conv2d(in_channles, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, groups=C)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if self.downsample:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channles, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.projection(residual)

        out += residual
        out = self.relu(out)

        return out

class ResNeXT(ImageClassificationBase):
    
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        print("Initializing ResNeXT")

        super(ResNeXT, self).__init__()
        self.InputSize = InputSize
        self.OutputSize = OutputSize

        out_features = [256, 512, 1024, 2048]
        num_blocks = [3, 3, 3, 3]


        self.blocks = nn.ModuleList([ResNextResidualBlock(64, 256, 1, True)])


        for i in range(len(out_features)):
            if i > 0:
                print(f"Call --> {i}")
                self.blocks.append(ResNextResidualBlock(out_features[i-1], out_features[i], 2))
            for _ in range(num_blocks[i] - 1):
                self.blocks.append(ResNextResidualBlock(out_features[i], out_features[i], 1))

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, OutputSize)

        self.relu = nn.ReLU()
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        for block in self.blocks:
            out = block(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# DenseNet Model
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        inter_channels = 4 * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        print("Initializing DenseNet")

        super(DenseNet, self).__init__()
        self.InputSize = InputSize
        self.OutputSize = OutputSize

        growth_rate = 16
        block_config = [6, 12, 24, 16]
        num_init_features = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)

        features = []

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            features.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                features.append(trans)
                num_features = num_features // 2

        self.features = nn.Sequential(*features)
        self.bn2 = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, OutputSize)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.features(out)
        out = F.relu(self.bn2(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
