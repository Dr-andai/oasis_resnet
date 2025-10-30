import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Basic Residual Block
# ---------------------------
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, 
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------------
# ResNet3D
# ---------------------------
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1, in_channels=1):
        super().__init__()

        # Track channels (important for later layers)
        self.in_channels = 64

        # First layer
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def extract_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

    # def forward(self, x):
    #     # Input: [B, C, D, H, W]
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.maxpool(x)

    #     x = self.layer1(x)   # 64
    #     x = self.layer2(x)   # 128
    #     x = self.layer3(x)   # 256
    #     x = self.layer4(x)   # 512

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
        return x


def resnet18_3d(in_channels=1):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2],in_channels=in_channels)

class MRI_AgeModel(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        self.backbone = backbone or resnet18_3d(in_channels=1)

        # 512 features from CNN + 1 diagnosis scalar
        self.fc_age = nn.Sequential(
            nn.Linear(512 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, volume, diagnosis):
        feats = self.backbone.extract_features(volume)  # [B,512]
        diagnosis = diagnosis.float().unsqueeze(1)      # [B,1]

        combined = torch.cat([feats, diagnosis], dim=1) # [B,513]
        age_pred = self.fc_age(combined)                # [B,1]
        return age_pred
