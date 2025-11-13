import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. define a basic residual block
It solves the vanishing gradient problem, where gradients become extremely small as they are backpropagated through layers,
causing the early layers to learn very slowly or stop learning entirely.

Once the deeper models start converging (after accounting vanishing/exploading gradient) we possibly have another problem. It is degradation.
Optimisation becomes very difficult as the depth of the network increases.

To solve the problem of degradation, we have a setup called skip-connections.
It skips some of the layers in the architecture and feeds the output of the previous layer to the current position, thus helping in optimization.

A residual (difference between expected output and input). The stacked layers try to learn the mapping for the residual.
"""
# Basic Residual Block
"""
The residual block has two 3x3 convolutional layers with the same number of output channels. 
Each convolutional layer is followed by a batch normalization layer and a ReLU activation function.
Then, we skip these two convolution operations and add the input directly before the final ReLU activation function.
"""
class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
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

# ResNet3D
class ResNet3D(nn.Module):
    """
    Designing for volumetric data
    """
    def __init__(self, block, layers, num_classes=1, in_channels=1):
        super().__init__()

        # Track channels (important for later layers)
        self.in_channels = 64

        # First layer
        """
        Designing for volumetric data
        - Conv3d: scanning the 3d volume with 7x7x7
        - stride=2: downsampling
        - padding=3: accounting for edges
        - BatchNorm: normalizes activations so training stays stable
        """
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        """
        The stride=2 every few layers means the network is zooming out — looking at the image at coarser scales while increasing channel depth (64 → 512).
        """
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        This function builds a stack of blocks and handles shortcuts when needed
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        return x


def resnet18_3d(in_channels=1):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2],in_channels=in_channels)

"""
Let the CNN understand how the brain looks, and let the MLP combine that with what we already know about the patient to make a more accurate prediction.
"""
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
