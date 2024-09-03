
from torch import nn
from torch.nn import functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Calculate attention scores
        attention_scores = self.conv1(x)
        attention_scores = F.softmax(attention_scores, dim=2)
        
        # Apply attention to input features
        attended_features = x * attention_scores
        
        return attended_features

class DCT_Attention_Fusion_Conv(nn.Module):
    def __init__(self, channels):
        super(DCT_Attention_Fusion_Conv, self).__init__()
        self.rgb_attention = SpatialAttention(channels)
        self.depth_attention = SpatialAttention(channels)
        self.rgb_pooling = nn.AdaptiveAvgPool2d(1)
        self.depth_pooling = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, rgb_features, DCT_features):
        # Spatial attention for both modalities
        rgb_attended_features = self.rgb_attention(rgb_features)
        depth_attended_features = self.depth_attention(DCT_features)
        
        # Adaptive pooling for both modalities
        rgb_pooled = self.rgb_pooling(rgb_attended_features)
        depth_pooled = self.depth_pooling(depth_attended_features)
        
        # Upsample attended and pooled features to the original size
        rgb_upsampled = F.interpolate(rgb_pooled, size=rgb_features.size()[2:], mode='bilinear', align_corners=False)
        depth_upsampled = F.interpolate(depth_pooled, size=DCT_features.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate the upsampled features
        fused_features = F.relu(rgb_upsampled+depth_upsampled)
        # fused_features = fused_features.sum(dim=1)
        
        return fused_features
    