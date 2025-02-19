import torch
import torch.nn as nn
import torch.nn.functional as F

COMPRESSED_LENGTH = 125  # Compressed length for the bottleneck layer (selected for 1000 turns)

class ImprovedConvAutoencoder(nn.Module):
    def __init__(self, input_channels, bottleneck_size=16):  # Default 16 - Seems to be best
        super(ImprovedConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1, dtype=torch.float32),
            nn.BatchNorm1d(32, dtype=torch.float32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, dtype=torch.float32),
            nn.BatchNorm1d(64, dtype=torch.float32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, dtype=torch.float32),
            nn.BatchNorm1d(128, dtype=torch.float32),
            nn.LeakyReLU(0.01)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(128 * COMPRESSED_LENGTH, bottleneck_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(bottleneck_size, 128 * COMPRESSED_LENGTH, dtype=torch.float32)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.float32),
            nn.BatchNorm1d(64, dtype=torch.float32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.float32),
            nn.BatchNorm1d(32, dtype=torch.float32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.float32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x.view(x.size(0), -1)).view(x.size(0), 128, -1)
        x = self.decoder(x)
        return x

class SimpleConvAutoencoder(nn.Module):
    def __init__(self, input_channels=563, latent_channels=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv1d(32, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), 
        )

    def forward(self, x):
        x = self.encoder(x)  # shape => (B, latent_channels, compressed_length)
        x = self.decoder(x)  # shape => (B, input_channels, original_length)
        return x
    

class ChannelLayerNorm(nn.Module):
    """
    Applies Layer Normalization over the channel dimension for 1D data.
    Input shape: (B, C, L)
    """
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # Transpose so that channels are the last dimension, apply LayerNorm, then transpose back.
        x = x.transpose(1, 2)
        x = self.ln(x)
        return x.transpose(1, 2)

class SimpleSkipAutoencoder(nn.Module):
    def __init__(self, input_channels, base_channels=32):
        """
        A fully convolutional autoencoder using dilated convolutions, LayerNorm, and skip connections.
        
        Args:
            input_channels (int): Number of input channels.
            base_channels (int): Number of channels for the first convolution.
        """
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.ln1 = ChannelLayerNorm(base_channels)
        
        # Downsample with a stride-2 convolution
        self.enc2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.ln2 = ChannelLayerNorm(base_channels * 2)
        
        # Further downsample using a dilated convolution (dilation=2) to capture a wider context without extra parameters
        self.enc3 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=2, dilation=2)
        self.ln3 = ChannelLayerNorm(base_channels * 2)
        
        # Decoder
        self.dec1 = nn.ConvTranspose1d(base_channels * 2, base_channels * 2, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)
        self.ln_d1 = ChannelLayerNorm(base_channels * 2)
        
        self.dec2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)
        self.ln_d2 = ChannelLayerNorm(base_channels)
        
        self.dec3 = nn.Conv1d(base_channels, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder path
        e1 = F.leaky_relu(self.ln1(self.enc1(x)), negative_slope=0.01)    # shape: (B, base_channels, L)
        e2 = F.leaky_relu(self.ln2(self.enc2(e1)), negative_slope=0.01)     # shape: (B, base_channels*2, L/2)
        e3 = F.leaky_relu(self.ln3(self.enc3(e2)), negative_slope=0.01)     # shape: (B, base_channels*2, L/4)
        
        # Decoder path with skip connections
        d1 = F.leaky_relu(self.ln_d1(self.dec1(e3)), negative_slope=0.01)   # shape: (B, base_channels*2, L/2)
        d1 = d1 + e2  # Skip connection from e2
        
        d2 = F.leaky_relu(self.ln_d2(self.dec2(d1)), negative_slope=0.01)   # shape: (B, base_channels, L)
        d2 = d2 + e1  # Skip connection from e1
        
        d3 = torch.tanh(self.dec3(d2))  # Final output in [-1, 1]
        return d3