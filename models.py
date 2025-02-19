import torch.nn as nn
import torch

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


class FrequencyPreservingAutoencoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32):
        """
        Args:
            input_channels (int): Number of input channels (e.g. number of BPMs, or 1 if each signal is separate).
            base_channels (int): Base number of channels for the convolutional layers.
        """
        super(FrequencyPreservingAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.01)
        )  # output: (base_channels, T)
        
        # Downsample by factor 2
        self.pool1 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.01)
        )  # output: (base_channels*2, T/2)
        
        self.pool2 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Sequential(
            # Using dilation can help capture longer-range oscillatory patterns
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.Tanh()  # Using Tanh here to encourage a smooth, symmetric representation
        )  # output: (base_channels*4, T/4)
        
        # Bottleneck: you can add additional convolutional layers if desired.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.Tanh(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.Tanh()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.01)
        )
        
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.01)
        )
        
        # Final output layer: use Tanh so that the output is in [-1, 1]
        self.out_conv = nn.Conv1d(base_channels, input_channels, kernel_size=3, padding=1)
        self.out_activation = nn.Tanh()
        
    def forward(self, x):
        # x shape: (batch, channels, T)
        # Encoder
        e1 = self.enc1(x)         # e1: (B, base_channels, T)
        p1 = self.pool1(e1)       # p1: (B, base_channels, T/2)
        
        e2 = self.enc2(p1)        # e2: (B, base_channels*2, T/2)
        p2 = self.pool2(e2)       # p2: (B, base_channels*2, T/2) with downsampling -> (B, base_channels*2, T/4)
        
        e3 = self.enc3(p2)        # e3: (B, base_channels*4, T/4)
        
        # Bottleneck
        b = self.bottleneck(e3)   # b: (B, base_channels*4, T/4)
        
        # Decoder
        up1 = self.up1(b)         # up1: (B, base_channels*2, T/2)
        # Skip connection from e2: concatenate along the channel dimension
        cat1 = torch.cat([up1, e2], dim=1)  # (B, base_channels*4, T/2)
        d1 = self.dec1(cat1)      # (B, base_channels*2, T/2)
        
        up2 = self.up2(d1)        # (B, base_channels, T)
        # Skip connection from e1
        cat2 = torch.cat([up2, e1], dim=1)  # (B, base_channels*2, T)
        d2 = self.dec2(cat2)      # (B, base_channels, T)
        
        out = self.out_conv(d2)   # (B, input_channels, T)
        out = self.out_activation(out)  # Ensures output in [-1, 1]
        return out

# Example usage:
# model = FrequencyPreservingAutoencoder(input_channels=1, base_channels=32)
# x = torch.randn(8, 1, 1000)  # batch of 8 samples, 1 channel, 1000 timesteps
# output = model(x)
