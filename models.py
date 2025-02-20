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
    
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)