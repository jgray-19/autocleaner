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

class ConvFC_Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int,   # e.g. number of BPMs if you treat each BPM as a channel
        input_length: int,     # e.g. number of turns
        bottleneck_dim: int=64 # fully connected bottleneck size
    ):
        super().__init__()

        # -------------------------
        # 1) ENCODER
        # -------------------------
        # Two convolutional layers, each halving the time dimension (stride=2).
        # If you prefer dividing by 4 in one layer, just set stride=4 or add more layers.
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # After two stride=2 layers, the time dimension is divided by 4:
        #   new_length = input_length // 2 // 2 = input_length // 4
        reduced_length = input_length // 4
        # Flatten dimension = (#channels after last conv) * reduced_length
        self.flat_dim = 64 * reduced_length

        # -------------------------
        # 2) BOTTLENECK (Fully Connected)
        # -------------------------
        self.fc_enc = nn.Linear(self.flat_dim, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, self.flat_dim)

        # -------------------------
        # 3) DECODER
        # -------------------------
        # We unflatten from (batch, flat_dim) to (batch, 64, reduced_length)
        # Then use ConvTranspose1d to upsample by factors of 2, reversing the encoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, input_channels, kernel_size=4, stride=2, padding=1),
            # If your data is strictly within [-1,1], you can add nn.Tanh() here;
            # for unbounded (normal) data, leaving it linear is often fine.
        )

        # Apply Xavier uniform initialization to all Conv/Linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, input_length)
        Returns reconstructed output with same shape.
        """
        # Encoder
        x = self.encoder(x)
        # Flatten for the fully connected bottleneck
        x = x.view(x.size(0), -1)
        # FC bottleneck
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        # Unflatten
        batch_size = x.size(0)
        # shape => (batch_size, 64, input_length//4)
        x = x.view(batch_size, 64, -1)
        # Decoder
        x = self.decoder(x)
        return x
