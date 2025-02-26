import torch
import torch.nn as nn
from config import BASE_CHANNELS, BOTTLENECK_SIZE, NUM_PLANES

H_ENC = 141
W_ENC = 250
class Conv2DAutoencoder(nn.Module):
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS, bottleneck_dim=BOTTLENECK_SIZE):
        """
        Args:
            in_channels (int): Number of input channels (2 for X and Y).
            base_channels (int): Base number of channels for the convolutional layers.
            bottleneck_dim (int): Dimensionality of the bottleneck (FC latent space).
            nbpms (int): Number of BPMs (image height).
            nturns (int): Number of turns (image width).
        """
        super(Conv2DAutoencoder, self).__init__()
        # ----------------------------
        # ENCODER: Use Conv2d layers with stride=2.
        # ----------------------------
        # First conv: from (B, 2, 563, 1000) to (B, base_channels, 282, 500)
        # Second conv: from (B, base_channels, 282, 500) to (B, base_channels*2, 141, 250)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # After encoder, the feature map is expected to have shape:
        #    (B, base_channels*2, H_enc, W_enc) where H_enc = 141, W_enc = 250.
        self.enc_shape = (base_channels * 2, H_ENC, W_ENC)
        flat_dim = base_channels * 2 * H_ENC * W_ENC

        # ----------------------------
        # BOTTLENECK (FC layers)
        # ----------------------------
        self.fc_enc = nn.Linear(flat_dim, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, flat_dim)

        # ----------------------------
        # DECODER: Use ConvTranspose2d layers to upsample.
        # ----------------------------
        # First deconv: from (B, base_channels*2, 141, 250) to (B, base_channels, 282, 500)
        # Second deconv: from (B, base_channels, 282, 500) to (B, in_channels, 563, 1000)
        # We carefully choose output_padding to exactly recover the original dimensions.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, 
                base_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels, 
                in_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(0, 1)
            ),
            # Note: No final activation; we want a linear output to cover the full range.
        )

    def forward(self, x):
        """
        Forward pass.
        Input x should have shape (B, 2, nbpms, nturns), e.g. (B, 2, 563, 1000).
        """
        # Encoder
        x_enc = self.encoder(x)  # -> (B, base_channels*2, 141, 250)
        B = x_enc.size(0)
        # Flatten the encoded features
        x_flat = x_enc.view(B, -1)
        # Bottleneck compression and expansion
        latent = self.fc_enc(x_flat)
        x_dec_flat = self.fc_dec(latent)
        # Reshape back to the convolutional feature map shape
        x_dec = x_dec_flat.view(B, *self.enc_shape)
        # Decoder to reconstruct the image
        x_out = self.decoder(x_dec)
        return x_out
    
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
class SineConv2DAutoencoder(nn.Module):
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS, latent_dim=BOTTLENECK_SIZE):
        """
        A 2D convolutional autoencoder using sine activations.
        Args:
            in_channels (int): Number of input channels (2 for X and Y).
            base_channels (int): Base number of channels for the conv layers.
            latent_dim (int): Dimension of the bottleneck.
            nbpms (int): Number of BPMs (image height).
            nturns (int): Number of turns (image width).
        """
        super(SineConv2DAutoencoder, self).__init__()
        # ENCODER: Two conv layers downsample by roughly a factor of 4.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            Sine(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            Sine()
        )
        # Calculate the encoded feature-map size.
        # For nbpms=563, nturns=1000: H_enc = ceil(563/4)=141, W_enc = ceil(1000/4)=250.
        self.enc_shape = (base_channels * 2, H_ENC, W_ENC)
        flat_dim = base_channels * 2 * H_ENC * W_ENC
        
        # Fully connected bottleneck layers.
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        
        # DECODER: Use transposed convolutions to upsample.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, 
                base_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(1,1)
            ),
            Sine(),
            nn.ConvTranspose2d(
                base_channels, 
                in_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(0,1)
            )
            # Note: No final activation; we want a linear output to cover the full range.
        )

    def forward(self, x):
        """
        Forward pass.
        Input x should have shape (B, 2, nbpms, nturns), e.g. (B, 2, 563, 1000).
        """
        x_enc = self.encoder(x)          # -> (B, base_channels*2, 141, 250)
        B = x_enc.size(0)
        x_flat = x_enc.view(B, -1)
        latent = self.fc_enc(x_flat)
        x_dec_flat = self.fc_dec(latent)
        x_dec = x_dec_flat.view(B, *self.enc_shape)
        out = self.decoder(x_dec)
        return out


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
