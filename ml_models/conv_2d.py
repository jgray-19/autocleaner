import torch
import torch.nn as nn
from config import BASE_CHANNELS, BOTTLENECK_SIZE, NUM_PLANES, INIT

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

class Conv2DAutoencoderLeaky(nn.Module):
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS, bottleneck_dim=BOTTLENECK_SIZE):
        """
        A 2D convolutional autoencoder with LeakyReLU activations in hidden layers.
        
        Args:
            in_channels (int): Number of input channels (2 for X and Y).
            base_channels (int): Base number of channels for the convolutional layers.
            bottleneck_dim (int): Dimensionality of the bottleneck (latent space).
            nbpms (int): Number of BPMs (image height).
            nturns (int): Number of turns (image width).
        """
        super(Conv2DAutoencoderLeaky, self).__init__()
        
        # ----------------------------
        # ENCODER
        # ----------------------------
        # First Conv2d: downsample by a factor of 2.
        # Second Conv2d: further downsample by a factor of 2.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        # After two stride=2 layers:
        #   Height: 563 -> ceil(563/2)=282 -> ceil(282/2)=141
        #   Width:  1000 -> 500 -> 250
        self.enc_shape = (base_channels * 2, H_ENC, W_ENC)
        flat_dim = base_channels * 2 * H_ENC * W_ENC
        
        # ----------------------------
        # BOTTLENECK
        # ----------------------------
        self.fc_enc = nn.Linear(flat_dim, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, flat_dim)
        
        # ----------------------------
        # DECODER
        # ----------------------------
        # Use ConvTranspose2d layers to upsample. Output paddings are chosen to match the original dimensions.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, 
                base_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(1,1)
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(
                base_channels, 
                in_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=(0,1)
            )
        )
        
        # Optionally, initialize weights here.
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if INIT == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif INIT == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            else:
                raise ValueError(f"Unknown init type: {INIT}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, nbpms, nturns)
        Returns:
            Reconstructed tensor with the same shape.
        """
        # Encoder
        x_enc = self.encoder(x)  # Shape: (B, base_channels*2, 141, 250)
        B = x_enc.size(0)
        # Flatten for fully connected bottleneck
        x_flat = x_enc.view(B, -1)
        latent = self.fc_enc(x_flat)
        x_dec_flat = self.fc_dec(latent)
        # Reshape back to conv feature map dimensions
        x_dec = x_dec_flat.view(B, *self.enc_shape)
        # Decoder: Upsample back to original dimensions
        x_out = self.decoder(x_dec)
        return x_out

class Conv2DAutoencoderLeakyNoFC(nn.Module):
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS):
        """
        A fully convolutional autoencoder with LeakyReLU activations,
        similar to Conv2DAutoencoderLeaky but without a fully connected bottleneck.
        
        Args:
            in_channels (int): Number of input channels (e.g., 2 for X and Y).
            base_channels (int): Base number of channels for the conv layers.
        """
        super(Conv2DAutoencoderLeakyNoFC, self).__init__()
        
        # ----------------------------
        # ENCODER: Convolutional downsampling.
        # ----------------------------
        self.encoder = nn.Sequential(
            # First conv: Downsample by factor 2.
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Second conv: Downsample by factor 2.
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        # After two downsamplings, for an input of shape (B, in_channels, 563, 1000),
        # the feature map will be (B, base_channels*2, 141, 250) (approximately).
        
        # ----------------------------
        # DECODER: Convolutional upsampling.
        # ----------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 1)
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1)
            )
            # No final activation; keeping the output linear allows full-range reconstruction.
        )
        
        # Initialize weights (using your chosen initialization scheme)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if INIT == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif INIT == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            else:
                raise ValueError(f"Unknown init type: {INIT}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Encode input to a spatial latent representation.
        x_enc = self.encoder(x)
        # Decode back to the original dimensions.
        x_out = self.decoder(x_enc)
        return x_out

class FourierLayer(nn.Module):
    """
    A Fourier layer that transforms input features to the frequency domain,
    applies a learnable complex multiplication to a selected set of low-frequency modes,
    and then transforms back to the spatial domain.
    """
    def __init__(self, in_channels, out_channels, modes_height=20, modes_width=20):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes_height (int): Number of Fourier modes to keep along the height dimension.
            modes_width (int): Number of Fourier modes to keep along the width dimension.
        """
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width

        # Initialize a learnable complex weight for the lower-frequency modes.
        # Shape: (in_channels, out_channels, modes_height, modes_width)
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        # Compute 2D Fourier transform (real FFT) along the spatial dimensions.
        x_ft = torch.fft.rfft2(x, norm="ortho")  # shape: (B, C, H, W//2 + 1)

        # Prepare an output Fourier tensor with shape (B, out_channels, H, W//2+1)
        out_ft = torch.zeros(B, self.out_channels, H, x_ft.shape[-1],
                               device=x.device, dtype=torch.cfloat)
        
        # Limit the number of modes to process.
        mh = min(self.modes_height, H)
        mw = min(self.modes_width, x_ft.shape[-1])
        
        # Correct einsum: Multiply x_ft slice of shape (B, in_channels, mh, mw)
        # with weight of shape (in_channels, out_channels, mh, mw) to produce (B, out_channels, mh, mw).
        out_ft[:, :, :mh, :mw] = torch.einsum("bcij,cdij->bdij",
                                               x_ft[:, :, :mh, :mw],
                                               self.weight[:, :, :mh, :mw])
        
        # Inverse FFT to convert back to spatial domain.
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x_out

class Conv2DAutoencoderLeakyFourier(nn.Module):
    """
    A fully convolutional autoencoder that integrates a Fourier layer in the latent space.
    The encoder downsamples the input, the Fourier layer processes the latent features,
    and the decoder upsamples back to the original dimensions.
    """
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS,
                 modes_height=20, modes_width=20):
        super(Conv2DAutoencoderLeakyFourier, self).__init__()
        
        # Encoder: Two convolutional layers with downsampling.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # Fourier layer applied to the latent representation.
        self.fourier = FourierLayer(in_channels=base_channels * 2,
                                    out_channels=base_channels * 2,
                                    modes_height=modes_height,
                                    modes_width=modes_width)
        
        # Decoder: Two transposed convolutional layers to upsample.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 1)
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1)
            )
            # No final activation to allow a linear output range.
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if INIT == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif INIT == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            else:
                raise ValueError(f"Unknown init type: {INIT}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x_enc = self.encoder(x)
        x_fourier = self.fourier(x_enc)
        x_out = self.decoder(x_fourier)
        return x_out

class DeepConvAutoencoder(nn.Module):
    def __init__(self, in_channels=NUM_PLANES, base_channels=BASE_CHANNELS):
        """
        A deep, fully convolutional autoencoder.
        This architecture uses four convolutional layers to downsample
        and four transposed convolutional layers to upsample.
        
        With base_channels=64, the encoder will have roughly 1.55M parameters,
        so the total model (encoder + decoder) is well above 1M parameters.
        
        Args:
            in_channels (int): Number of input channels (e.g., 2 for X and Y).
            base_channels (int): Base number of channels for the first conv layer.
        """
        super(DeepConvAutoencoder, self).__init__()
        # Encoder: 4 layers downsampling by factor of 2 each.
        # Input shape: (B, 2, 563, 1000)
        # After Layer 1: (B, 64, ceil(563/2)=282, ceil(1000/2)=500)
        # After Layer 2: (B, 128, ceil(282/2)=141, ceil(500/2)=250)
        # After Layer 3: (B, 256, ceil(141/2)=71, ceil(250/2)=125)
        # After Layer 4: (B, 512, ceil(71/2)=36, ceil(125/2)=63)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Decoder: 4 layers upsampling to recover the original dimensions.
        # We choose output_padding carefully so that:
        #   - Height: 36 -> 71 -> 141 -> 282 -> 563
        #   - Width:  63 -> 125 -> 250 -> 500 -> 1000
        self.decoder = nn.Sequential(
            # Decoder Layer 1: (B,512,36,63) -> (B,256,71,125)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                               kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.01, inplace=True),

            # Decoder Layer 2: (B,256,71,125) -> (B,128,141,250)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.01, inplace=True),

            # Decoder Layer 3: (B,128,141,250) -> (B,64,282,500)
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.01, inplace=True),

            # Decoder Layer 4: (B,64,282,500) -> (B,2,563,1000)
            nn.ConvTranspose2d(base_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=(0, 1))
            # No final activation, to allow a linear output range.
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if INIT == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif INIT == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            else:
                raise ValueError(f"Unknown init type: {INIT}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_out = self.decoder(x_enc)
        return x_out