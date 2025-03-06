import torch
import torch.nn as nn
import torch.fft
from config import NUM_PLANES, BASE_CHANNELS, MODEL_DEPTH

class SpectralConv2d(nn.Module):
    """
    A spectral convolution layer that transforms the input to the frequency domain,
    applies a learned complex multiplication to the lower-frequency modes, and then
    transforms back to the spatial domain.
    
    This implementation is inspired by Li et al. (2020) (https://github.com/zongyi-li/fourier_neural_operator) :contentReference[oaicite:1]{index=1}.
    """
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height  # Number of Fourier modes to keep along height.
        self.modes_width = modes_width    # Number of Fourier modes to keep along width.

        # Initialize weights for the low-frequency modes.
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # Performs complex multiplication using Einstein summation.
        # Input shape: (batch, in_channels, height, width_ft) and weights shape: (in_channels, out_channels, modes_height, modes_width)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, _, height, width = x.shape
        # Compute 2D Fourier transform. The result has shape (batch, in_channels, height, width_ft)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        # Create an output tensor in the Fourier domain.
        out_ft = torch.zeros(batchsize, self.out_channels, height, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        # Multiply only the lower frequency modes.
        out_ft[:, :, :self.modes_height, :self.modes_width] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_height, :self.modes_width], self.weights)
        
        # Return to spatial domain.
        x_out = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x_out

class FNO2d(nn.Module):
    """
    A Fourier Neural Operator model for 2D data.
    
    Architecture:
      • Lifting layer (1x1 convolution) increases the channel dimension from in_channels (e.g. 2 for X and Y) to a higher "width".
      • A sequence of spectral convolution blocks (each combining a SpectralConv2d with a pointwise convolution) extracts frequency-domain features.
      • Two final 1x1 convolutions project the features back to out_channels.
    
    This model is inspired by Li et al. (2020) (https://github.com/zongyi-li/fourier_neural_operator) :contentReference[oaicite:2]{index=2}.
    """
    def __init__(self, in_channels=NUM_PLANES, out_channels=NUM_PLANES, width=BASE_CHANNELS, modes_height=10, modes_width=10, depth=MODEL_DEPTH):
        super(FNO2d, self).__init__()
        self.width = width
        self.depth = depth

        # Lifting layer: Increase number of channels.
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)
        
        # Create a sequence of spectral conv blocks.
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes_height, modes_width) for _ in range(depth)
        ])
        self.w_convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)
        ])

        # Projection layers: Reduce channels back to out_channels.
        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_channels, height, width)
        Returns:
            Tensor of shape (batch, out_channels, height, width)
        """
        x = self.fc0(x)
        for spec_conv, w_conv in zip(self.spectral_convs, self.w_convs):
            # Apply spectral convolution and pointwise convolution.
            x1 = spec_conv(x)
            x2 = w_conv(x)
            x = x1 + x2
            x = torch.nn.functional.gelu(x)
            # x = torch.nn.functional.leaky_relu(x)
        x = self.fc1(x)
        # x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x