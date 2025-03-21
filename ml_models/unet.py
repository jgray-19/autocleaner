import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BASE_CHANNELS, NUM_CHANNELS, MODEL_DEPTH
from torch.utils.checkpoint import checkpoint

class UNetAutoencoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, base_channels=BASE_CHANNELS, depth=MODEL_DEPTH):
        """
        A U-Net style autoencoder for oscillatory data.
        
        Design rationale:
          • Fully convolutional to preserve spatial (phase and frequency) structure.
          • Skip connections ensure fine details (critical for phase and oscillatory features)
            are preserved during the encoding-decoding process.
          • No fully connected bottleneck is used because it would mix spatial information,
            potentially destroying the local phase relationships.
          • The depth (number of down/upsampling blocks) controls the receptive field so that both
            local and global oscillatory patterns can be captured.
        
        Args:
            in_channels (int): Number of input channels (e.g. 2 for X and Y data).
            base_channels (int): Number of channels for the first encoder block.
            depth (int): Number of downsampling steps.
        """
        super(UNetAutoencoder, self).__init__()
        self.depth = depth

        # Encoder: Create a list of convolutional blocks.
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        curr_channels = base_channels
        for i in range(depth):
            self.encoders.append(self.conv_block(prev_channels, curr_channels))
            prev_channels = curr_channels
            curr_channels *= 2

        # Bottleneck (fully convolutional)
        self.bottleneck = self.conv_block(prev_channels, curr_channels)

        # Decoder: Each block upsamples and then concatenates with the corresponding encoder output.
        self.decoders = nn.ModuleList()
        for i in range(depth):
            curr_channels //= 2
            self.decoders.append(nn.ModuleDict({
                'up_conv': nn.ConvTranspose2d(curr_channels * 2, curr_channels, kernel_size=2, stride=2),
                'conv': self.conv_block(curr_channels * 2, curr_channels)
            }))

        # Final layer to bring the number of channels back to in_channels.
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """A double convolution block with BatchNorm and LeakyReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder path
        out = x
        for encoder in self.encoders:
            out = encoder(out)
            skip_connections.append(out)
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        # Bottleneck (no fully connected layers—preserving spatial layout)
        out = self.bottleneck(out)

        # Decoder path with skip connections
        for decoder in self.decoders:
            out = decoder['up_conv'](out)
            # Retrieve corresponding encoder output (skip connection)
            skip = skip_connections.pop()
            # In case dimensions do not match exactly (due to rounding), we pad accordingly.
            if out.size()[2:] != skip.size()[2:]:
                diffY = skip.size(2) - out.size(2)
                diffX = skip.size(3) - out.size(3)
                out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
            # Concatenate along channel dimension
            out = torch.cat([skip, out], dim=1)
            out = decoder['conv'](out)

        # Final output convolution
        out = self.final_conv(out)
        return out

class UNetAutoencoderFixedDepth(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, base_channels=BASE_CHANNELS):
        """
        A U-Net-style autoencoder with fixed depth 4.
        
        Design rationale:
          • Fully convolutional: preserves spatial structure (critical for preserving phase and frequency).
          • Skip connections: help retain fine oscillatory details by merging low-level encoder features with decoder features.
          • Fixed depth (4 levels): For inputs of shape (2, 563, 1000), four downsamplings yield feature maps
            with sizes roughly [282, 141, 71, 36] in height and [500, 250, 125, 63] in width—sufficient for multi-scale
            feature extraction without over-collapsing spatial information.
          • No fully connected bottleneck: This avoids losing local relationships that are essential in oscillatory data.
        
        Args:
            in_channels (int): Number of input channels (e.g. 2 for X and Y BPM data).
            base_channels (int): Number of channels for the first encoder block.
        """
        super(UNetAutoencoderFixedDepth, self).__init__()
        
        # ----- Encoder -----
        # Level 1
        self.enc1 = self.conv_block(in_channels, base_channels)
        # Level 2
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        # Level 3
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        # Level 4
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck (fully convolutional; preserves spatial dimensions)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # ----- Decoder -----
        # Level 4 decoding: upsample bottleneck to combine with encoder level 4
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)  # concatenated channels
        
        # Level 3 decoding
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        
        # Level 2 decoding
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        # Level 1 decoding
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        # Final output convolution to get back to in_channels
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        """A double convolution block with BatchNorm and LeakyReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        # ----- Encoder pathway -----
        e1 = self.enc1(x)         # shape: (B, base_channels, H, W)
        p1 = F.max_pool2d(e1, 2)    # Downsample by 2
        e2 = self.enc2(p1)        # shape: (B, base_channels*2, H/2, W/2)
        p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2)        # shape: (B, base_channels*4, H/4, W/4)
        p3 = F.max_pool2d(e3, 2)
        e4 = self.enc4(p3)        # shape: (B, base_channels*8, H/8, W/8)
        p4 = F.max_pool2d(e4, 2)
        
        # Bottleneck
        b = self.bottleneck(p4)   # shape: (B, base_channels*16, H/16, W/16)
        
        # ----- Decoder pathway with skip connections -----
        d4 = self.up4(b)          # Upsample: (B, base_channels*8, H/8, W/8)
        # Ensure dimensions match (pad if necessary)
        if d4.size()[2:] != e4.size()[2:]:
            d4 = self._pad_to_match(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate along channels -> (B, base_channels*16, H/8, W/8)
        d4 = self.dec4(d4)        # (B, base_channels*8, H/8, W/8)
        
        d3 = self.up3(d4)         # (B, base_channels*4, H/4, W/4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = self._pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)  # (B, base_channels*8, H/4, W/4)
        d3 = self.dec3(d3)        # (B, base_channels*4, H/4, W/4)
        
        d2 = self.up2(d3)         # (B, base_channels*2, H/2, W/2)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = self._pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, base_channels*4, H/2, W/2)
        d2 = self.dec2(d2)        # (B, base_channels*2, H/2, W/2)
        
        d1 = self.up1(d2)         # (B, base_channels, H, W)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = self._pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)  # (B, base_channels*2, H, W)
        d1 = self.dec1(d1)        # (B, base_channels, H, W)
        
        out = self.final_conv(d1) # (B, in_channels, H, W)
        return out

    def _pad_to_match(self, x, ref):
        """
        Pad x so that its spatial dimensions match those of ref.
        """
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

class UNetAutoencoderFixedDepthCheckpoint(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, base_channels=BASE_CHANNELS):
        """
        A U-Net–style autoencoder with fixed depth 4 using activation checkpointing
        to reduce GPU memory usage. This design is well-suited for oscillatory data since it:
          • Remains fully convolutional (preserving phase and frequency information),
          • Uses skip connections to retain fine details,
          • Has a fixed depth (4 levels) that balances global context with local detail.
          
        Activation checkpointing is applied to each convolutional block so that intermediate
        activations are recomputed during backpropagation rather than stored in memory.
        """
        super(UNetAutoencoderFixedDepthCheckpoint, self).__init__()
        
        # Encoder blocks
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck block (fully convolutional)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        """A double convolution block with BatchNorm and LeakyReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=False),  # Use inplace=False for checkpointing safety.
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=False)
        )
    
    def _pad_to_match(self, x, ref):
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
    
    def forward(self, x):
        # Encoder path with checkpointing.
        e1 = checkpoint(self.enc1, x, use_reentrant=False)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        
        e2 = checkpoint(self.enc2, p1, use_reentrant=False)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2)
        
        e3 = checkpoint(self.enc3, p2, use_reentrant=False)
        p3 = F.max_pool2d(e3, kernel_size=2, stride=2)
        
        e4 = checkpoint(self.enc4, p3, use_reentrant=False)
        p4 = F.max_pool2d(e4, kernel_size=2, stride=2)
        
        # Bottleneck with checkpointing.
        b = checkpoint(self.bottleneck, p4, use_reentrant=False)
        
        # Decoder path with checkpointing.
        d4 = self.up4(b)
        if d4.size()[2:] != e4.size()[2:]:
            d4 = self._pad_to_match(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = checkpoint(self.dec4, d4, use_reentrant=False)
        
        d3 = self.up3(d4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = self._pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = checkpoint(self.dec3, d3, use_reentrant=False)
        
        d2 = self.up2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = self._pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = checkpoint(self.dec2, d2, use_reentrant=False)
        
        d1 = self.up1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = self._pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = checkpoint(self.dec1, d1, use_reentrant=False)
        
        out = self.final_conv(d1)
        return out

