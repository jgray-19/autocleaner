import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint

from config import BASE_CHANNELS, INIT, MASK_MAX_GAIN, MODEL_DEPTH, NUM_CHANNELS


# -------------------------------------------------------------------
# Xavier initialization helper
def xavier_init(m):
    """
    Xavier-initialize Conv2d, ConvTranspose2d and Linear layers.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class UNetAutoencoder(nn.Module):
    def __init__(
        self, in_channels=NUM_CHANNELS, base_channels=BASE_CHANNELS, depth=MODEL_DEPTH
    ):
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
        self.apply(xavier_init)
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
            self.decoders.append(
                nn.ModuleDict(
                    {
                        "up_conv": nn.ConvTranspose2d(
                            curr_channels * 2, curr_channels, kernel_size=2, stride=2
                        ),
                        "conv": self.conv_block(curr_channels * 2, curr_channels),
                    }
                )
            )

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
            nn.LeakyReLU(0.01, inplace=True),
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
            out = decoder["up_conv"](out)
            # Retrieve corresponding encoder output (skip connection)
            skip = skip_connections.pop()
            # In case dimensions do not match exactly (due to rounding), we pad accordingly.
            if out.size()[2:] != skip.size()[2:]:
                diffY = skip.size(2) - out.size(2)
                diffX = skip.size(3) - out.size(3)
                out = F.pad(
                    out,
                    [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
                )
            # Concatenate along channel dimension
            out = torch.cat([skip, out], dim=1)
            out = decoder["conv"](out)

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
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = self.conv_block(
            base_channels * 16, base_channels * 8
        )  # concatenated channels

        # Level 3 decoding
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)

        # Level 2 decoding
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)

        # Level 1 decoding
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self._initialize_weights()
        # (optionally) start near identity  → exp(0)=1
        if INIT == "weiner":
            torch.nn.init.zeros_(self.final_conv.weight)
            torch.nn.init.zeros_(self.final_conv.bias)

    def conv_block(self, in_ch, out_ch):
        """A double convolution block with BatchNorm and LeakyReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, mag_spec):
        # ----- Encoder pathway -----
        e1 = self.enc1(mag_spec)  # shape: (B, base_channels, H, W)
        p1 = F.max_pool2d(e1, 2)  # Downsample by 2
        e2 = self.enc2(p1)  # shape: (B, base_channels*2, H/2, W/2)
        p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2)  # shape: (B, base_channels*4, H/4, W/4)
        p3 = F.max_pool2d(e3, 2)
        e4 = self.enc4(p3)  # shape: (B, base_channels*8, H/8, W/8)
        p4 = F.max_pool2d(e4, 2)

        # Bottleneck
        b = self.bottleneck(p4)  # shape: (B, base_channels*16, H/16, W/16)

        # ----- Decoder pathway with skip connections -----
        d4 = self.up4(b)  # Upsample: (B, base_channels*8, H/8, W/8)
        # Ensure dimensions match (pad if necessary)
        if d4.size()[2:] != e4.size()[2:]:
            d4 = self._pad_to_match(d4, e4)
        d4 = torch.cat(
            [d4, e4], dim=1
        )  # Concatenate along channels -> (B, base_channels*16, H/8, W/8)
        d4 = self.dec4(d4)  # (B, base_channels*8, H/8, W/8)

        d3 = self.up3(d4)  # (B, base_channels*4, H/4, W/4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = self._pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)  # (B, base_channels*8, H/4, W/4)
        d3 = self.dec3(d3)  # (B, base_channels*4, H/4, W/4)

        d2 = self.up2(d3)  # (B, base_channels*2, H/2, W/2)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = self._pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, base_channels*4, H/2, W/2)
        d2 = self.dec2(d2)  # (B, base_channels*2, H/2, W/2)

        d1 = self.up1(d2)  # (B, base_channels, H, W)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = self._pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)  # (B, base_channels*2, H, W)
        d1 = self.dec1(d1)  # (B, base_channels, H, W)

        mask_log = self.final_conv(d1)  # raw logits
        mask = torch.exp(mask_log)  # allow <1 (kill noise) & >1 (boost peaks)
        return torch.clamp(mask, max=MASK_MAX_GAIN)  # pl-module will do mask * x_noisy

    def _pad_to_match(self, x, ref):
        """
        Pad x so that its spatial dimensions match those of ref.
        """
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return F.pad(
            x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                m, torch.nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


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
        self.apply(xavier_init)

        # Encoder blocks
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck block (fully convolutional)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """A double convolution block with BatchNorm and LeakyReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(
                0.01, inplace=False
            ),  # Use inplace=False for checkpointing safety.
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=False),
        )

    def _pad_to_match(self, x, ref):
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return F.pad(
            x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

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


class DoubleConv(nn.Module):
    """(Convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = ((3 - 1) // 2) * dilation
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


def center_crop(tensor, target_height, target_width):
    _, _, h, w = tensor.size()
    start_y = (h - target_height) // 2
    start_x = (w - target_width) // 2
    return tensor[
        :, :, start_y : start_y + target_height, start_x : start_x + target_width
    ]


class UpBlock(nn.Module):
    """Upsampling block with center-cropping both tensors to match dimensions."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Convert in_channels (from previous decoder) to out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, concatenate with skip (skip_channels) and then apply double conv
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Determine the target spatial size (minimum of both tensors)
        target_h = min(x.size(2), skip.size(2))
        target_w = min(x.size(3), skip.size(3))
        x = center_crop(x, target_h, target_w)
        skip = center_crop(skip, target_h, target_w)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DownBlock(nn.Module):
    """Down block: double conv followed by max pooling.
    Returns both the conv output (for skip connection) and the pooled result."""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dilation=dilation)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pooled = F.max_pool2d(x_conv, kernel_size=2, ceil_mode=True)
        return x_conv, x_pooled


class ModifiedUNetFixed(nn.Module):
    def __init__(
        self,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        base_channels=BASE_CHANNELS,
    ):
        super().__init__()
        self.inc = DoubleConv(
            in_channels, base_channels
        )  # output: base_channels (e.g., 16)
        self.down1 = DownBlock(base_channels, base_channels * 2)  # skip: 32, pooled: 32
        self.down2 = DownBlock(
            base_channels * 2, base_channels * 4, dilation=2
        )  # skip: 64, pooled: 64
        self.down3 = DownBlock(
            base_channels * 4, base_channels * 8, dilation=4
        )  # skip: 128, pooled: 128
        self.down4 = DownBlock(
            base_channels * 8, base_channels * 16, dilation=8
        )  # skip: 256, pooled: 256

        # Bottleneck: no pooling, still at lowest resolution.
        self.bottleneck = DoubleConv(
            base_channels * 16, base_channels * 16, dilation=2
        )  # output: 256

        # Decoder: each UpBlock takes the bottleneck/previous decoder output and the corresponding skip.
        self.up1 = UpBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 16,
            out_channels=base_channels * 8,
        )  # from 256 to 128
        self.up2 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 4,
        )  # from 128 to 64
        self.up3 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
        )  # from 64 to 32
        self.up4 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
        )  # from 32 to 16

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        if INIT == "weiner":
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2_skip, x2 = self.down1(x1)
        x3_skip, x3 = self.down2(x2)
        x4_skip, x4 = self.down3(x3)
        x5_skip, x5 = self.down4(x4)
        x_bottleneck = self.bottleneck(x5)
        x = self.up1(x_bottleneck, x5_skip)
        x = self.up2(x, x4_skip)
        x = self.up3(x, x3_skip)
        x = self.up4(x, x2_skip)
        x = self.outc(x)
        mask = torch.exp(x)
        mask = torch.clamp(mask, max=MASK_MAX_GAIN)
        return mask
