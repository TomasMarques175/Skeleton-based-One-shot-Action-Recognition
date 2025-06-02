import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 padding_mode: str, # Expect 'causal' or 'same'
                 activation_str='relu',
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=True): # Keras-TCN often uses weight_norm
        super(ResidualBlock, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        if callable(activation_str):
            self.activation_fn = activation_str
        else:
            self.activation_fn = getattr(F, activation_str)
        self.padding_mode = padding_mode

        # Convolutional layers
        conv_padding1 = 0
        conv_padding2 = 0

        if self.padding_mode == 'causal':
            # Pad on the left for causality
            self.pad1 = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
            self.pad2 = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        elif self.padding_mode == 'same':
            # Symmetric padding for 'same' output length
            conv_padding1 = (kernel_size - 1) * dilation // 2
            conv_padding2 = (kernel_size - 1) * dilation // 2
            self.pad1 = nn.Identity()
            self.pad2 = nn.Identity()
        else:
            raise ValueError(f"padding_mode must be 'causal' or 'same', got {padding_mode}")

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=conv_padding1, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=conv_padding2, dilation=dilation)

        print(f"\t\t* self.conv1 kernel_size={self.conv1.kernel_size}, "
              f"self.conv1 padding={self.pad1}, dilation={self.conv1.dilation}")
        print(f"\t\t* self.conv2 kernel_size={self.conv2.kernel_size}, "
                f"self.conv2 padding={self.pad2}, dilation={self.conv2.dilation}")

        if use_weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)

        # Normalization layers
        self.norm1, self.norm2 = None, None
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif use_layer_norm:
            # LayerNorm on (N, C, L) should normalize over C (channels)
            # normalized_shape=out_channels achieves this for input (..., C)
            # For (N,C,L) directly, need to specify normalized_shape as just out_channels
            self.norm1 = nn.LayerNorm(out_channels) 
            self.norm2 = nn.LayerNorm(out_channels)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x_input):  # x_input shape: (batch, channels, time)
        print(f"\n\t\t>> Input to ResidualBlock: {x_input.shape}")

        # Block 1
        out = self.pad1(x_input)
        print(f"\t\t  After pad1: {out.shape}")
        out = self.conv1(out)
        print(f"\t\t  After conv1: {out.shape}")
        if self.norm1:
            out = self.norm1(out)
            print(f"\t\t  After norm1: {out.shape}")
        out = self.activation_fn(out)
        out = self.dropout1(out)
        print(f"\t\t  After activation + dropout1: {out.shape}")

        # Block 2
        out = self.pad2(out)
        print(f"\t\t  After pad2: {out.shape}")
        out = self.conv2(out)
        print(f"\t\t  After conv2: {out.shape}")
        if self.norm2:
            out = self.norm2(out)
            print(f"\t\t  After norm2: {out.shape}")
        out = self.activation_fn(out)
        out = self.dropout2(out)
        print(f"\t\t  After activation + dropout2: {out.shape}")

        # Skip connection path
        res = x_input
        if self.downsample:
            print("\t\t  * Applying 1x1 conv to match input/output channels")
            res = self.downsample(res)
            print(f"\t\t  After downsample (res): {res.shape}")

        # Length check
        if out.size(-1) != res.size(-1):
            print(f"\t\t  * Sequence length mismatch! out: {out.shape[-1]}, res: {res.shape[-1]}")
            target_len = res.size(-1)
            if out.size(-1) > target_len:
                out = out[..., :target_len]
                print(f"\t\t  -> Truncated out to: {out.shape}")
            elif out.size(-1) < target_len:
                padding_diff = target_len - out.size(-1)
                out = F.pad(out, (0, padding_diff))
                print(f"\t\t  -> Padded out to: {out.shape}")

        output_sum = out + res
        print(f"\t\t  After residual addition: {output_sum.shape}")

        # Final activation
        final_output = self.activation_fn(output_sum)
        print(f"\t\t  Final output after activation: {final_output.shape}")

        return final_output


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, nb_filters, kernel_size, dilations, nb_stacks=1,
                 dropout_rate=0.0, activation=F.relu, padding='causal',
                 use_skip_connections=True, use_batch_norm=False,
                 use_layer_norm=False, kernel_initializer=None):
        super(TemporalConvNet, self).__init__()

        self.padding = padding
        self.nb_stacks = nb_stacks
        if isinstance(dilations[0], list):
            self.dilations = [d for sublist in dilations for d in sublist]
        else:
            self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        self.residual_blocks = nn.ModuleList()
        
        total_num_blocks = nb_stacks * len(dilations)
        if not use_skip_connections:
            total_num_blocks += 1

        print(f"\t\t* TCN: padding: {self.padding}")
        print(f"\t\t* TCN: Number of stacks: {self.nb_stacks}")
        print(f"\t\t* TCN: Number of filters: {self.nb_filters}")
        print(f"\t\t* TCN: Kernel size: {self.kernel_size}")
        print(f"\t\t* TCN: Dilations: {self.dilations}")
        print(f"\t\t* TCN: Dropout rate: {self.dropout_rate}")
        print(f"\t\t* TCN: Activation function: {self.activation.__name__}")
        print(f"\t\t* TCN: Use skip connections: {self.use_skip_connections}")
        print(f"\t\t* TCN: Use batch normalization: {self.use_batch_norm}")
        print(f"\t\t* TCN: Use layer normalization: {self.use_layer_norm}")
        print(f"\t\t* TCN: Total number of blocks: {total_num_blocks}")
        print("\n")
        for s in range(nb_stacks):
            for i, d in enumerate(self.dilations):
                print(f"\t\t* [DEBUG] dilation at block {len(self.residual_blocks)}: {d} (type: {type(d)})")
                res_block_filters = nb_filters[i] if isinstance(nb_filters, list) else nb_filters
                in_channels = input_dim if len(self.residual_blocks) == 0 else res_block_filters
                block = ResidualBlock(
                    in_channels=in_channels,
                    out_channels=res_block_filters,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding_mode=padding,
                    dropout_rate=dropout_rate,
                    activation_str=activation,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    use_weight_norm=False
                )
                self.residual_blocks.append(block)
        self.output_slice_index = None

    def forward(self, x):
        # Expecting input shape: (batch, channels, time)
        for block in self.residual_blocks:
            x = block(x)
        time_dim = x.shape[-1]
        if self.padding == 'same':
            self.output_slice_index = time_dim // 2
        else:
            self.output_slice_index = -1

        return x  # Shape: (N, C, L)
