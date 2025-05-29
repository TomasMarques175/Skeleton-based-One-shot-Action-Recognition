import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 padding,
                 activation='relu',
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 use_layer_norm=False):
        super(ResidualBlock, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = activation if callable(activation) else getattr(F, activation)
        
        # Calculate effective padding if 'same' is requested
        if padding == 'same':
            padding_amt = ((dilation * (kernel_size - 1)) // 2)
        else:
            padding_amt = padding

        # Conv layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=padding_amt, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                            padding=padding_amt, dilation=dilation)

        # Optional normalization
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif use_layer_norm:
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Shape match if in/out channels differ
        self.match_shape = (in_channels != out_channels)
        if self.match_shape:
            self.shape_match_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.batch_norm2(out)
        out = self.activation(out)
        out = self.dropout2(out)

        if self.shape_match_conv:
            x = self.shape_match_conv(x)

        res = out + x
        if out.size(-1) != residual.size(-1):
            min_len = min(out.size(-1), residual.size(-1))
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]

        return out


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, nb_filters, kernel_size, dilations, nb_stacks=1,
                 dropout_rate=0.0, activation=F.relu, padding='same',
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

        print(f"TCN: padding: {self.padding}")
        print(f"TCN: Number of stacks: {self.nb_stacks}")
        print(f"TCN: Number of filters: {self.nb_filters}")
        print(f"TCN: Kernel size: {self.kernel_size}")
        print(f"TCN: Dilations: {self.dilations}")
        print(f"TCN: Dropout rate: {self.dropout_rate}")
        print(f"TCN: Activation function: {self.activation.__name__}")
        print(f"TCN: Use skip connections: {self.use_skip_connections}")
        print(f"TCN: Use batch normalization: {self.use_batch_norm}")
        print(f"TCN: Use layer normalization: {self.use_layer_norm}")
        print(f"TCN: Total number of blocks: {total_num_blocks}")
        
        for s in range(nb_stacks):
            for i, d in enumerate(self.dilations):
                print(f"[DEBUG] dilation at block {len(self.residual_blocks)}: {d} (type: {type(d)})")
                res_block_filters = nb_filters[i] if isinstance(nb_filters, list) else nb_filters
                in_channels = input_dim if len(self.residual_blocks) == 0 else res_block_filters
                block = ResidualBlock(
                    in_channels=in_channels,
                    out_channels=res_block_filters,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=padding,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm
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

        x = x[:, :, self.output_slice_index]  # slice along time axis
        return x
