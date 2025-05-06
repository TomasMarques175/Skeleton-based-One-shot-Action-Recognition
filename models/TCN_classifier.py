#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:21:43 2020

@author: asabater
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn import TCN
from pytorch_tcn import TemporalConvNet
from tensorflow.keras import Model, Sequential
# BatchNormalization
from tensorflow.keras.layers import Dense, RepeatVector, TimeDistributed, Lambda, Masking
from tensorflow.keras.layers import Input
import tensorflow as tf
import tensorflow.keras.backend as K
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np


class EncoderTCN1(Model):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections,
                 lstm_dropout, padding, dilations,
                 masking=False,
                 prediction_mode=False,
                 tcn_batch_norm=False,
                 **kwargs
                 ):
        
        super(EncoderTCN1, self).__init__()
        
        self.encoder_layers = []

        # Add masking layer
        if masking:
            print('MASKING')
            self.encoder_layers.append(Masking())

        num_tcn = len(dilations)
        print('num_tcn:', num_tcn)
        
        for i in range(num_tcn-1):
            l = TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks=nb_stacks,
                use_skip_connections=use_skip_connections,
                padding=padding,
                dilations=dilations[i],
                dropout_rate=lstm_dropout,
                return_sequences=True,
                use_batch_norm=tcn_batch_norm
            )
            self.encoder_layers.append(l)
            print('TCN', i, dilations[i], l.receptive_field)

        l = TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            use_skip_connections=use_skip_connections,
            padding=padding,
            dilations=dilations[-1],
            dropout_rate=lstm_dropout,
            return_sequences=prediction_mode,
            use_batch_norm=tcn_batch_norm
        )
        self.encoder_layers.append(l)
        # print('TCN', -1, dilations[-1], l.receptive_field)

        for l in self.encoder_layers:
            print(l)

        self.encoder = Sequential(self.encoder_layers)


    def call(self, x):
        encoder = self.encoder(x)
        return encoder

    def get_embedding(self, x):
        emb = self.encoder(x)
        return emb

# conv_params -> nb_filters, kernel_size, nb_stacks, use_skip_connections
class TCN_clf1(Model):
    def __init__(self,
                 num_feats,
                 conv_params,
                 lstm_dropout,
                 masking,
                 triplet, classification,
                 clf_neurons=None,
                 num_classes=None,
                 prediction_mode=False,
                 lstm_decoder=False,
                 num_layers=None,
                 num_neurons=None,
                 tcn_batch_norm=False,
                 use_gru=False,
                 **kwargs
                 ):
        super(TCN_clf1, self).__init__()

        if len(conv_params) == 4:
            nb_filters, kernel_size, nb_stacks, use_skip_connections = conv_params
            padding = 'causal'
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 5:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding = conv_params
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 6:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding, dilations = conv_params
            if type(dilations) == int:
                dilations = [dilations]
            if type(dilations) == str:
                dilations = ast.literal_eval(dilations)
            else:
                dilations = [[i for i in [1, 2, 4, 8, 16, 32] if i <= d]
                             for d in dilations]
            print('dilations', dilations)
        else:
            raise ValueError(
                'conv_params length not recognized', len(conv_params))

        self.encoder_net = EncoderTCN1(
            num_feats=num_feats,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            use_skip_connections=use_skip_connections,
            padding=padding,
            dilations=dilations,
            lstm_dropout=lstm_dropout,
            masking=masking,
            prediction_mode=prediction_mode,
            tcn_batch_norm=tcn_batch_norm)

        self.norm = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.triplet = triplet

        if clf_neurons != 0:
            self.clf_dense = Dense(clf_neurons, activation='relu')
        self.clf_neurons = clf_neurons

        if classification:
            self.clf_out = Dense(
                num_classes, activation='softmax', name='out_clf')
        self.classification = classification

        # self.unify = Lambda(lambda x: x/tf.math.reduce_sum(x, axis=-1, keepdims=True))

    def call(self, x):
        encoder_raw = self.encoder_net(x)

        if self.clf_neurons != 0:
            encoder = self.clf_dense(encoder_raw)
        else:
            encoder = encoder_raw

        out = []

        if self.triplet:
            emb = self.norm(encoder)
            out.append(emb)
        
        if self.classification:
            clf = self.clf_out(encoder)
            out.append(clf)

        return out

    def get_embedding(self, x, batch=None, unify=False):

        if batch is None or batch <= 0:
            emb = self.encoder_net(x)
            if self.clf_neurons != 0:
                emb = self.clf_dense(emb)
            emb = self.norm(emb)
            return emb

        elif batch > 0:
            embs_data = []
            for num_sample in range(x.shape[0]):
                for num_frame in range(x.shape[1]):
                    start, end = max(0, num_frame-batch+1), num_frame+1
                    embs_data.append(pad_sequences(
                        x[num_sample:num_sample+1, start:end], maxlen=batch, dtype='float32', padding='pre'))
            preds_batch = [self.get_embedding(np.concatenate(
                embs_data[ind:ind+512]), batch=None) for ind in range(0, len(embs_data), 512)]
            preds_batch = np.concatenate(preds_batch)
            preds_batch = preds_batch.reshape(
                (x.shape[0], x.shape[1], preds_batch.shape[1]))
            return preds_batch

        else:
            raise ValueError('Incorrect prediction batch')

    def set_encoder_return_sequences(self, return_sequences):
        l = [l for l in self.encoder_net.layers if type(l) == TCN][-1]
        l.return_sequences = return_sequences

# --- PyTorch Version ---

class EncoderTCN(nn.Module):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections, # Added use_skip_connections for clarity, though not used in pytorch_tcn's TemporalConvNet directly
                 lstm_dropout, padding, dilations, # padding isn't directly used by pytorch_tcn TemporalConvNet constructor either
                 masking=False,
                 prediction_mode=False, # prediction_mode controls output shape now
                 tcn_batch_norm=False): # tcn_batch_norm not implemented in the provided pytorch_tcn
        super().__init__()

        self.prediction_mode = prediction_mode # Store whether to return sequence or last step

        layers = []

        if masking:
            # Masking in PyTorch usually requires handling padding explicitly
            # or using PackedSequences if dealing with variable lengths.
            # nn.Conv1d doesn't have a built-in masking argument like Keras.
            # You might need to handle padding/masking *before* the TCN or
            # use libraries that support masked convolutions if needed.
            print("MASKING — requires manual implementation or specific handling in PyTorch")

        # Note: The provided pytorch_tcn.TemporalConvNet implementation uses a fixed dilation scheme (2**i)
        # and doesn't directly use nb_stacks, use_skip_connections, padding, or the complex dilations list structure
        # from the TensorFlow version setup. It uses num_channels (list of filters per level) and dropout.
        # We'll adapt based on the structure called in train.py.
        # conv_params = [256, 4, 2, True, "causal", [4]] -> nb_filters=256, kernel_size=4, nb_stacks=2, dilations=[[1,2,4]]

        # Let's assume nb_filters is the number of channels for *each* level in the stack,
        # and nb_stacks determines the depth of *each* TemporalBlock (which isn't how TemporalConvNet is structured).
        # Let's reinterpret based on TemporalConvNet: num_channels defines filters per *level* of dilation.

        # Reinterpreting params for pytorch_tcn.TemporalConvNet:
        # nb_filters -> use as the number of channels in the output list
        # nb_stacks -> Use as the number of levels (length of num_channels list)
        # kernel_size -> kernel_size
        # lstm_dropout -> dropout

        num_levels = nb_stacks # Number of TemporalBlocks based on TF param interpretation
        num_channels = [nb_filters] * num_levels # List of output channels for each level

        print(f"PyTorch TCN: num_levels={num_levels}, num_channels={num_channels}, kernel_size={kernel_size}, dropout={lstm_dropout}")

        # *** This assumes the pytorch_tcn.TemporalConvNet is the desired structure ***
        # It differs significantly from the structure implied by the TF EncoderTCN1 loop structure.
        # If you need the *exact* TF structure (multiple TCN blocks sequentially), you'd build it differently.
        self.tcn = TemporalConvNet(
            num_inputs=num_feats,
            num_channels=num_channels, # e.g., [256, 256] if nb_stacks=2
            kernel_size=kernel_size,
            dropout=lstm_dropout
        )
        # If you *did* want multiple sequential TemporalConvNets like in TF:
        # layers = []
        # current_channels = num_feats
        # for i in range(num_tcn): # num_tcn from len(dilations)
        #     # You'd need to adapt TemporalConvNet or TemporalBlock to handle custom dilations per block
        #     # The current pytorch_tcn code has hardcoded dilations = 2**i within TemporalConvNet
        #     tcn_layer = TemporalConvNet(...) # Needs adjustment for inputs/dilations
        #     layers.append(tcn_layer)
        #     current_channels = nb_filters # output of TCN
        # self.encoder = nn.Sequential(*layers) # Use this if building sequentially

        # Using the single TemporalConvNet based on its implementation in pytorch_tcn.py
        self.encoder = nn.Sequential(self.tcn) # Wrap in sequential for consistency if needed, or just use self.tcn directly

    def forward(self, x):
        # Input x: (N, L, C_in)
        x = x.permute(0, 2, 1)  # Permute to (N, C_in, L) for Conv1d
        y = self.encoder(x)    # Output y: (N, C_out, L_out)
        if self.prediction_mode:
            y = y.permute(0, 2, 1) # Permute back to (N, L_out, C_out) if returning sequence
            return y
        else:
            # Return only the last time step
            return y[:, :, -1] # Output shape: (N, C_out)

    # get_embedding should likely return the raw encoder output (last step)
    def get_embedding(self, x):
         # Input x: (N, L, C_in)
        x = x.permute(0, 2, 1)  # Permute to (N, C_in, L) for Conv1d
        y = self.encoder(x)    # Output y: (N, C_out, L_out)
        # Return the last time step as the embedding
        return y[:, :, -1]     # Output shape: (N, C_out)


class TCN_clf(nn.Module):
    def __init__(self,
                 num_feats,
                 conv_params,
                 lstm_dropout,
                 masking,
                 triplet,
                 classification,
                 clf_neurons=None,
                 num_classes=None,
                 prediction_mode=False, # Pass this to EncoderTCN
                 # These params seem related to the TF LSTM version, not TCN, but keep if needed elsewhere
                 lstm_decoder=False,
                 num_layers=None,
                 num_neurons=None,
                 # ---
                 tcn_batch_norm=False, # Pass to EncoderTCN
                 use_gru=False,        # Not used in TCN part
                 **kwargs):

        super(TCN_clf, self).__init__()

        # Parse conv_params (assuming the same logic as TF version for setup)
        if len(conv_params) == 4:
            nb_filters, kernel_size, nb_stacks, use_skip_connections = conv_params
            padding = 'causal' # Note: padding param isn't directly used by pytorch_tcn
            dilations = [1, 2, 4, 8, 16, 32] # Note: dilations param isn't directly used by pytorch_tcn structure
        elif len(conv_params) == 5:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding = conv_params
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 6:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding, dilations_param = conv_params
            # Process dilations if needed for a custom TCN structure, pytorch_tcn ignores this complex format
            # Example parsing (adapt if needed):
            if isinstance(dilations_param, int):
                dilations = [dilations_param]
            elif isinstance(dilations_param, str):
                dilations = ast.literal_eval(dilations_param)
            else: # Assuming list of max dilations per TCN block
                 # This structure implies multiple sequential TCNs like in TF EncoderTCN1
                 # The current pytorch_tcn.TemporalConvNet doesn't support this directly
                 dilations_structure = [[i for i in [1, 2, 4, 8, 16, 32] if i <= d] for d in dilations_param]
                 print("Complex dilations structure:", dilations_structure, "- Not directly used by provided pytorch_tcn.py")
                 dilations = dilations_structure # Store it, but may not be used by EncoderTCN as written
        else:
            raise ValueError('conv_params length not recognised', len(conv_params))

        print("Initializing PyTorch EncoderTCN...")
        # Encoder - Pass relevant interpreted parameters
        self.encoder_net = EncoderTCN(num_feats=num_feats,
                                      nb_filters=nb_filters,
                                      kernel_size=kernel_size,
                                      nb_stacks=nb_stacks, # Interpreted as num_levels for pytorch_tcn
                                      use_skip_connections=use_skip_connections, # Not used by pytorch_tcn
                                      padding=padding,          # Not used by pytorch_tcn
                                      dilations=dilations,        # Not used by pytorch_tcn
                                      lstm_dropout=lstm_dropout,  # Used as dropout
                                      masking=masking,          # Placeholder info
                                      prediction_mode=prediction_mode, # Controls encoder output shape
                                      tcn_batch_norm=tcn_batch_norm) # Not implemented in pytorch_tcn

        self.triplet = triplet
        self.classification = classification

        # Determine the input size for the classifier head
        # EncoderTCN outputs (N, nb_filters) when prediction_mode=False
        encoder_output_features = nb_filters

        # Optional classifier head dense layer (applied *after* the TCN encoder)
        self.clf_dense = None
        self.clf_neurons = clf_neurons if clf_neurons is not None else 0 # Store the config
        clf_input_features = encoder_output_features
        if self.clf_neurons > 0:
            self.clf_dense = nn.Linear(encoder_output_features, self.clf_neurons)
            clf_input_features = self.clf_neurons # Input to the final layer is now output of clf_dense

        # Final classification output layer
        self.clf_out = None
        if classification:
            if num_classes is None:
                raise ValueError("num_classes must be provided if classification is True")
            self.clf_out = nn.Linear(clf_input_features, num_classes)
            # Note: Softmax is often applied *outside* the model during loss calculation (e.g., with nn.CrossEntropyLoss)
            # If you need softmax output directly, add nn.Softmax(dim=1) here or apply it in forward pass.


    def forward(self, x):
        # Input x: (N, L, C)
        # Encoder output depends on prediction_mode
        # Assuming prediction_mode=False (default for training/classification) -> (N, C_out)
        encoder_features = self.encoder_net(x) # Shape (N, nb_filters)

        # Pass through optional intermediate dense layer
        if self.clf_dense is not None:
            features_for_clf = F.relu(self.clf_dense(encoder_features))
        else:
            features_for_clf = encoder_features

        out = []

        # Triplet output (embedding) - Use the features *before* the final clf_out layer
        if self.triplet:
            # Usually normalize the features *after* the optional clf_dense if it exists,
            # or the raw encoder output if it doesn't. Let's use features_for_clf.
            emb = F.normalize(features_for_clf, p=2, dim=-1)
            out.append(emb)

        # Classification output
        if self.classification:
            if self.clf_out is None:
                 raise RuntimeError("Model configured for classification, but clf_out layer is missing.")
            # Apply final classification layer
            clf_logits = self.clf_out(features_for_clf)
            # Apply softmax here if you need probabilities directly from the model
            # clf_probs = F.softmax(clf_logits, dim=1)
            # out.append(clf_probs)
            # Otherwise, return logits (common for nn.CrossEntropyLoss)
            out.append(clf_logits)

        # Return list of outputs (consistent with Keras multi-output models)
        # If only one output is active, consider returning just that tensor
        if len(out) == 1:
             return out[0]
        else:
             return out # Return list [emb, clf] or [emb] or [clf]

    # Get embedding: normalized output *before* final classification layer
    def get_embedding(self, x, batch=None, unify=False):
        # Ensure unify param is handled if needed (it's not used currently)

        if batch is None or batch <= 0:
            self.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculation
                # Get raw encoder output (last time step) -> (N, nb_filters)
                emb_raw = self.encoder_net.get_embedding(x)

                # Apply the intermediate dense layer if it exists
                if self.clf_dense is not None:
                   features_for_norm = F.relu(self.clf_dense(emb_raw))
                else:
                   features_for_norm = emb_raw

                # Normalize to get the final embedding
                emb = F.normalize(features_for_norm, p=2, dim=-1)
            return emb

        # --- Batch > 0 Logic ---
        # This needs careful translation from TF/Numpy to PyTorch tensors
        # The TF version calculates embeddings frame-by-frame with padding, which can be slow.
        # Consider if this exact frame-by-frame embedding is necessary or if a simpler approach works.
        elif batch > 0:
            self.eval()
            with torch.no_grad():
                # Example PyTorch implementation (might need optimization)
                N, L, C = x.shape
                # Ensure x is a torch tensor on the correct device
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32) # Convert if numpy
                x = x.to(next(self.parameters()).device) # Move to model's device

                embs_list = []
                # This loop is potentially very slow in Python
                for i in range(N):
                    sample_embs = []
                    for j in range(L):
                        start = max(0, j - batch + 1)
                        end = j + 1
                        sub_seq = x[i:i+1, start:end, :] # Shape (1, L_sub, C)

                        # Pad sequence on the left (pre-padding)
                        pad_len = batch - sub_seq.shape[1]
                        if pad_len < 0: # Should not happen if batch >= 1
                            padded_seq = sub_seq[:, -batch:, :] # Take last 'batch' frames
                        elif pad_len > 0:
                            pad_tensor = torch.zeros((1, pad_len, C), device=x.device, dtype=x.dtype)
                            padded_seq = torch.cat([pad_tensor, sub_seq], dim=1) # Shape (1, batch, C)
                        else:
                            padded_seq = sub_seq # Shape (1, batch, C)

                        # Get embedding for this padded subsequence
                        # Note: get_embedding itself expects (N, L, C) and handles batch=None case
                        emb = self.get_embedding(padded_seq, batch=None) # Shape (1, embedding_dim)
                        sample_embs.append(emb)

                    embs_list.append(torch.cat(sample_embs, dim=0)) # Shape (L, embedding_dim)

                # Stack embeddings for all samples
                all_embs = torch.stack(embs_list, dim=0) # Shape (N, L, embedding_dim)
                return all_embs
                # --- End Batch > 0 Logic ---
        else:
            raise ValueError('Incorrect prediction batch value')

    # This method seems specific to the Keras TCN implementation and isn't standard in PyTorch TCNs
    # def set_encoder_return_sequences(self, return_sequences):
    #     # In our PyTorch EncoderTCN, this is controlled by the 'prediction_mode' flag
    #     self.encoder_net.prediction_mode = return_sequences
    #     print(f"EncoderTCN prediction_mode set to: {self.encoder_net.prediction_mode}")
