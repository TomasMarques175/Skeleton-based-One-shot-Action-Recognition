#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:21:43 2020

@author: asabater
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_tcn import TCN
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


class EncoderTCN(nn.Module):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections,
                 lstm_dropout, padding, dilations,
                 masking=False,
                 prediction_mode=False,
                 tcn_batch_norm=False):
        super().__init__()

        layers = []

        if masking:
            print("MASKING — not implemented yet")

        num_tcn = len(dilations)
        print('num_tcn:', num_tcn)

        for i in range(num_tcn - 1):
            tcn_layer = TCN(
                input_size=num_feats if i == 0 else nb_filters,
                output_size=nb_filters,
                num_channels=[nb_filters]*nb_stacks,
                kernel_size=kernel_size,
                dropout=lstm_dropout,
                batch_norm=tcn_batch_norm,
                return_sequences=True
            )
            print(f"TCN {i}", dilations[i])
            layers.append(tcn_layer)

        final_layer = TCN(
            input_size=nb_filters,
            output_size=nb_filters,
            num_channels=[nb_filters]*nb_stacks,
            kernel_size=kernel_size,
            dropout=lstm_dropout,
            batch_norm=tcn_batch_norm,
            return_sequences=prediction_mode
        )
        layers.append(final_layer)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

    def get_embedding(self, x):
        return self.encoder(x)


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
                 prediction_mode=False,
                 lstm_decoder=False,
                 num_layers=None,
                 num_neurons=None,
                 tcn_batch_norm=False,
                 use_gru=False,
                 **kwargs):

        super(TCN_clf, self).__init__()

        # Parse conv_params
        if len(conv_params) == 4:
            nb_filters, kernel_size, nb_stacks, use_skip_connections = conv_params
            padding = 'causal'
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 5:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding = conv_params
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 6:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding, dilations = conv_params
            if isinstance(dilations, int):
                dilations = [dilations]
            elif isinstance(dilations, str):
                import ast
                dilations = ast.literal_eval(dilations)
            else:
                dilations = [[i for i in [1, 2, 4, 8, 16, 32] if i <= d]
                             for d in dilations]
        else:
            raise ValueError('conv_params length not recognised', len(conv_params))

        # Encoder
        self.encoder_net = EncoderTCN(num_feats=num_feats,
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

        self.triplet = triplet
        self.classification = classification

        # Optional classifier head
        if clf_neurons and clf_neurons != 0:
            self.clf_dense = nn.Linear(nb_filters, clf_neurons)
            self.clf_neurons = clf_neurons
        else:
            self.clf_dense = None
            self.clf_neurons = 0

        if classification:
            self.clf_out = nn.Linear(clf_neurons or nb_filters, num_classes)

    def forward(self, x):
        encoder_raw = self.encoder_net(x)

        if self.clf_dense is not None:
            encoder = F.relu(self.clf_dense(encoder_raw))
        else:
            encoder = encoder_raw

        out = []

        if self.triplet:
            emb = F.normalize(encoder, dim=-1)
            out.append(emb)

        if self.classification:
            clf = self.clf_out(encoder)
            out.append(clf)

        return out

    def get_embedding(self, x, batch=None, unify=False):
        with torch.no_grad():
            if batch is None or batch <= 0:
                emb = self.encoder_net(x)
                if self.clf_dense:
                    emb = F.relu(self.clf_dense(emb))
                emb = F.normalize(emb, dim=-1)
                return emb
            else:
                raise NotImplementedError("Batch mode get_embedding not implemented yet.")
