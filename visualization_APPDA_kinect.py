#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:10:29 2020

@author: asabater
"""

# --- Standard library imports ---
import argparse
import copy
import glob
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from collections import Counter
from functools import partial
from shutil import copyfile

# --- Third-party imports ---
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import (
    pyplot as plt,
    animation
)
from scipy.special import comb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# --- Local project imports ---
from models.TCN_classifier import TCN_clf
from pytorch_dataset_mp import (
    TripletPoseDataset,
    TherapyDataset,
    get_num_feats,
    get_scaler_filename,
)

# --- Framework-agnostic or to-be-adapted utility imports ---
# Ensure these functions do not have hard TensorFlow dependencies.
# It's assumed these are available in your environment, possibly from data_generator.py
# If data_generator.py has TF dependencies, these functions need to be extracted/rewritten.
# try:
#     from data_generator import get_scaler_filename
# except ImportError:
#     print("Warning: Could not import from data_generator.py. Ensure get_scaler_filename and get_num_feats are defined or TF-free.")
#     # Define placeholders if they are critical and not available, but this is not ideal
# 
#     def get_num_feats(
#         **kwargs): raise NotImplementedError("get_num_feats is not defined/imported")
# 
#     def get_scaler_filename(
#         **kwargs): raise NotImplementedError("get_scaler_filename is not defined/imported")


if __name__ == "__main__":
    raw_data_path = './datasets/therapies_dataset/'
    actions_data = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
    video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))

    actions_data = actions_data[~actions_data.action.isin(['no', 'si'])]
    actions_data = actions_data.sort_values(by=['patient', 'session', 'video', 'ex_num'])

    timestamps, frame_indices, skeletons = video_skels[actions_data['preds_filename'][1]]
    
    print("skeletons shape:", skeletons.shape)
    
    # Setup plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Skeleton Animation (First Sample)")

    # Define skeleton edges
    edges = [
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),
        (1, 0), (20, 2), (2, 3), (1, 20),
        (20, 8), (9, 10), (8, 9), (10, 24), (10, 11), (11, 23),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]
    
    # Apply remap: (oldZ → newX, oldX → newY, oldY → newZ)
    xs = skeletons[:, :, 0].flatten()
    ys = skeletons[:, :, 1].flatten()
    zs = skeletons[:, :, 2].flatten()

    print("[DEBUG] New X (old Z) range:", xs.min(), xs.max())
    print("[DEBUG] New Y (old X) range:", ys.min(), ys.max())
    print("[DEBUG] New Z (old Y) range:", zs.min(), zs.max())
    
    ax.set_xlim(xs.min(), xs.max())   # now showing old Z
    ax.set_ylim(ys.min(), ys.max())    # now showing old X
    ax.set_zlim(zs.min(), zs.max())    # now showing old Y
    
    # ax.set_xlim(-1, 0)   # now showing old Z
    # ax.set_ylim(-1.5, 0.3)    # now showing old X
    # ax.set_zlim(1.5, 3)    # now showing old Y


    #ax.set_xlim(-0.2, 0.2)   # now showing old Z
    #ax.set_ylim(-0.6, 0.2)    # now showing old X
    #ax.set_zlim(-0.6, 0.2)    # now showing old Y

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    scatter = ax.scatter([], [], [], c="blue", s=40)
    lines = [ax.plot([], [], [], c='black')[0] for _ in edges]

    def init():
        scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in lines:
            line.set_data(np.zeros(2), np.zeros(2))
            line.set_3d_properties(np.zeros(2))
        return [scatter] + lines

    def update(frame_idx):
        frame = skeletons[frame_idx]

        xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]  # your remapped X/Y/Z
        scatter._offsets3d = (xs, ys, zs)

        for k, (i, j) in enumerate(edges):
            x_vals = np.array([frame[i, 0], frame[j, 0]])
            y_vals = np.array([frame[i, 1], frame[j, 1]])  # old Z → new Y
            z_vals = np.array([frame[i, 2], frame[j, 2]])  # old Y → new Z
            
            lines[k].set_data(x_vals, y_vals)
            lines[k].set_3d_properties(z_vals)
            
        return [scatter] + lines

    ani = animation.FuncAnimation(fig, update, frames=skeletons.shape[0],
                                init_func=init, blit=True, interval=10)
    
    plt.show()
    
    exit()
    
    
    print("First video key:", first_key)

    first_video = video_skels[first_key]
    print("Type of data for this video:", type(first_video))

    # If it's an array, check shape
    if isinstance(first_video, np.ndarray):
        print("Shape:", first_video.shape)
        print("First 5 frames:\n", first_video[:5])

