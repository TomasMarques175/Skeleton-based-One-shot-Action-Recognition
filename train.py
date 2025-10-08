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

try:
    import train_utils  # For create_model_folder
except ImportError:
    print("Warning: train_utils not found. create_model_folder might fail.")

    class train_utils_placeholder:  # Basic placeholder
        @staticmethod
        def create_model_folder(path_results, model_name_base):
            from datetime import datetime
            ts = datetime.now().strftime("%m%d_%H%M%S")
            model_name_folder = f"{str(model_name_base)}_{ts}_run{np.random.randint(1000):03d}"
            path_model = os.path.join(path_results, model_name_folder)
            return path_model
    train_utils = train_utils_placeholder


# Seed PyTorch
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = False

def export_dataset_to_txt(annotations_file, model_params, output_file="dataset_dump.txt", max_samples=None):
    """
    Export the dataset features + labels to a .txt file.

    Args:
        annotations_file (str): Path to dataset annotations.
        model_params (dict): Dict with dataset/model params.
        output_file (str): Where to save the .txt file.
        max_samples (int): Limit number of samples (None = full dataset).
    """
    dataset = TripletPoseDataset(
        pose_annotations_file=annotations_file,
        validation_mode=False,
        in_memory=model_params.get("in_memory_generator_train", False),
        **model_params
    )

    with open(output_file, "w") as f:
        for i, (features, label) in enumerate(dataset):
            # Convert to numpy for easy saving
            features_np = features.numpy()
            label_val = int(label.item())

            f.write(f"Sample {i}, Label {label_val}\n")

            # Save each frame in the sequence
            for frame in features_np:
                line = " ".join(map(str, frame.tolist()))
                f.write(line + "\n")

            f.write("\n")  # Blank line between samples

            if max_samples and i + 1 >= max_samples:
                break

    print(f"✅ Dataset exported to {output_file}")

def animate_first_sample_skeleton_3d_upper_body(annotations_file, model_params):
    """
    Animate the 3D skeleton for all frames of the first sample.
    """
    # Load dataset
    dataset = TripletPoseDataset(
        pose_annotations_file=annotations_file,
        validation_mode=False,
        in_memory=model_params.get('in_memory_generator_train', False),
        **model_params
    )

        # Find the first sample with label 0
    features, label = None, None
    for i in range(len(dataset)):
        f, l = dataset[i]
        if l != 0: # Looking for label 1
            features, label = f, l
            print(f"Found sample with label 0 at index {i}")
            # break

        if features is None:
            raise ValueError("No sample with label 0 found in dataset!")

        print(f"Features shape: {features.shape}, label: {label}")

        offset = model_params.get("num_features", 24) + (model_params.get("num_features", 24) - 1) - 1 # Start of keypoints
        num_keypoints = model_params.get("num_features", 24) * 3
        joints_num = model_params.get("joints_num", 24)
        coords_dim = model_params.get("joints_dim", 3)
        
        # Setup plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Skeleton Animation (Upper Body)")

        # --- Define dropped joints ---
        drop_joints = {0, 1, 2, 3, 4, 5, 13, 19, 20, 21, 22, 23}
        keep_indices = [j for j in range(24) if j not in drop_joints]
        old_to_new = {old: new for new, old in enumerate(keep_indices)}

        print("Kept joints:", keep_indices)
        print("Old → New mapping:")
        for old, new in old_to_new.items():
            print(f"  {old:2d} → {new:2d}")

        # --- Connecting joints ---
        CONNECTING_JOINT_OLD = [
            1, 0, 1, 2, 2, 14, 3, 14, 3, 4, 4, 5,
            14, 16, 15, 16, 12, 15, 6, 7, 7, 8, 8, 12,
            9, 12, 9, 10, 10, 11, 12, 17, 17, 18, 18, 19,
            13, 19, 21, 23, 19, 21, 19, 20, 20, 22
        ]

        CONNECTING_JOINT = []
        remap_pairs = []  # for debug printing
        for i in range(0, len(CONNECTING_JOINT_OLD), 2):
            j1, j2 = CONNECTING_JOINT_OLD[i], CONNECTING_JOINT_OLD[i+1]
            if j1 in old_to_new and j2 in old_to_new:
                CONNECTING_JOINT.extend([old_to_new[j1], old_to_new[j2]])
                remap_pairs.append((j1, j2, old_to_new[j1], old_to_new[j2]))

        print("\nConnecting joints (old → new):")
        for old1, old2, new1, new2 in remap_pairs:
            print(f"  ({old1:2d}, {old2:2d}) → ({new1:2d}, {new2:2d})")

        print("\nFinal CONNECTING_JOINT:", CONNECTING_JOINT)

        # --- Extract upper body coordinates ---
        upper_coords = (
            features.numpy()[:, offset : offset + num_keypoints]
            .reshape(-1, 24, 3)[:, keep_indices, :]
        )  # (frames, 12, 3)

        # Apply remap: (oldZ → newX, oldX → newY, oldY → newZ)
        xs = upper_coords[:, :, 1].flatten()
        ys = upper_coords[:, :, 2].flatten()
        zs = upper_coords[:, :, 0].flatten()

        # print("[DEBUG] New X (old Z) range:", xs.min(), xs.max())
        # print("[DEBUG] New Y (old X) range:", ys.min(), ys.max())
        # print("[DEBUG] New Z (old Y) range:", zs.min(), zs.max())

        all_min = min(xs.min(), ys.min(), zs.min())
        all_max = max(xs.max(), ys.max(), zs.max())

        # ax.set_xlim(all_min, all_max)
        # ax.set_ylim(all_min, all_max)
        # ax.set_zlim(all_min, all_max)
        
        # ax.set_xlim(-0.1, 0.1)   # now showing old Z
        # ax.set_ylim(-0.3, 0.1)    # now showing old X
        # ax.set_zlim(-0.3, 0.1)    # now showing old Y
        ax.set_xlim(-0.3, 0.5)
        ax.set_ylim(-0.1, 0.5)
        ax.set_zlim(-0.15, 0.5)

        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_zlabel("X")

        scatter = ax.scatter([], [], [], c="blue", s=40)
        lines = [ax.plot([], [], [], c='black')[0] for _ in CONNECTING_JOINT]

        def init():
            scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
            for line in lines:
                line.set_data(np.zeros(2), np.zeros(2))
                line.set_3d_properties(np.zeros(2))
            return [scatter] + lines

        def update(frame_idx):
            coords = upper_coords[frame_idx]  # Already (12,3)
            xs, ys, zs = coords[:, 1], coords[:, 2], coords[:, 0]  # remapped axes
            scatter._offsets3d = (xs, ys, zs)

            for k in range(0, len(CONNECTING_JOINT), 2):
                i, j = CONNECTING_JOINT[k], CONNECTING_JOINT[k + 1]
                x_vals = np.array([coords[i, 1], coords[j, 1]])
                y_vals = np.array([coords[i, 2], coords[j, 2]])
                z_vals = np.array([coords[i, 0], coords[j, 0]])
                lines[k].set_data(x_vals, y_vals)
                lines[k].set_3d_properties(z_vals)

            return [scatter] + lines

        ani = animation.FuncAnimation(fig, update, frames=upper_coords.shape[0],
                                    init_func=init, blit=True, interval=1000)

        # Create folder if it doesn't exist
        save_dir = "animations"
        os.makedirs(save_dir, exist_ok=True)
        
        same_has = "gif"

        if same_has == "mp4":
            print("Saving animation as MP4...")
            save_path = os.path.join(save_dir, f"skeleton_animation_sample_{i}_label_{label-1}.mp4")
            ani.save(save_path, writer="ffmpeg", fps=10)

        elif same_has == "gif":
            print("Saving animation as GIF...")
            save_path = os.path.join(save_dir, f"skeleton_animation_sample_{i}_label_{label-1}.gif")
            ani.save(save_path, writer="pillow", fps=10)

        print(f"Animation saved to {save_path}")
        exit(0)
        # plt.show()

def animate_first_sample_skeleton_3d(annotations_file, model_params):
    """
    Animate the 3D skeleton for all frames of the first sample.
    """
    # Load dataset
    dataset = TripletPoseDataset(
        pose_annotations_file=annotations_file,
        validation_mode=False,
        in_memory=model_params.get('in_memory_generator_train', False),
        **model_params
    )

    # Find the first sample with label 0
    features, label = None, None
    for i in range(len(dataset)):
        f, l = dataset[i]
        if l != 0: # Looking for label 1
            features, label = f, l
            print(f"Found sample with label 0 at index {i}")
            # break

        if features is None:
            raise ValueError("No sample with label 0 found in dataset!")

        print(f"Features shape: {features.shape}, label: {label}")

        offset = model_params.get("num_features", 24) + (model_params.get("num_features", 24) - 1) - 1 # Start of keypoints
        num_keypoints = model_params.get("num_features", 24) * 3
        joints_num = model_params.get("joints_num", 24)
        coords_dim = model_params.get("joints_dim", 3)

        # Define skeleton edges
        edges = [
            (1, 0), (1, 2), (2, 14), (3, 14), (3, 4), (4, 5),
            (14, 16), (15, 16), (12, 15), (6, 7), (7, 8), (8, 12),
            (9, 12), (9, 10), (10, 11), (12, 17), (17, 18), (18, 19),
            (13, 19), (21, 23), (19, 21), (19, 20), (20, 22)
        ]

        # Setup plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Skeleton Animation (First Sample)")
        # Debug: check min/max ranges after remapping (Z, X, Y)
        all_coords = features.numpy()[:, offset: offset + 72].reshape(-1, joints_num, coords_dim)

        # Apply remap: (oldZ → newX, oldX → newY, oldY → newZ)
        xs = all_coords[:, :, 1].flatten()
        ys = all_coords[:, :, 2].flatten()
        zs = all_coords[:, :, 0].flatten()

        print("[DEBUG] New X (old Z) range:", xs.min(), xs.max())
        print("[DEBUG] New Y (old X) range:", ys.min(), ys.max())
        print("[DEBUG] New Z (old Y) range:", zs.min(), zs.max())

        all_min = min(xs.min(), ys.min(), zs.min())
        all_max = max(xs.max(), ys.max(), zs.max())

        # ax.set_xlim(all_min, all_max)
        # ax.set_ylim(all_min, all_max)
        # ax.set_zlim(all_min, all_max)
        
        ax.set_xlim(-0.2, 0.2)   # now showing old Z
        ax.set_ylim(-0.6, 0.2)    # now showing old X
        ax.set_zlim(-0.6, 0.2)    # now showing old Y

        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_zlabel("X")

        scatter = ax.scatter([], [], [], c="blue", s=40)
        lines = [ax.plot([], [], [], c='black')[0] for _ in edges]

        def init():
            scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
            for line in lines:
                line.set_data(np.zeros(2), np.zeros(2))
                line.set_3d_properties(np.zeros(2))
            return [scatter] + lines

        def update(frame_idx):
            frame = features[frame_idx].numpy()
            coords = frame[offset: offset + num_keypoints].reshape(joints_num, coords_dim)
            xs, ys, zs = coords[:, 1], coords[:, 2], coords[:, 0]  # your remapped X/Y/Z
            scatter._offsets3d = (xs, ys, zs)

            for k, (i, j) in enumerate(edges):
                x_vals = np.array([coords[i, 1], coords[j, 1]])
                y_vals = np.array([coords[i, 2], coords[j, 2]])  # old Z → new Y
                z_vals = np.array([coords[i, 0], coords[j, 0]])  # old Y → new Z
                
                lines[k].set_data(x_vals, y_vals)
                lines[k].set_3d_properties(z_vals)
            return [scatter] + lines

        ani = animation.FuncAnimation(fig, update, frames=features.shape[0],
                                    init_func=init, blit=True, interval=1000)

        # Create folder if it doesn't exist
        save_dir = "animations"
        os.makedirs(save_dir, exist_ok=True)
        
        same_has = "gif"

        if same_has == "mp4":
            print("Saving animation as MP4...")
            save_path = os.path.join(save_dir, f"skeleton_animation_sample_{i}_label_{label-1}.mp4")
            ani.save(save_path, writer="ffmpeg", fps=10)

        elif same_has == "gif":
            print("Saving animation as GIF...")
            save_path = os.path.join(save_dir, f"skeleton_animation_sample_{i}_label_{label-1}.gif")
            ani.save(save_path, writer="pillow", fps=10)

        print(f"Animation saved to {save_path}")
        # plt.show()
        exit(0)

def visualize_first_sample_skeleton_3d(annotations_file, model_params):
    """
    Visualize the 3D skeleton joints for the first frame of the first sample.
    Assumes coordinates start after the first 47 features.
    """

    # 1. Load dataset
    dataset = TripletPoseDataset(
        pose_annotations_file=annotations_file,
        validation_mode=False,
        in_memory=model_params.get('in_memory_generator_train', False),
        **model_params
    )

    # 2. Get first sample
    features, label = dataset[39]  # (seq_len, num_feats)
    print(f"Features shape: {features.shape}, label: {label}")

    first_frame = features[12].numpy()

    print(f"First frame shape: {first_frame.shape}")
    print(f"First frame data (first 72 values): {first_frame[:72]}")
    
    # 3. Extract coordinates (start after 47 features)
    offset = 46
    joints_num = model_params.get("joints_num", 24)
    coords_dim = model_params.get("joints_dim", 3)

    # 4. Plot all frames in sequence
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Skeleton Joints (All Frames, First Sample)")
    ax.set_xlim(-0.3, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.15, 0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Skeleton edges (example, adjust to your dataset)
    edges = [
        (1, 0), (1, 2), (2, 14), (3, 14), (3, 4), (4, 5),
        (14, 16), (15, 16), (12, 15), (6, 7), (7, 8), (8, 12),
        (9, 12), (9, 10), (10, 11), (12, 17), (17, 18), (18, 19),
        (13, 19), (21, 23), (19, 21), (19, 20), (20, 22)
    ]
    
    # Loop over frames
    coords = first_frame[offset : offset + 72].reshape(joints_num, coords_dim)
    
    # Scatter joints
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="blue", s=20, alpha=0.5)
    
    # Draw skeleton lines
    for i, j in edges:
        if i < len(coords) and j < len(coords):
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    c="black", alpha=0.3)

    plt.show()

def visualize_pose_dataset_2d(annotations_file_1, annotations_file_2, model_params, max_samples=2000, method="pca"):
    """
    Compare two pose datasets by projecting their features into 2D using PCA or t-SNE.

    annotations_file_1 / annotations_file_2 : str
        Paths to annotation files for each dataset.
    model_params : dict
        Dictionary of model parameters passed to TripletPoseDataset.
    max_samples : int
        Maximum number of samples to visualize across both datasets.
    method : str
        "pca" or "tsne" for dimensionality reduction.
    """
    
    # 1. Load datasets
    dataset1 = TripletPoseDataset(
        pose_annotations_file=annotations_file_1,
        validation_mode=False,
        in_memory=model_params.get('in_memory_generator_train', False),
        **model_params
    )
    dataset2 = TripletPoseDataset(
        pose_annotations_file=annotations_file_2,
        validation_mode=False,
        in_memory=model_params.get('in_memory_generator_train', False),
        **model_params
    )

    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=True)

    # 2. Collect features
    def collect_features(loader, label_value, limit):
        features_list, labels_list = [], []
        total = 0
        for features, _ in loader:
            pooled = features.mean(dim=1).numpy()  # mean pooling
            features_list.append(pooled)
            labels_list.append(np.full(pooled.shape[0], label_value))
            total += pooled.shape[0]
            if total >= limit:
                break
        return np.vstack(features_list), np.concatenate(labels_list)

    half_samples = max_samples // 2
    X1, y1 = collect_features(loader1, label_value=0, limit=half_samples)
    X2, y2 = collect_features(loader2, label_value=1, limit=half_samples)

    all_features = np.vstack([X1, X2])
    all_labels = np.concatenate([y1, y2])

    # 3. Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    features_2d = reducer.fit_transform(all_features)

    # 4. Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=all_labels, cmap="coolwarm", alpha=0.7
    )
    plt.colorbar(scatter, ticks=[0, 1], label="Dataset Origin")
    plt.title(f"2D {method.upper()} Comparison of Two Pose Datasets")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def copy_scaler_if_needed(model_params):
    """
    Copy the scaler file to the model output directory if scaling is enabled.
    Uses `get_scaler_filename(**model_params)` to find the source file.

    Args:
        model_params (dict): Dictionary containing model configuration, including 'scale_data' and 'path_model'.
    """
    if model_params.get('scale_data', False):
        try:
            scaler_filename_src = get_scaler_filename(**model_params)
            if scaler_filename_src:
                scaler_filename_dst = os.path.join(
                    model_params['path_model'], 'scaler.pckl')
                if os.path.exists(scaler_filename_src):
                    copyfile(scaler_filename_src, scaler_filename_dst)
                    print(f"Copied scaler: {scaler_filename_src} to {scaler_filename_dst}")
                else:
                    print(f"Warning: Scaler source file '{scaler_filename_src}' not found. Not copied.")
            else:
                print("Warning: get_scaler_filename did not return a path. Scaler not copied.")
        except Exception as e:
            print(f"Warning: Error handling scaler file: {e}. Skipping scaler copy.")

def get_final_model_path(path_results, model_name, model_number):
    model_dir = os.path.join(path_results, model_name)
    
    # Look for subfolders that contain the model number
    for subfolder in os.listdir(model_dir):
        if f"_model_{model_number}" in subfolder:
            weights_dir = os.path.join(model_dir, subfolder, "weights")
            
            for file in os.listdir(weights_dir):
                if file.endswith(".pt"):
                    return os.path.join(weights_dir, file)

    raise FileNotFoundError(f"No .pt file found for model {model_number} in {model_dir}")

def train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, \
    device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch=0, train_verbose=1, log_interval=100):
    
    for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
        features_batch = features_batch.to(device, non_blocking=True)
        labels_batch = labels_batch.to(
            device, non_blocking=True)  # Integer class labels
        optimizer.zero_grad()

        # Forward pass
        # Model output: [embedding, clf_logits] if both active, or just one of them
        model_output = pytorch_model(features_batch)

        current_batch_total_loss = 0.0
        embeddings_batch = None
        clf_logits_batch = None

        if model_params.get('triplet', False) and model_params.get('classification', True):
            if isinstance(model_output, list) and len(model_output) == 2:
                embeddings_batch, clf_logits_batch = model_output
            else:
                print(
                    "Warning: Model output format unexpected for triplet+classification. Skipping batch.")
                continue
        elif model_params.get('triplet', False):
            embeddings_batch = model_output  # Assumes model returns only embeddings
        elif model_params.get('classification', True):
            clf_logits_batch = model_output  # Assumes model returns only clf_logits
        else:
            print(
                "Warning: Neither triplet nor classification is enabled. No loss to compute. Skipping batch.")
            continue

        if clf_logits_batch is not None:
            with torch.no_grad():
                probs = torch.softmax(clf_logits_batch, dim=1)
                softmax_outputs.append(probs.cpu().numpy())

        # Calculate Classification Loss
        if 'classification' in active_losses and clf_logits_batch is not None:
            # Ensure labels_batch is of type LongTensor for CrossEntropyLoss
            loss_c = active_losses['classification'](
                clf_logits_batch, labels_batch.to(device).long())
            current_batch_total_loss += loss_weights_pytorch_pt['classification'] * loss_c
            running_train_loss_clf += loss_c.item()
            # if batch_idx == 0 and epoch == 0:
            #     print(
            #         f"  Classification loss component active. Example batch loss_c: {loss_c.item():.4f}")

        # Calculate Triplet Loss (Requires Batch Mining)
        if 'triplet' in active_losses and embeddings_batch is not None:
            # TODO: NOT IMPLEMENTED: Triplet loss mining logic
            # Placeholder for triplet loss calculation
            if batch_idx == 0 and epoch == 0:
                print(
                    "  Triplet loss component active, but MINING LOGIC IS A PLACEHOLDER.")
            pass  # Placeholder for triplet loss calculation

        if isinstance(current_batch_total_loss, torch.Tensor):  # Check if any loss was added
            current_batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pytorch_model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            running_train_loss += current_batch_total_loss.item()
        elif current_batch_total_loss == 0.0 and not (model_params.get('triplet', False) or model_params.get('classification', True)):
            pass  # No losses active, normal
        else:
            if train_verbose > 0:
                print(
                    f"  Train Batch: {batch_idx+1}/{len(train_loader)} - No loss computed for this batch.")
            continue

        # if train_verbose > 0 and batch_idx % log_interval == 0 and isinstance(current_batch_total_loss, torch.Tensor):
        #     print(
        #         f"  Train Batch: {batch_idx+1}/{len(train_loader)} Loss: {current_batch_total_loss.item():.4f}")
    return running_train_loss, running_train_loss_clf

def validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt):
    """
    Validate the model on the validation dataset.
    Args:
        model_params (dict): Dictionary containing model configuration.
        device (torch.device): Device to run the validation on.
        val_loader (DataLoader): DataLoader for the validation dataset.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.
        epoch (int, optional): Current epoch number for logging.
    Returns:
        float: Average validation loss.
    """
    val_metrics = {}  # To store metrics like loss, accuracy
    if val_loader:
        pytorch_model.eval()
        running_val_loss = 0.0
        running_val_loss_clf = 0.0
        running_val_loss_triplet = 0.0
        all_clf_preds_val = []
        all_clf_labels_val = []
        all_clf_probs_val = []

        with torch.no_grad():
            for features_batch_val, labels_batch_val in val_loader:
                features_batch_val = features_batch_val.to(
                    device, non_blocking=True)
                labels_batch_val = labels_batch_val.to(
                    device, non_blocking=True)

                model_output_val = pytorch_model(features_batch_val)
                current_batch_val_total_loss = 0.0
                embeddings_batch_val = None
                clf_logits_batch_val = None

                if model_params.get('triplet', False) and model_params.get('classification', True):
                    if isinstance(model_output_val, list) and len(model_output_val) == 2:
                        embeddings_batch_val, clf_logits_batch_val = model_output_val
                    else:
                        continue  # Skip if format is wrong
                elif model_params.get('triplet', True):
                    embeddings_batch_val = model_output_val
                elif model_params.get('classification', True):
                    clf_logits_batch_val = model_output_val

                if 'classification' in active_losses and clf_logits_batch_val is not None:
                    loss_c_val = active_losses['classification'](
                        clf_logits_batch_val, labels_batch_val)
                    current_batch_val_total_loss += loss_weights_pytorch_pt['classification'] * loss_c_val
                    running_val_loss_clf += loss_c_val.item()

                    # (batch_size, num_classes)
                    probs_val = torch.softmax(clf_logits_batch_val, dim=1)
                    # Collect probabilities
                    all_clf_probs_val.extend(probs_val.cpu().numpy())
                    _, predicted_indices = torch.max(probs_val, 1)
                    all_clf_preds_val.extend(
                        predicted_indices.cpu().numpy())
                    all_clf_labels_val.extend(
                        labels_batch_val.cpu().numpy())

                if 'triplet' in active_losses and embeddings_batch_val is not None:
                    triplet_loss_val = active_losses['triplet'](
                        embeddings_batch_val, labels_batch_val)
                    current_batch_val_total_loss += loss_weights_pytorch_pt['triplet'] * \
                        triplet_loss_val
                    running_val_loss_triplet += triplet_loss_val.item()

                if isinstance(current_batch_val_total_loss, torch.Tensor):
                    running_val_loss += current_batch_val_total_loss.item()

    return val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, all_clf_preds_val, all_clf_labels_val, all_clf_probs_val

def Setup_optimizer_and_loss(pytorch_model, model_params, device, train_dataset=None):
    """ Set up the optimizer and loss functions for the PyTorch model.
    Args:
        model_params (dict): Dictionary containing model configuration.
    """
    # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
    # print(' * Defining losses and loss_weights (PyTorch)')
    active_losses = {}
    loss_weights_pytorch_pt = {}
    if model_params.get('classification', True):
        # Extract all labels from the training dataset
        train_labels_from_dataset = [label for _, label in train_dataset]
        # Convert tensors to integers if needed
        train_labels_from_dataset = [label.item() if torch.is_tensor(label) else label for label in train_labels_from_dataset]

        # Infer number of classes from dataset
        num_classes_model = model_params['num_classes']  # full size, e.g. 14

        # Count samples per class
        counts = np.bincount(train_labels_from_dataset, minlength=num_classes_model)
        total = counts.sum()

        # Compute inverse-frequency weights (avoid divide by zero)
        weights = []
        for i in range(num_classes_model):
            if counts[i] > 0:
                weights.append(total / (num_classes_model * counts[i]))
            else:
                weights.append(0.0)

        # Convert to tensor directly on device
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        # Use in loss function
        criterion_clf = nn.CrossEntropyLoss(weight=weights_tensor)
        active_losses['classification'] = criterion_clf

        # Match keras loss_weights['output_1'] = 0.4 if classification is the primary/first output
        loss_weights_pytorch_pt['classification'] = model_params.get('clf_loss_weight', 0.4)
    if model_params.get('triplet', False):  # If you also had triplet loss in Keras
        criterion_triplet_pt = nn.TripletMarginLoss(
            margin=model_params.get('triplet_margin', 1.0))
        active_losses['triplet'] = criterion_clf
        loss_weights_pytorch_pt['triplet'] = model_params.get(
            'triplet_loss_weight', 0.6)  # Example
        print(' * losses (PyTorch types):', active_losses)
        print(' * loss_weights (PyTorch):', loss_weights_pytorch_pt)
        # sample_weights_mode is not a direct PyTorch concept, handled manually if needed

    # print('\n * Setting optimizer (PyTorch)')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pytorch_model.parameters()),
                    lr=model_params['init_lr'])
    # Note: clipnorm is applied manually in PyTorch training loop (torch.nn.utils.clip_grad_norm_)
    # print(f"   Optimizer: {type(optimizer)}, LR: {model_params['init_lr']}")


    return active_losses, loss_weights_pytorch_pt, optimizer

def get_best_model_path(model_path_or_folder):
    # If a full .pt file is given, just return it directly
    if os.path.isfile(model_path_or_folder) and model_path_or_folder.endswith(".pt"):
        return model_path_or_folder

    # Otherwise, search in the folder for the .pt file with the best F1 score
    best_f1 = -1.0
    best_file = None

    for filename in os.listdir(model_path_or_folder):
        if filename.endswith(".pt"):
            match = re.search(r"f1([0-9.]+)", filename)
            if match:
                f1 = float(match.group(1))
                if f1 > best_f1:
                    best_f1 = f1
                    best_file = filename

    if best_file:
        return os.path.join(model_path_or_folder, best_file)
    else:
        raise FileNotFoundError("No valid model with F1 score found in the folder.")

def get_k_fold_model_path(model_path_or_folder, fold=None):
    # If a full .pt file is given, just return it directly
    if os.path.isfile(model_path_or_folder) and model_path_or_folder.endswith(".pt"):
        return model_path_or_folder

    # Step into the only subfolder (e.g., 0730_1921_model_1)
    subdirs = [d for d in os.listdir(model_path_or_folder) if os.path.isdir(os.path.join(model_path_or_folder, d))]
    if not subdirs:
        raise FileNotFoundError("No model subdirectory found.")
    
    model_subdir = os.path.join(model_path_or_folder, subdirs[0], "weights")

    best_f1 = -1.0
    best_file = None

    for filename in os.listdir(model_subdir):
        if not filename.endswith(".pt"):
            continue
        if fold is not None and f"fold_{fold}" not in filename:
            continue

        match = re.search(r"f1([0-9.]+)", filename)
        if match:
            f1 = float(match.group(1))
            if f1 > best_f1:
                best_f1 = f1
                best_file = filename

    if best_file:
        return os.path.join(model_subdir, best_file)
    else:
        raise FileNotFoundError(f"No valid model found for fold {fold} in {model_subdir}")

def save_average_k_fold_model(model_params, weights_save_path, model_class, save_name_prefix="average_k_fold", device="cpu"):
    """
    Averages all .pt models in all 'weights' folders under checkpoint_path and saves the result.

    Args:
        checkpoint_path (str): Base path where subfolders containing 'weights/*.pt' files exist.
        model_class (callable): Class or function that returns a new model instance.
        save_name_prefix (str): Prefix for the saved averaged model.
        device (str): 'cpu' or 'cuda'.
    """
    pt_paths = []

    # 1. Traverse weights folder and collect .pt model paths
    for filename in os.listdir(weights_save_path):
        if filename.endswith(".pt"):
            pt_paths.append(os.path.join(weights_save_path, filename))

    if not pt_paths:
        raise FileNotFoundError("No .pt model files found in any weights subfolder.")

    # 2. Load and average the state dicts
    avg_state_dict = None
    f1_scores = []

    for path in pt_paths:
        state_dict = torch.load(path, map_location=device)

        match = re.search(r"f1([0-9.]+)", os.path.basename(path))
        if match:
            f1_scores.append(float(match.group(1)))

        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k]

    for k in avg_state_dict:
        avg_state_dict[k] /= len(pt_paths)

    # 3. Create averaged model from the first saved model
    model = model_class(
        num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
        lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
        triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
        clf_neurons=model_params['clf_neurons'], num_classes=model_params['num_classes']
    )
    model.load_state_dict(avg_state_dict)  # replace with averaged weights
    model.to(device)

    # 4. Save model with F1 in name inside weights folder
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    save_name = f"{save_name_prefix}-f1{avg_f1:.5f}.pt"

    # Full save path inside weights folder
    save_path = os.path.join(weights_save_path, save_name)

    torch.save(model.state_dict(), save_path)
    print(f"\nAveraged model saved to: {save_path}")

def get_average_k_fold_model_path(model_path_or_folder):
    """
    Searches for the averaged k-fold model with highest F1 score inside 'weights' folders
    under subdirectories of the given path.
    """
    if os.path.isfile(model_path_or_folder) and model_path_or_folder.endswith(".pt"):
        return model_path_or_folder

    best_f1 = -1.0
    best_file_path = None

    # Traverse subdirectories and look for average_k_fold models
    for subdir in os.listdir(model_path_or_folder):
        weights_dir = os.path.join(model_path_or_folder, subdir, "weights")
        if not os.path.isdir(weights_dir):
            continue

        for filename in os.listdir(weights_dir):
            if "average_k_fold" in filename and filename.endswith(".pt"):
                match = re.search(r"f1([0-9]+\.[0-9]+)(?=\.|_|$)", filename)
                if match:
                    f1 = float(match.group(1))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_file_path = os.path.join(weights_dir, filename)

    if best_file_path:
        return best_file_path
    else:
        raise FileNotFoundError("No averaged k-fold model file found.")

def create_pytorch_model(model_params, fold=None):
    # --- PyTorch Model Instantiation ---
    # print('\n* Setting model parameters (PyTorch)')  # Mimicking Keras log
    # Ensure num_classes is correctly derived if dataset re-indexing happened,
    # though for this debug block, it might not matter if we don't train.
    num_classes_for_model = model_params.get(
        'actual_num_classes', model_params['num_classes'])
    initial_state_dict = None  # Initialize to None, will be set later
    
    # If the model is in TensorFlow/Keras format, we will convert it to PyTorch.
    if model_params.get('model_converter', False) and \
        not model_params.get('model_is_pytorch', False):
        
        # --- PyTorch Model Instantiation ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )
        
        """
        print("\n=== PyTorch state_dict keys ===")
        for k, v in pytorch_model.state_dict().items():
            print(f"{k} {tuple(v.shape)}")
        """
        
        # ------------------------------
        # 2. Load the TensorFlow SavedModel
        tf_model = tf.saved_model.load(model_params["pretrained_model_path"])
        
        """
        print("\n=== TensorFlow variables ===")
                for var in tf_model.variables:
                    print(f"{var.name} {tuple(var.shape)}")
        """
                # --- 3. Extract TensorFlow weights to numpy dict ---
        
        tf_weights = {}
        for var in tf_model.variables:
            # Convert tensor to numpy
            tf_weights[var.name] = var.numpy()
        #    print(var.name, var.shape)

        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])
        
        # Define which keys are replaced with new ones (i.e., re-initialized)
        excluded_tf_prefixes = model_params.get('excluded_tf_prefixes', [])

        """ 
        # Filter and display PyTorch parameters
        included_pt_params = [(name, p.shape) for name, p in pytorch_model.named_parameters() if name not in excluded_pt_keys]
        total_pt_params = sum(p.numel() for name, p in pytorch_model.named_parameters() if name not in excluded_pt_keys)

        print(f"\nTotal PyTorch parameters (excluding excluded layers): {total_pt_params}")
        print("[Included PyTorch parameter keys with shapes:]")
        for name, shape in included_pt_params:
            print(f"  {name}: {tuple(shape)}") 
        """

        # Filter and display TensorFlow weights
        tf_weights_filtered = {
            k: v for k, v in tf_weights.items()
            if not any(k.startswith(prefix) for prefix in excluded_tf_prefixes)
        }
        # total_tf_params = sum(np.prod(v.shape) for v in tf_weights_filtered.values())

        #print(f"\nTotal TF parameters from checkpoint (excluding excluded layers): {total_tf_params}")
        #print("[Included TensorFlow keys with shapes:]")
        #for k, v in tf_weights_filtered.items():
        #    print(f"  {k}: {v.shape}")
        
        # --- 4. Get PyTorch state dict ---
        pt_state_dict = pytorch_model.state_dict()

        # --- 5. Convert TensorFlow weights to PyTorch ---
        converted_weights = convert_tf_to_torch(tf_weights_filtered, pt_state_dict)

        # --- 6. Update PyTorch state dict and load weights ---
        pt_state_dict.update(converted_weights)
        pytorch_model.load_state_dict(pt_state_dict)
        
        # Validate weights transferred correctly
        with torch.no_grad():
            for name, pt_w in pytorch_model.named_parameters():
                if name in converted_weights:
                    tf_w = converted_weights[name]
                    if pt_w.shape != tf_w.shape:
                        print(f"[MISMATCH] {name}: shape mismatch {pt_w.shape} vs {tf_w.shape}")
                        continue
                    diff = torch.abs(pt_w - tf_w).mean().item()
                    #print(f"{name}: mean abs diff = {diff:.6f}")
                # else:
                #     print(f"[SKIP] {name}: not found in converted weights")

        """         
        converted_param_count = sum(
            v.numel() for k, v in converted_weights.items() if k not in excluded_pt_keys
        )
        print(f"Total converted parameters (excluding excluded layers): {converted_param_count}")
        print(f"Coverage: {converted_param_count} / {total_pt_params} = {converted_param_count / total_pt_params:.2%}")
        """
        
        # --- 7. Freeze unconverted parameters ---
        #print("\n=== Freezing unconverted parameters ===")
        # Freeze all parameters that are in the excluded list
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
        
        # for name, param in pytorch_model.named_parameters():
        #     print(f"{name}: {'❄️ Frozen' if not param.requires_grad else '🔥 Trainable'}")
        
        initial_state_dict = copy.deepcopy(pytorch_model.state_dict())

    if model_params.get('model_converter', False) and \
        model_params.get('model_is_pytorch', False) and \
            model_params.get("use_pretrained_model", False) and \
                model_params.get('fine_tunning', False) and \
                    not model_params.get('average_k_fold', False):
        
        checkpoint_path = model_params["pretrained_model_path"]
        fold_model_path = get_k_fold_model_path(checkpoint_path, fold)
        
        # Load old model checkpoint
        old_model = TCN_clf(
            num_feats=model_params['old_model_input_feature_size'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )
        # Use the original params used to train it
        old_model.load_state_dict(torch.load(fold_model_path), strict=False)

        # Load new model
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )

        for (name_old, param_old), (name_new, param_new) in zip(old_model.named_parameters(), pytorch_model.named_parameters()):
            if param_old.shape != param_new.shape:
                print(f"Layer {name_old} changed: old {param_old.shape}, new {param_new.shape}")

        # Load compatible weights
        old_dict = old_model.state_dict()
        new_dict = pytorch_model.state_dict()
        filtered_dict = {}

        for k, v in old_dict.items():
            if k in new_dict and v.shape == new_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping {k}: checkpoint {v.shape} vs new model {new_dict[k].shape}")

        # Update new model's state_dict
        new_dict.update(filtered_dict)
        pytorch_model.load_state_dict(new_dict)
        
        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])

        # Freeze only the layers you want
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
                # print(f"Froze: {name}")
            else:
                param.requires_grad = True  # Unfrozen by default

        # Capture the initial state_dict *after freezing*
        initial_state_dict = {k: v.clone().detach().cpu() for k, v in pytorch_model.state_dict().items()}

    # If not converting or using a pre-trained model, initialize a new model
    if not model_params.get('model_converter', False) and \
        not model_params.get("use_pretrained_model", False):
        
        # If not converting or using a pre-trained model, initialize a new model
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )
    
    # If the model is in Pytorch format, we assume it has been converted or is ready to be used.
    if not model_params.get('model_converter', False) and \
        model_params.get("use_pretrained_model", False) and \
            not model_params.get('fine_tunning', False) and \
                not model_params.get('average_k_fold', False):

        # --- PyTorch Model Instantiation ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )

        # After loading the model with the pretrained weights
        checkpoint_path = model_params["pretrained_model_path"]
        best_model_path = get_best_model_path(checkpoint_path)
        pytorch_model.load_state_dict(torch.load(best_model_path))

        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])

        # Freeze only the layers you want
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
                # print(f"Froze: {name}")
            else:
                param.requires_grad = True  # Unfrozen by default

        # Capture the initial state_dict *after freezing*
        initial_state_dict = {k: v.clone().detach().cpu() for k, v in pytorch_model.state_dict().items()}

    # If the model is in Pytorch format, we assume it has been converted or is ready to be used.
    if not model_params.get('model_converter', False) and \
        model_params.get("use_pretrained_model", False) and \
            model_params.get('fine_tunning', False) and \
                not model_params.get('average_k_fold', False):
        
        # --- PyTorch Model Instantiation ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )

        # After loading the model with the pretrained weights
        checkpoint_path = model_params["pretrained_model_path"]
        best_model_path = get_k_fold_model_path(checkpoint_path, fold)
        pytorch_model.load_state_dict(torch.load(best_model_path))

        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])

        # Freeze only the layers you want
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Unfrozen by default

        # Capture the initial state_dict *after freezing*
        initial_state_dict = {k: v.clone().detach().cpu() for k, v in pytorch_model.state_dict().items()}

    # If the model is in Pytorch format, we assume it has been converted or is ready to be used.
    if not model_params.get('model_converter', False) and \
        model_params.get("use_pretrained_model", False) and \
            model_params.get('fine_tunning', False) and \
                model_params.get('average_k_fold', False):
        
        # --- PyTorch Model Instantiation ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )

        # After loading the model with the pretrained weights
        checkpoint_path = model_params["pretrained_model_path"]
        best_model_path = get_average_k_fold_model_path(checkpoint_path)
        pytorch_model.load_state_dict(torch.load(best_model_path))

        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])

        # Freeze only the layers you want
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
                # print(f"Froze: {name}")
            else:
                param.requires_grad = True  # Unfrozen by default

        # Capture the initial state_dict *after freezing*
        initial_state_dict = {k: v.clone().detach().cpu() for k, v in pytorch_model.state_dict().items()}
    
    
    return pytorch_model, initial_state_dict

def Setup_training(model_params, pytorch_model, device, optimizer):

    # print("\n--- Setting up PyTorch Training ---")
    monitor_metric_name = model_params.get('monitor', 'val_loss')
    monitor_is_loss = 'loss' in monitor_metric_name.lower()
    # Keras min_monitor=False means maximize. PyTorch mode='max'.
    # Keras min_monitor=True means minimize. PyTorch mode='min'.
    # Default to minimizing if loss, maximizing otherwise, unless min_monitor overrides.
    default_mode_for_metric = 'min' if monitor_is_loss else 'max'
    if 'min_monitor' in model_params:  # User explicitly set min_monitor
        monitor_mode = 'min' if model_params['min_monitor'] else 'max'
    else:  # Infer from metric name
        monitor_mode = default_mode_for_metric

    best_monitor_metric_val = float(
        '-inf') if monitor_mode == 'max' else float('inf')
    weights_save_path = os.path.join(model_params['path_model'], 'weights')
    os.makedirs(weights_save_path, exist_ok=True)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode=monitor_mode,
                                                        factor=model_params.get(
                                                            'lr_factor', 0.1),
                                                        patience=model_params.get(
                                                            'lr_patience', 25), # 3
                                                        min_lr=model_params.get('lr_min_delta', 1e-5)) # 1e-7
    early_stopping_patience = model_params.get('es_patience', 50) # 6
    min_delta = model_params.get("early_stopping_min_delta", 0.001)
    early_stopping_counter = 0

    return monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter

def convert_tf_to_torch(tf_weights, pt_state_dict):
    import re
    import torch

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    converted = {}
    matched_keys = set()
    unmatched_keys = []
    available_tf_keys = set(tf_weights.keys())

    for k in pt_state_dict.keys():
        if k.startswith("encoder_net.encoder.0.residual_blocks."):
            # Extract block index and layer info
            m = re.match(
                r"encoder_net\.encoder\.0\.residual_blocks\.(\d+)\.(\w+)(?:\.(weight|bias))?", k)
            if not m:
                print(f"{YELLOW}[WARN]{RESET} Couldn't parse key: {k}")
                unmatched_keys.append(k)
                continue

            block_idx = int(m.group(1))
            layer_name = m.group(2)
            param_type = m.group(3)

            tf_prefix = f"encoder_tcn/sequential/tcn/residual_block_{block_idx}"

            try:
                tf_key = None

                if block_idx == 0:
                    if layer_name == "conv1":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_0/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_0/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "conv2":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_1/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_1/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "downsample":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/matching_conv1D/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/matching_conv1D/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                else:
                    if layer_name == "conv1":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_0/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_0/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "conv2":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_1/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_1/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                if tf_key is not None:
                    matched_keys.add(k)
                    # print(f"{GREEN}[OK]{RESET} Converted {k} from {tf_key}")

            except KeyError:
                # print(f"{RED}[ERROR]{RESET} Missing TF key: {tf_key}")
                unmatched_keys.append(k)

    all_keys = set(pt_state_dict.keys())
    unmatched_keys = sorted(all_keys - matched_keys)

    # for k in unmatched_keys:
    #     print(f"{YELLOW}[UNMATCHED]{RESET} Did not match {k}")

    # print(f"\n[SUMMARY]")
    # print(f"{GREEN}Matched keys: {len(matched_keys)}{RESET}")
    # print(f"{RED if unmatched_keys else GREEN}Unmatched keys: {len(unmatched_keys)}{RESET}")

    return converted

def init_weights_constant(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.constant_(m.weight, 0.01)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0.01)

def collate_fn_classification_pre_pad(batch):
    """
    Pads variable-length sequences on the left (pre-padding).
    Assumes batch = [(sample, label), ...] where:
    - sample: Tensor of shape [seq_len, feat_dim]
    - label: int or tensor
    """
    samples, labels = zip(*batch)  # Unzip list of tuples
    lengths = [s.shape[0] for s in samples]
    max_len = max(lengths)
    feat_dim = samples[0].shape[1]

    padded_samples = []
    for s in samples:
        pad_len = max_len - s.shape[0]
        pad_tensor = torch.zeros((pad_len, feat_dim), dtype=s.dtype)
        padded = torch.cat((pad_tensor, s), dim=0)  # 👈 Pre-padding
        padded_samples.append(padded)

    padded_samples = torch.stack(padded_samples)
    labels = torch.tensor(labels)

    return padded_samples, labels

def Data_Loader_Classification(model_params, train_data, val_data, test_data, batch_size, device):
    # TODO: MP dataset
    # --- Data Loading ---
    print("\n--- Setting up PyTorch DataLoaders ---")
    dataset_params_for_loader = model_params.copy()
    try:
        train_dataset = TripletPoseDataset(pose_annotations_file=model_params['train_annotations'],
                                        validation_mode=False,
                                        in_memory=model_params['in_memory_generator_train'],
                                        **dataset_params_for_loader)
        train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                num_workers=model_params.get(
                                    'num_workers', 0),
                                pin_memory=True if device.type == 'cuda' else False, drop_last=False)
        print(
            f"Train DataLoader: Batches per epoch approx {len(train_loader)}")
    except Exception as e:
        print(f"CRITICAL Error creating train_dataset or train_loader: {e}")
        import traceback
        traceback.print_exc()
        return

    val_loader = None
    if model_params.get('val_annotations') and model_params['val_annotations'] != '':
        try:
            val_dataset = TripletPoseDataset(pose_annotations_file=model_params['val_annotations'],
                                            validation_mode=True,
                                            in_memory=model_params['in_memory_generator_val'],
                                            **dataset_params_for_loader)
            val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                    num_workers=model_params.get(
                                        'num_workers', 0),
                                    pin_memory=True if device.type == 'cuda' else False, drop_last=False)
            print(
                f"Validation DataLoader: Batches per epoch approx {len(val_loader)}")
        except Exception as e:
            print(
                f"Error creating val_dataset or val_loader: {e}. Proceeding without validation.")
            val_loader = None
    else:
        print("No validation annotations provided, val_loader will be None.")
    
    return train_loader, val_loader

def Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data):
    # Create datasets
    if train_data is None:
        train_dataset = None
    else:
        train_dataset = TherapyDataset(train_data, video_skels,
                                        in_memory=model_params['in_memory_generator_train'],
                                        validation=False, **model_params)
    
    if val_data is None:
        val_dataset = None
    else:
        val_dataset = TherapyDataset(val_data, video_skels,
                                    in_memory=model_params['in_memory_generator_val'],
                                    validation=True, **model_params)

    # Create sampler
    class_counts = train_data['action'].value_counts()
    class_weights = 1. / class_counts
    sample_weights = train_data['action'].map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create dataloaders
    if train_dataset is not None :
        train_loader = DataLoader(train_dataset, 
                                batch_size=model_params['batch_size'],
                                sampler=sampler, 
                                num_workers=model_params['num_workers'],
                                drop_last=False, 
                                collate_fn=collate_fn_classification_pre_pad)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, 
                                batch_size=model_params['batch_size'],
                                shuffle=False, 
                                num_workers=model_params['num_workers'],
                                drop_last=False, 
                                collate_fn=collate_fn_classification_pre_pad)

    if train_data is None and val_data is None:
        return None, None, None, None
    elif train_data is not None and val_data is None:
        return train_loader, None, train_dataset, None
    elif train_data is None and val_data is not None:
        return None, val_loader, None, val_dataset
    else:
        return train_loader, val_loader, train_dataset, val_dataset

def Create_MP_Dataloader(model_params, train_data, video_skels, val_data):
    # Create datasets
    if train_data is None:
        train_dataset = None
    else:
        train_dataset = TherapyDataset(train_data, video_skels,
                                        in_memory=model_params['in_memory_generator_train'],
                                        validation=False, **model_params)
    
    if val_data is None:
        val_dataset = None
    else:
        val_dataset = TherapyDataset(val_data, video_skels,
                                    in_memory=model_params['in_memory_generator_val'],
                                    validation=True, **model_params)

    # Create sampler
    class_counts = train_data['action'].value_counts()
    class_weights = 1. / class_counts
    sample_weights = train_data['action'].map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create dataloaders
    if train_dataset is not None :
        train_loader = DataLoader(train_dataset, 
                                batch_size=model_params['batch_size'],
                                sampler=sampler, 
                                num_workers=model_params['num_workers'],
                                drop_last=False, 
                                collate_fn=collate_fn_classification_pre_pad)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, 
                                batch_size=model_params['batch_size'],
                                shuffle=False, 
                                num_workers=model_params['num_workers'],
                                drop_last=False, 
                                collate_fn=collate_fn_classification_pre_pad)

    if train_data is None and val_data is None:
        return None, None, None, None
    elif train_data is not None and val_data is None:
        return train_loader, None, train_dataset, None
    elif train_data is None and val_data is not None:
        return None, val_loader, None, val_dataset
    else:
        return train_loader, val_loader, train_dataset, val_dataset

def Get_Confusion_Matrix(model_params, all_clf_preds_val, all_clf_labels_val, model_number, fold):
    """
    Compute and save confusion matrix for the validation set.
    Args:
        model_params (dict): Dictionary containing model configuration.
        pytorch_model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number for logging.
        all_clf_preds_val (list): List of predicted labels from validation set.
        all_clf_labels_val (list): List of true labels from validation set.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate one level up and into "Conversion comparison"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_name = model_params['model_name']
    # Create a directory for saving metrics
    confusion_matrix_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    # Compute confusion matrix
    conf_mat = confusion_matrix(all_clf_labels_val, all_clf_preds_val)

    # File name using model number and fold only
    base_filename = f'conf_matrix_model_{model_number}_fold_{fold}'
    
    # Save raw matrix (.npy)
    npy_path = os.path.join(confusion_matrix_dir, base_filename + '.npy')
    np.save(npy_path, conf_mat)

    # Save heatmap (.png)
    num_classes = conf_mat.shape[0]
    scale = min(max(num_classes / 5, 5), 40)
    fontsize = min(max(300 // num_classes, 5), 20)

    plt.figure(figsize=(scale + 5, (scale + 5) * 0.9))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        annot_kws={"size": fontsize + 6, "color": "black", "weight": "bold"},
        linewidths=1,
        linecolor='gray',
        square=False,  # Allow cells to stretch
        cbar=True,
    )

    # Set labels and title
    plt.xlabel("Predicted", fontsize=fontsize + 4, labelpad=20)
    plt.ylabel("Actual", fontsize=fontsize + 4, labelpad=20)
    plt.title(f"Confusion Matrix - Fold {fold}", fontsize=fontsize + 6, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize + 2)
    plt.yticks(rotation=0, fontsize=fontsize + 2)
    plt.tight_layout(pad=3.0)

    # Save the figure
    png_path = os.path.join(confusion_matrix_dir, base_filename + '.png')
    plt.savefig(png_path)
    plt.close()

def evaluate_model_on_all_data(model, full_eval_data, video_skels, model_params, device, fold,save_path):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create action mapping
    all_actions = full_eval_data['action'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(sorted(all_actions))}

    # Prepare role-based subsets
    subsets = {
        "all": full_eval_data,
        "children": full_eval_data[full_eval_data['is_therapist'] == 'n'],
        "therapists": full_eval_data[full_eval_data['is_therapist'] == 'y']
    }

    for group_name, subset_df in subsets.items():
        if subset_df.empty:
            print(f"Skipping {group_name}: no samples.")
            continue

        # Dataset & DataLoader
        dataset = TherapyDataset(
            subset_df,
            video_skels,
            in_memory=model_params.get('in_memory_generator_val', False),
            validation=True,
            **model_params
        )
        loader = DataLoader(
            dataset,
            batch_size=model_params.get('batch_size', 32),
            shuffle=False,
            num_workers=model_params.get('num_workers', 4),
            drop_last=False,
            collate_fn=collate_fn_classification_pre_pad
        )

        # Inference
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Report
        report = classification_report(all_labels, all_preds)
        print(f"\n--- Classification Report ({group_name}) ---")
        print(report)

        # Save report to file
        report_save_path = os.path.join(
            os.path.dirname(save_path),
            f"{os.path.splitext(os.path.basename(save_path))[0]}_fold{fold}_{group_name}_report.txt"
        )
        with open(report_save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Classification report saved to: {report_save_path}")

        # Confusion Matrix
        conf_mat = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix ({group_name})")
        plt.tight_layout()

        # Save per group
        group_save_path = os.path.join(
            os.path.dirname(save_path),
            f"{os.path.splitext(os.path.basename(save_path))[0]}_fold{fold}_{group_name}.png"
        )
        plt.savefig(group_save_path)
        print(f"Confusion matrix saved to: {save_path}")

def evaluate_model_on_all_data_mp(model, model_number, model_params, device, fold, save_path):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()

    full_dataset = TripletPoseDataset(
        pose_annotations_file="./datasets_annotations/mp_train_upper_body.txt",
        validation_mode=False,
        in_memory=model_params['in_memory_generator_train'],
        **model_params
    )

    labels = np.array([s['class_id'] for s in full_dataset.samples])
    for i in range(len(full_dataset)):
        sample, label = full_dataset[i]
        # print(f"Sample {i} - Shape: {sample.shape}, Label: {label}")

    # Child dataset
    child_dataset = TripletPoseDataset(
        pose_annotations_file="./datasets_annotations/mp_train_child_upper_body.txt",
        validation_mode=False,
        in_memory=model_params['in_memory_generator_train'],
        **model_params
    )

    # Therapist dataset
    therapist_dataset = TripletPoseDataset(
        pose_annotations_file="./datasets_annotations/mp_train_therapist_upper_body.txt",
        validation_mode=False,
        in_memory=model_params['in_memory_generator_train'],
        **model_params
    )

    # Dataloaders
    loaders = {
        "full": DataLoader(full_dataset,
                           batch_size=model_params.get('batch_size', 32),
                           shuffle=False,
                           num_workers=model_params.get('num_workers', 0),
                           pin_memory=True if device.type == 'cuda' else False,
                           drop_last=False),
        "child": DataLoader(child_dataset,
                            batch_size=model_params.get('batch_size', 32),
                            shuffle=False,
                            num_workers=model_params.get('num_workers', 0),
                            pin_memory=True if device.type == 'cuda' else False,
                            drop_last=False),
        "therapist": DataLoader(therapist_dataset,
                                batch_size=model_params.get('batch_size', 32),
                                shuffle=False,
                                num_workers=model_params.get('num_workers', 0),
                                pin_memory=True if device.type == 'cuda' else False,
                                drop_last=False)
    }

    # Evaluate each group
    for group_name, loader in loaders.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Report
        report = classification_report(all_labels, all_preds)
        print(f"\n--- Classification Report ({group_name}) ---")
        print(report)

        save_dir = save_path  # if save_path is a directory
        model_name = f"model_{model_number}_fold{fold}_{group_name}"

        # Save classification report
        report_save_path = os.path.join(save_dir, f"{model_name}_report.txt")
        with open(report_save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Classification report saved to: {report_save_path}")

        # Confusion matrix
        conf_mat = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix ({group_name})")
        plt.tight_layout()

        conf_mat_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(conf_mat_path)
        plt.close()
        print(f"Confusion matrix saved to: {conf_mat_path}")


def main(model_params):
    train_verbose = model_params.get('train_verbose', 1)
    log_interval = model_params.get('log_interval', 10)

    # --- Path and Feature Calculation (Cleaned Up) ---
    # print("--- Initializing Parameters and Paths ---")
    model_params['path_model'] = train_utils.create_model_folder(
        model_params['path_results'], model_params['model_name']
    )
    os.makedirs(model_params['path_model'], exist_ok=True)

    # Calculate num_jcd_feats and add to model_params (for record-keeping and if get_num_feats uses it)
    try:
        model_params['num_jcd_feats'] = int(
            comb(model_params.get('joints_num', 23), 2))
    except Exception as e:
        print(
            f"Warning: Could not calculate num_jcd_feats using comb: {e}. Setting to None or a fallback.")
        model_params['num_jcd_feats'] = None  # Or a fallback value if critical

    # Calculate final num_feats using get_num_feats
    try:
        calculated_num_feats = get_num_feats(**model_params)
        # Check if a 'num_feats' was pre-set in model_params and if it differs.
        # The value from get_num_feats should be authoritative.
        if model_params.get('num_feats') != calculated_num_feats and model_params.get('num_feats') is not None:
            print(f"Warning: Initial model_params['num_feats'] ({model_params.get('num_feats')}) "
                  f"differs from get_num_feats() calculation ({calculated_num_feats}). "
                  f"Using calculated value from get_num_feats: {calculated_num_feats}.")
        model_params['num_feats'] = calculated_num_feats
    except Exception as e:
        print(
            f"CRITICAL Error calling get_num_feats: {e}. num_feats might be incorrect.")
        if 'num_feats' not in model_params or model_params['num_feats'] is None:
            # If get_num_feats fails and no num_feats is set, it's a critical issue.
            print("CRITICAL: num_feats is not set and get_num_feats failed. Aborting.")
            return  # Exit if essential num_feats cannot be determined

    # Save the complete model_params to JSON
    model_params_save_path = os.path.join(
        model_params['path_model'], 'model_params.json')
    try:
        with open(model_params_save_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Saved model parameters to {model_params_save_path}")
    except Exception as e:
        print(f"Error saving model_params.json: {e}")

    # print(' * Final Model params for this run:',
    #       json.dumps(model_params, indent=2))

    copy_scaler_if_needed(model_params)

    # --- Annotation File Counts (for informational purposes) ---
    num_train_files, num_val_files = 0, 0
    try:
        if model_params.get('train_annotations'):
            with open(model_params['train_annotations'], 'r') as f:
                num_train_files = len(f.read().splitlines())
        if model_params.get('val_annotations'):
            with open(model_params['val_annotations'], 'r') as f:
                num_val_files = len(f.read().splitlines())
        print(
            f"Num train annotation lines: {num_train_files}, Num val annotation lines: {num_val_files}")
    except FileNotFoundError as e:
        print(f"Warning: Annotation file not found: {e}. Counts will be 0.")
    except Exception as e:
        print(
            f"Warning: Error reading annotation files: {e}. Counts might be inaccurate.")


    """ # --- Keras-like Debugging Block ---
    print('\n* Building model (PyTorch does not require explicit build)\n')
    # No model.build() in PyTorch, it's built on first forward pass or by knowing input shape

    print('\n* Initializing inputs and outputs (PyTorch)')
    # Use parameters from model_params for dummy input size
    # Keras dummy_inpt was (batch_size, max(abs(max_seq_len), 123), num_feats)
    dummy_batch_size = model_params.get('batch_size', 32)
    dummy_seq_len_abs = abs(model_params.get('max_seq_len', -32))
    dummy_seq_len = max(dummy_seq_len_abs if dummy_seq_len_abs >
                        0 else 32, 123)  # Match Keras logic
    dummy_num_feats = model_params['num_feats']
    print(
        f"\n* Dummy input shape: (batch_size={dummy_batch_size}, seq_len={dummy_seq_len}, num_feats={dummy_num_feats})")
    dummy_inpt_np = np.random.rand(
        dummy_batch_size, dummy_seq_len, dummy_num_feats).astype(np.float32)
    print(f"\n* Dummy input shape (NumPy): {dummy_inpt_np.shape}")

    # Convert NumPy to PyTorch tensor
    dummy_inpt_torch = torch.from_numpy(dummy_inpt_np).to(device)

    pytorch_model.eval()  # Set model to evaluation mode for consistent output
    with torch.no_grad():  # Disable gradients for this test
        # PyTorch model call (equivalent to Keras model(dummy_inpt))
        dummy_pred_torch_call = pytorch_model(dummy_inpt_torch)
        if isinstance(dummy_pred_torch_call, list):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):',
                  [p.shape for p in dummy_pred_torch_call])
        elif isinstance(dummy_pred_torch_call, torch.Tensor):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):',
                  dummy_pred_torch_call.shape)
        else:
            print(' * dummy_pred_torch_call type (PyTorch model(input)):',
                  type(dummy_pred_torch_call))

        # PyTorch get_embedding call
        # Ensure get_embedding also takes a PyTorch tensor
        dummy_emb_torch = pytorch_model.get_embedding(dummy_inpt_torch)
        print(' * dummy_emb_torch shape (PyTorch get_embedding(input)):',
              dummy_emb_torch.shape)

    # If model returns a single tensor instead of a list
    if isinstance(dummy_pred_torch_call, torch.Tensor):
        dummy_pred_torch_call = [dummy_pred_torch_call]

    # Save each full output (not each sample)
    for i, out in enumerate(dummy_pred_torch_call):
        np.save(f'pytorch_output_{i}.npy', out.cpu().numpy())
        # Optional readable version
        reshaped = out.view(out.size(0), -1).cpu().numpy()
        np.savetxt(f'pytorch_output_{i}.txt', reshaped)
    """

    # --- Data Loading (MP) ---   
    # print("\n--- Setting up PyTorch DataLoaders ---")
    # dataset_params_for_loader = model_params.copy()
    # try:
    #     train_dataset = TripletPoseDataset(pose_annotations_file=model_params['train_annotations'],
    #                                        validation_mode=False,
    #                                        in_memory=model_params['in_memory_generator_train'],
    #                                        **dataset_params_for_loader)
    #     train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True,
    #                               num_workers=model_params.get('num_workers', 0),
    #                               pin_memory=True if device.type == 'cuda' else False, drop_last=True)
    #     print(f"Train DataLoader: Batches per epoch approx {len(train_loader)}")
    # except Exception as e:
    #     print(f"CRITICAL Error creating train_dataset or train_loader: {e}")
    #     import traceback; traceback.print_exc(); return# 

    # val_loader = None
    # if model_params.get('val_annotations') and model_params['val_annotations'] != '':
    #     try:
    #         val_dataset = TripletPoseDataset(pose_annotations_file=model_params['val_annotations'],
    #                                          validation_mode=True,
    #                                          in_memory=model_params['in_memory_generator_val'],
    #                                          **dataset_params_for_loader)
    #         val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True,
    #                                 num_workers=model_params.get('num_workers', 0),
    #                                 pin_memory=True if device.type == 'cuda' else False, drop_last=False)
    #         print(f"Validation DataLoader: Batches per epoch approx {len(val_loader)}")
    #     except Exception as e:
    #         print(f"Error creating val_dataset or val_loader: {e}. Proceeding without validation.")
    #         val_loader = None
    # else:
    #     print("No validation annotations provided, val_loader will be None.") 

    

    # --- Data Loading Therapist ---
    """
    # Load your raw data
    raw_data_path = './datasets/therapies_dataset/'
    actions_data = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
    video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))

    # Filter out unwanted actions (same as your TF code)
    actions_data = actions_data[~actions_data.action.isin(['no', 'si'])]
    print(f"Loaded actions_data with {len(actions_data)} entries and video_skels with {len(video_skels)} videos.")
    
    actions_data = actions_data.sort_values(by=['patient', 'session', 'video', 'ex_num'])
    
    # Show the full DataFrame without truncation
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #    print(actions_data)
    
    #actions_data, actions_data_final_val = train_test_split(
    #    actions_data,
    #    test_size=0.15,  # 15% = ~24 samples
    #    stratify=actions_data['action'],
    #    random_state=42
    #)

    ## Debug print for actions_data_final_val
    #print("First few rows of actions_data_final_val:")
    #print(actions_data_final_val.head())
    
    ## How many lines have the same action in actions_data_final_val?
    #action_counts_final_val = actions_data_final_val['action'].value_counts()
    #print(f"Action counts in final validation set:\n{action_counts_final_val}")
    
    # Print first few rows of actions_data for debugging
    #print("First few rows of actions_data:")
    # all_actions = actions_data['action'].unique()
    #print(f"Unique actions found: {all_actions}")
    #
    ## How many lines have the same action?
    # action_counts = actions_data['action'].value_counts()
    # print(f"Action counts:\n{action_counts}")
    
    # --- Prepare data ---
    # Create a mapping from action names to indices
    all_actions = actions_data['action'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(sorted(all_actions))}
    labels = actions_data['action'].map(action_to_idx).values

    # Print Labels ID
    print("Labels ID:")
    for action, idx in action_to_idx.items():
        print(f"  {action}: {idx}")

    # === Make a copy of the full dataset for later evaluation ===
    full_eval_data = actions_data.copy()

    # # Separate into children and therapist subsets (assuming you have a 'role' column or similar)
    # children_data = full_eval_data[full_eval_data['role'] == 'child']
    # therapist_data = full_eval_data[full_eval_data['role'] == 'therapist']

    # # If you want labels for these too:
    # children_labels = children_data['action'].map(action_to_idx).values
    # therapist_labels = therapist_data['action'].map(action_to_idx).values
    # """
    
    # --- Cross-Validation Setup ---
    if model_params.get("K", None) is not None:
        k_folds = model_params.get("K", 5)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Load full dataset once (MP)
        full_dataset = TripletPoseDataset(
            pose_annotations_file=model_params['train_annotations'],
            validation_mode=False,
            in_memory=model_params['in_memory_generator_train'],
            **model_params
        )
        
        labels = np.array([s['class_id'] for s in full_dataset.samples])
        for i in range(len(full_dataset)):
            sample, label = full_dataset[i]
            print(f"Sample {i} - Shape: {sample.shape}, Label: {label}")
        
        print(f"Full dataset size: {len(full_dataset)} samples")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique classes in dataset: {np.unique(labels)}")
        print(f"Full dataset labels: {labels}")
        
        # --- Training Loop ---
        fold_val_f1_scores = []
        fold_val_auc_scores = []

        # for fold, (train_idx, val_idx) in enumerate(skf.split(actions_data, labels)):
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):

            # --- PyTorch Model Instantiation ---
            pytorch_model, initial_state_dict = create_pytorch_model(model_params, fold)
            
            # --- Model Summary ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pytorch_model.to(device)
            
            # --- Create subsets (Therapist) ---
            # train_data = actions_data.iloc[train_idx].reset_index(drop=True)
            # val_data = actions_data.iloc[val_idx].reset_index(drop=True)
            
            # --- Create DataLoaders for this fold (Therapist) ---
            # train_loader, val_loader, train_dataset, val_dataset = Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data)
            
            # --- Create subsets (MP) ---
            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            val_subset = torch.utils.data.Subset(full_dataset, val_idx)
            
            # --- Create DataLoaders for this fold (MP)---
            train_loader = DataLoader(
                train_subset,
                batch_size=model_params['batch_size'],
                shuffle=True,
                num_workers=model_params.get('num_workers', 0),
                pin_memory=True if device.type == 'cuda' else False,
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=model_params['batch_size'],
                shuffle=False,
                num_workers=model_params.get('num_workers', 0),
                pin_memory=True if device.type == 'cuda' else False,
                drop_last=False
            )
            
            # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
            active_losses, loss_weights_pytorch_pt, optimizer = Setup_optimizer_and_loss(pytorch_model, model_params, device, train_subset)
            
            tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
            model_folder = os.path.basename(os.path.dirname(tb_log_dir))

            match = re.search(r'_model_(\d+)', model_folder)
            if match:
                model_number = match.group(1)
                # print(f"Extracted model number: {model_number}")
            else:
                # print("No model number found.")
                pass

            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)

            # --- PyTorch Training Setup ---
            # print("\n--- Starting PyTorch Training Loop ---")

            monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, \
                lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter = \
                    Setup_training(model_params, pytorch_model, device, optimizer)

            num_epochs = model_params.get('epochs', 1)
            # print(
            #     f"Training for {num_epochs} epochs with batch size {model_params['batch_size']}")
            # time.sleep(5)  # Small delay for readability in logs
            softmax_outputs = []  # To store softmax outputs if needed
            train_losses = []
            
            val_losses = []
            val_f1_scores = []
            val_auc_scores = []  # Move this to the top, before the epoch loop

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                # print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                pytorch_model.train()
                running_train_loss = 0.0
                # Store individual loss components if needed for logging
                running_train_loss_clf = 0.0
                running_train_loss_triplet = 0.0

                # --- Training Phase ---
                running_train_loss, running_train_loss_clf = train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, \
                    device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch, train_verbose, log_interval)
                
                avg_epoch_train_loss = running_train_loss / \
                    len(train_loader) if len(train_loader) > 0 else 0
                train_losses.append(avg_epoch_train_loss)
                tb_writer.add_scalar('LossEpoch_Train/Total',
                                    avg_epoch_train_loss, epoch)
                if 'classification' in active_losses:
                    tb_writer.add_scalar('LossEpoch_Train/Classification', running_train_loss_clf /
                                        len(train_loader) if len(train_loader) > 0 else 0, epoch)
                if 'triplet' in active_losses:  # Add if triplet loss is calculated
                    tb_writer.add_scalar('LossEpoch_Train/Triplet', running_train_loss_triplet /
                                        len(train_loader) if len(train_loader) > 0 else 0, epoch)

                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('LearningRate', current_lr, epoch)
                # print(
                #     f"Epoch {epoch+1} Train Summary: \
                #         Avg Total Loss: {avg_epoch_train_loss:.4f}, \
                #             LR: {current_lr}    ", end='')

                # --- Validation Phase ---
                if val_loader is not None:
                    val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, \
                    all_clf_preds_val, all_clf_labels_val, all_clf_probs_val \
                        = validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt)
                    
                    # Calculate average val losses
                    avg_epoch_val_loss = running_val_loss / \
                        len(val_loader) if len(val_loader) > 0 else 0
                    val_metrics['val_loss'] = avg_epoch_val_loss
                    val_losses.append(avg_epoch_val_loss)
                    tb_writer.add_scalar('LossEpoch_Val/Total',
                                        avg_epoch_val_loss, epoch)

                    # AUC-ROC Calculation (if classification and probabilities available)
                    if all_clf_labels_val and all_clf_probs_val:
                        y_true = np.array(all_clf_labels_val)
                        y_scores = np.array(all_clf_probs_val)
                        
                        if y_scores.shape[1] == 2:
                            # Binary classification: take score for class 1
                            auc = roc_auc_score(y_true, y_scores[:, 1])
                        else:
                            # Multiclass classification with only present classes
                            present_classes = np.unique(y_true)
                            y_true_bin = label_binarize(y_true, classes=present_classes)
                            y_scores_filtered = y_scores[:, present_classes]
                            auc = roc_auc_score(y_true_bin, y_scores_filtered, average="macro", multi_class="ovr")

                        val_metrics['val_auc'] = auc
                        tb_writer.add_scalar('AUC_ROC/val', auc, epoch)
                        val_auc_scores.append(auc)

                    if 'classification' in active_losses:
                        avg_val_loss_clf = running_val_loss_clf / len(val_loader)
                        tb_writer.add_scalar(
                            'LossEpoch_Val/Classification', avg_val_loss_clf, epoch)
                        val_metrics['val_clf_loss'] = avg_val_loss_clf

                    if all_clf_labels_val:
                        correct_val = sum(p == t for p, t in zip(
                            all_clf_preds_val, all_clf_labels_val))
                        val_accuracy = correct_val / len(all_clf_labels_val)
                        val_metrics['val_accuracy'] = val_accuracy
                        tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                        f1_macro = f1_score(all_clf_labels_val,
                                            all_clf_preds_val, average='macro')
                        val_metrics['val_f1_macro'] = f1_macro
                        tb_writer.add_scalar('F1Score/val_macro', f1_macro, epoch)
                        val_f1_scores.append(f1_macro)

                        status = (
                            f"\r    Fold {fold+1}/{k_folds} | "
                            f"Epoch {epoch+1} | "
                            f"Loss: {avg_epoch_val_loss:.4f} | "
                            f"Acc: {val_accuracy:.4f} | "
                            f"F1: {f1_macro:.4f} | "
                            f"AUC: {auc:.4f}"
                        )
                        sys.stdout.write(status)
                        sys.stdout.flush()
                    #else:
                    #    print(
                    #        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f} (No classification preds for accuracy)")

                    if 'triplet' in active_losses:
                        avg_val_loss_triplet = running_val_loss_triplet / \
                            len(val_loader)
                        val_metrics['val_triplet_loss'] = avg_val_loss_triplet
                        tb_writer.add_scalar(
                            'LossEpoch_Val/Triplet', avg_val_loss_triplet, epoch)

                    if 'classification' not in active_losses:
                        print(f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f}")

                    # --- "Callbacks" logic for this epoch ---
                    current_metric_for_scheduler_es = val_metrics.get(
                        monitor_metric_name, avg_epoch_val_loss)

                    # Step LR scheduler
                    lr_scheduler.step(current_metric_for_scheduler_es)

                    # ModelCheckpoint
                    if (monitor_mode == 'max' and (current_metric_for_scheduler_es - best_monitor_metric_val) > min_delta) or \
                            (monitor_mode == 'min' and (best_monitor_metric_val - current_metric_for_scheduler_es) > min_delta):
                        best_monitor_metric_val = current_metric_for_scheduler_es
                        # Keras format: 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
                        # Using train loss for 'loss' part of filename for consistency with Keras.
                        best_val_f1 = np.max(val_f1_scores)
                        best_val_auc = np.max(val_auc_scores)
                        checkpoint_filename = (
                            f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                            f"{monitor_metric_name.replace('val_', '')}{current_metric_for_scheduler_es:.5f}-"
                            f"f1{best_val_f1:.5f}_"
                            f"model_{model_number}_fold_{fold}.pt"
                        )
                        full_checkpoint_path = os.path.join(weights_save_path, checkpoint_filename)
                        torch.save(pytorch_model.state_dict(), full_checkpoint_path)
                        best_checkpoint_filename = checkpoint_filename
                        
                        # Save confusion matrix for this fold
                        Get_Confusion_Matrix(model_params, all_clf_preds_val, all_clf_labels_val, model_number, fold)
                        
                        # Delete other checkpoints for this fold, keep only the best one
                        for fname in os.listdir(weights_save_path):
                            if (
                                fname.endswith(f"fold_{fold}.pt") and
                                fname != best_checkpoint_filename
                            ):
                                try:
                                    os.remove(os.path.join(weights_save_path, fname))
                                except Exception as e:
                                    print(f"Could not delete {fname}: {e}")

                        # print(
                        #     f"  Saved checkpoint: {checkpoint_filename} (Monitored '{monitor_metric_name}': {current_metric_for_scheduler_es:.5f})")
                        # Update best_val_for_early_stop if this is the metric early stopping also monitors
                        # Check if it's the same metric
                        if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                            best_val_for_early_stop = best_monitor_metric_val
                            early_stopping_counter = 0
                    else:  # Only increment early stopping counter if not improving on its specific metric
                        if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                            early_stopping_counter += 1

                    # EarlyStopping (check against its own best metric, which might be same as checkpointing or different)
                    if early_stopping_counter >= early_stopping_patience:
                        # print(
                        #     f"  Early stopping triggered after {early_stopping_patience} epochs without improvement on '{monitor_metric_name}'.")
                        break  # Break from epoch loop
                else:  # No val_loader
                    print("  No validation loader. Skipping validation phase, LR scheduling based on val_metrics, and early stopping.")
                    # Optionally, save model at end of epoch if no validation
                    checkpoint_filename = f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-no_val.pt"
                    torch.save(pytorch_model.state_dict(), os.path.join(
                        weights_save_path, checkpoint_filename))
                    print(
                        f"  Saved model checkpoint (no validation): {checkpoint_filename}")

            epoch_duration = time.time() - epoch_start_time
            # print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
            if epoch_duration > 0:
                tb_writer.add_scalar(
                    'Performance/epoch_duration_sec', epoch_duration, epoch)
            
            # Save final model F1 and AUC scores for this fold
            fold_val_f1_scores.append(best_val_f1)
            fold_val_auc_scores.append(best_val_auc)
            
            # Determine folder one level up
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
            model_name = model_params['model_name']
            
            # Create a directory for saving metrics
            metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
            os.makedirs(metrics_save_dir, exist_ok=True)

            # 1. Find the best values (assuming best means minimum loss, maximum score)
            best_train_loss = np.min(train_losses)
            best_val_loss = np.min(val_losses)

            # 2. Create filename with best values embedded (rounded for readability)
            filename = (
                f"pytorch_therapy_classifier_train_loss-{best_train_loss:.4f}_"
                f"val_loss-{best_val_loss:.4f}_"
                f"val_f1-{best_val_f1:.4f}_"
                f"val_auc-{best_val_auc:.4f}_"
                f"model_{model_number}_"
                f"fold_{fold}.npz"
            )
            
            # 3. Save arrays with this filename
            np.savez(os.path.join(metrics_save_dir, filename),
                train_losses=np.array(train_losses),
                val_losses=np.array(val_losses),
                val_f1_scores=np.array(val_f1_scores),
                val_auc_scores=np.array(val_auc_scores),
            )
            
            # Evaluate on all data
            # evaluate_model_on_all_data_mp(pytorch_model, model_number, model_params, device, fold, metrics_save_dir)
        
        # Save the mean of the best F1 scores across folds
        # in order to report the overall performance
        mean_best_val_f1 = np.mean(fold_val_f1_scores)
        
        # ----------------------------
        # Save averaged K-fold model
        # ----------------------------
        save_average_k_fold_model(model_params=model_params, weights_save_path=weights_save_path, model_class=TCN_clf, device='cuda')
        # ----------------------------
        
    else:
        # --- Only Training for the best Hyperparameters using all the data ---
        print("\n--- Training with all data (no K-Fold) ---")
        
        # --- Create DataLoaders for this fold ---
        # train_loader, val_loader, train_dataset, val_dataset = Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data)
        train_loader, val_loader, train_dataset, val_dataset = Create_Therapy_Dataloader(model_params, actions_data, video_skels, None)

        # --- PyTorch Model Instantiation ---
        pytorch_model, initial_state_dict = create_pytorch_model(model_params)
        
        # --- Model Summary ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model.to(device)

        # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
        active_losses, loss_weights_pytorch_pt, optimizer = Setup_optimizer_and_loss(pytorch_model, model_params, device, train_dataset)
        
        tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
        model_folder = os.path.basename(os.path.dirname(tb_log_dir))

        match = re.search(r'_model_(\d+)', model_folder)
        if match:
            model_number = match.group(1)
            print(f"Extracted model number: {model_number}")
        else:
            print("No model number found.")

        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard: Logging to {tb_log_dir}")

        # --- PyTorch Training Setup ---
        # print("\n--- Starting PyTorch Training Loop ---")

        monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, \
            lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter = \
                Setup_training(model_params, pytorch_model, device, optimizer)
        
        num_epochs = model_params.get('epochs', 1)
        # print(
        #     f"Training for {num_epochs} epochs with batch size {model_params['batch_size']}")
        # time.sleep(5)  # Small delay for readability in logs
        softmax_outputs = []  # To store softmax outputs if needed
        train_losses = []
        val_losses = []
        val_f1_scores = []
        val_auc_scores = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            pytorch_model.train()
            running_train_loss = 0.0
            # Store individual loss components if needed for logging
            running_train_loss_clf = 0.0
            running_train_loss_triplet = 0.0

            # --- Training Phase ---
            running_train_loss, running_train_loss_clf = train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, \
                device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch, train_verbose, log_interval)
            
            avg_epoch_train_loss = running_train_loss / \
                len(train_loader) if len(train_loader) > 0 else 0
            train_losses.append(avg_epoch_train_loss)
            tb_writer.add_scalar('LossEpoch_Train/Total',
                                avg_epoch_train_loss, epoch)
            if 'classification' in active_losses:
                tb_writer.add_scalar('LossEpoch_Train/Classification', running_train_loss_clf /
                                    len(train_loader) if len(train_loader) > 0 else 0, epoch)
            if 'triplet' in active_losses:  # Add if triplet loss is calculated
                tb_writer.add_scalar('LossEpoch_Train/Triplet', running_train_loss_triplet /
                                    len(train_loader) if len(train_loader) > 0 else 0, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            tb_writer.add_scalar('LearningRate', current_lr, epoch)
            status = (
                f"\r    Epoch {epoch+1} | "
                f"Loss: {avg_epoch_train_loss:.4f} | "
                f"current_lr: {current_lr:.4f} | "
            )
            sys.stdout.write(status)
            sys.stdout.flush()

            # --- Validation Phase ---
            if val_loader is not None:
                val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, \
                    all_clf_preds_val, all_clf_labels_val, all_clf_probs_val \
                        = validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt)
                
                # Calculate average val losses
                avg_epoch_val_loss = running_val_loss / \
                    len(val_loader) if len(val_loader) > 0 else 0
                val_metrics['val_loss'] = avg_epoch_val_loss
                val_losses.append(avg_epoch_val_loss)
                tb_writer.add_scalar('LossEpoch_Val/Total',
                                    avg_epoch_val_loss, epoch)

                # AUC-ROC Calculation (if classification and probabilities available)
                if all_clf_labels_val and all_clf_probs_val:
                    y_true = np.array(all_clf_labels_val)
                    y_scores = np.array(all_clf_probs_val)
                    
                    if y_scores.shape[1] == 2:
                        # Binary classification: take score for class 1
                        auc = roc_auc_score(y_true, y_scores[:, 1])
                    else:
                        # Multiclass classification with only present classes
                        present_classes = np.unique(y_true)
                        y_true_bin = label_binarize(y_true, classes=present_classes)
                        y_scores_filtered = y_scores[:, present_classes]
                        auc = roc_auc_score(y_true_bin, y_scores_filtered, average="macro", multi_class="ovr")

                    val_metrics['val_auc'] = auc
                    tb_writer.add_scalar('AUC_ROC/val', auc, epoch)
                    val_auc_scores.append(auc)

                if 'classification' in active_losses:
                    avg_val_loss_clf = running_val_loss_clf / len(val_loader)
                    tb_writer.add_scalar(
                        'LossEpoch_Val/Classification', avg_val_loss_clf, epoch)
                    val_metrics['val_clf_loss'] = avg_val_loss_clf

                if all_clf_labels_val:
                    correct_val = sum(p == t for p, t in zip(
                        all_clf_preds_val, all_clf_labels_val))
                    val_accuracy = correct_val / len(all_clf_labels_val)
                    val_metrics['val_accuracy'] = val_accuracy
                    tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                    # TODO: Usar Wheighted F1 Score?
                    f1_macro = f1_score(all_clf_labels_val,
                                        all_clf_preds_val, average='macro')
                    val_metrics['val_f1_macro'] = f1_macro
                    tb_writer.add_scalar('F1Score/val_macro', f1_macro, epoch)
                    val_f1_scores.append(f1_macro)

                    status = (
                        f"\r    Fold {fold+1}/{k_folds} | "
                        f"Epoch {epoch+1} | "
                        f"Loss: {avg_epoch_val_loss:.4f} | "
                        f"Acc: {val_accuracy:.4f} | "
                        f"F1: {f1_macro:.4f} | "
                        f"AUC: {auc:.4f}"
                    )
                    sys.stdout.write(status)
                    sys.stdout.flush()

                else:
                    print(
                        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f} (No classification preds for accuracy)")

                if 'triplet' in active_losses:
                    avg_val_loss_triplet = running_val_loss_triplet / \
                        len(val_loader)
                    val_metrics['val_triplet_loss'] = avg_val_loss_triplet
                    tb_writer.add_scalar(
                        'LossEpoch_Val/Triplet', avg_val_loss_triplet, epoch)

                if 'classification' not in active_losses:
                    print(
                        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f}")

                # --- "Callbacks" logic for this epoch ---
                current_metric_for_scheduler_es = val_metrics.get(
                    monitor_metric_name, avg_epoch_val_loss)

                # Step LR scheduler
                lr_scheduler.step(current_metric_for_scheduler_es)

                # ModelCheckpoint
                if (monitor_mode == 'max' and (current_metric_for_scheduler_es - best_monitor_metric_val) > min_delta) or \
                        (monitor_mode == 'min' and (best_monitor_metric_val - current_metric_for_scheduler_es) > min_delta):
                    best_monitor_metric_val = current_metric_for_scheduler_es
                    # Keras format: 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
                    # Using train loss for 'loss' part of filename for consistency with Keras.
                    best_val_f1 = np.max(val_f1_scores)
                    checkpoint_filename = (
                        f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                        f"{monitor_metric_name.replace('val_', '')}{current_metric_for_scheduler_es:.5f}-"
                        f"f1{best_val_f1:.5f}.pt"
                    )
                    full_checkpoint_path = os.path.join(weights_save_path, checkpoint_filename)
                    torch.save(pytorch_model.state_dict(), full_checkpoint_path)
                    best_checkpoint_filename = checkpoint_filename

                    print(
                        f"  Saved checkpoint: {checkpoint_filename} (Monitored '{monitor_metric_name}': {current_metric_for_scheduler_es:.5f})")
                    # Update best_val_for_early_stop if this is the metric early stopping also monitors
                    # Check if it's the same metric
                    if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                        best_val_for_early_stop = best_monitor_metric_val
                        early_stopping_counter = 0
                else:  # Only increment early stopping counter if not improving on its specific metric
                    if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                        early_stopping_counter += 1

                # Confusion Matrix (Every 10 epochs)
                # Every 10 epochs, save confusion matrix (starting from epoch 0)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    Get_Confusion_Matrix(model_params, all_clf_preds_val, all_clf_labels_val, model_number, fold)
                # EarlyStopping (check against its own best metric, which might be same as checkpointing or different)
                if early_stopping_counter >= early_stopping_patience:
                    # print(
                    #     f"  Early stopping triggered after {early_stopping_patience} epochs without improvement on '{monitor_metric_name}'.")
                    break  # Break from epoch loop
            else:  # No val_loader
                # print("  No validation loader. Skipping validation phase, LR scheduling based on val_metrics, and early stopping.")
                # Optionally, save model at end of epoch if no validation
                checkpoint_filename = (
                    f"Best_Model-ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                    f"f1{model_params.get('best_val_f1', 0):.5f}.pt"
                )
                torch.save(pytorch_model.state_dict(), os.path.join(
                    weights_save_path, checkpoint_filename))
                
                # Delete all other checkpoints except the best
                best_checkpoint_filename = checkpoint_filename
                for fname in os.listdir(weights_save_path):
                    if fname.endswith('.pt') and fname != best_checkpoint_filename:
                        try:
                            os.remove(os.path.join(weights_save_path, fname))
                        except Exception as e:
                            print(f"Could not delete {fname}: {e}")

        epoch_duration = time.time() - epoch_start_time
        # print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        if epoch_duration > 0:
            tb_writer.add_scalar(
                'Performance/epoch_duration_sec', epoch_duration, epoch)
        
        # Determine folder one level up
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        model_name = model_params['model_name']
        # Create a directory for saving metrics
        metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
        os.makedirs(metrics_save_dir, exist_ok=True)
        
        # 1. Find the best values (assuming best means minimum loss, maximum score)
        best_train_loss = np.min(train_losses)
        mean_best_val_f1 = model_params.get('best_val_f1', 0)

        # 2. Create filename with best values embedded (rounded for readability)
        filename = (
            f"pytorch_therapy_classifier_train_loss-{best_train_loss:.4f}_"
            f"val_f1-{mean_best_val_f1:.4f}_"
            f"final_model_{model_number}_.npz"
        )
        
        # 3. Save arrays with this filename
        np.savez(os.path.join(metrics_save_dir, filename),
                train_losses=np.array(train_losses),
                )

    # tb_writer.close()

    # final_state_dict = pytorch_model.state_dict()

    # changed_keys = []

    # print("\n--- PyTorch Training Summary ---")
    # print(f"Total epochs: {num_epochs}")
    
    # for key in initial_state_dict:
    #     if not torch.equal(initial_state_dict[key], final_state_dict[key].cpu()):
    #         changed_keys.append(key)
    # for key in excluded_pt_keys:
    #     if not torch.equal(initial_state_dict[key], final_state_dict[key].cpu()):
    #         print(f"OK: Layer {key} changed!")
    #     else:
    #         print(f"Warning: Frozen layer {key} changed.")
# 
    # print("\n")
    # print("Weights that changed during training:")
    # for key in changed_keys:
    #     print(f" - {key}")
    # print("\n")
# 
    # for key in changed_keys:
    #     diff = torch.norm(final_state_dict[key].cpu() - initial_state_dict[key].cpu()).item()
    #     print(f"{key}: Δ = {diff:.6f}")
# 
    print("\n--- PyTorch Training Finished ---")
    
    return mean_best_val_f1, model_number

    # --- Post-training actions ---
    # TODO: Example: remove_suboptimal_weights (needs careful implementation)
    # if model_params.get('remove_suboptimal_weights', False):
    #     try:
    #         # This function needs to be defined and robust
    #         from remove_suboptimal_weights import remove_path_weights_pytorch
    #         remove_path_weights_pytorch(weights_save_path, monitor_metric_name, mode=monitor_mode)
    #     except ImportError:
    #         print("Warning: remove_suboptimal_weights_pytorch not found.")
    #     except Exception as e_rsw:
    #         print(f"Error during remove_suboptimal_weights: {e_rsw}")


def objective(trial, static_params):
    # Define hyperparameter search space
    optuna_params = {        
        # Training-related
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),

        # LSTM
        "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
        "lstm_recurrent_dropout": trial.suggest_float("lstm_recurrent_dropout", 0.0, 0.5),

        # ReduceLROnPlateau
        "lr_min_delta": trial.suggest_float("lr_min_delta", 1e-5, 1e-2, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-7, 1e-4, log=True),

        # EarlyStopping
        "early_stopping_min_delta": trial.suggest_float("early_stopping_min_delta", 1e-5, 1e-2, log=True),

        # Changes if training only the 1st layer or also other the layers
        "init_lr": trial.suggest_float("init_lr", 1e-5, 1e-3, log=True),
        "lr_factor": trial.suggest_float("lr_factor", 0.1, 0.9),
        "lr_patience": trial.suggest_int("lr_patience", 2, 10),
        "es_patience": trial.suggest_int("es_patience", 3, 12),
    }
    
    # Combine the two
    model_params = {**static_params, **optuna_params}

    # Train your model and return the validation F1 or loss
    best_val_f1, model_number = main(model_params)  # Use cross-validation here
    
    # Update best F1 and best model number if improved
    if static_params["best_val_f1"] is None or best_val_f1 > static_params["best_val_f1"]:
        static_params["best_val_f1"] = best_val_f1
        static_params["best_model_number"] = model_number
        print(f"New best model: {model_number} with F1: {best_val_f1:.4f}")

    return best_val_f1

if __name__ == "__main__":

    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "No GPU found")
    
    time.sleep(2)  # Small delay for readability in logs
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, help='Number of training epochs')
    # parser.add_argument('--init_lr', type=float, help='Initial learning rate')
    # parser.add_argument('--batch_size', type=int, help='Batch size')
    # parser.add_argument('--train_annotations', type=str, help='Path to train annotations')
    # parser.add_argument('--val_annotations', type=str, help='Path to val annotations')
    # Add more if you want — just match the key names in model_params
    #
    # args = parser.parse_args()

    # TODO: Change this after each run to avoid overwriting
    # Fixed params — the rest of what your model expects
    static_params = {
        "best_val_f1": 0,  # Best F1 score from the study
        "best_model_number": 0,  # Best model number from the study
        "joints_num": 24, # 25 for Kinect / 24 for MP / 12 for upper body MP CADDIN
        "num_classes": 14, # Number of classes for classification (NTU-120 has 120, CADDIN has 12 and Therapies has 14)

        "epochs": 300, # Number of training epochs
        "K": 5,  # Number of folds for cross-validation
        "n_trials": 40,  # Number of trials for Optuna

        # Set to 0 for no training logs, 1 for basic logs, >1 for more detailed logs
        "train_verbose": 1,
        "num_workers": 0,  # Number of workers for DataLoader, adjust based on your system
        "path_results": "./pretrained_models_Pytorch/",

        # TODO: Change based on if you want to continue to use the models from the previous K-fold
        "fine_tunning": True,  # Set to True if you want to fine-tune the previous K models
        
        # TODO: Change based on if you want to use the K-folds models
        "average_k_fold": False,  # Set to True if you want to average the results of the K-folds models
        
        # TODO: Change every time you switch to the next model
        # "model_name": "Models_Therapist_Classifier_Block_5_4_3_2_1_0",
        # "model_name": "Models_MP_Classifier_Block_0_1_2_3_4_5(k_fold_separated_c_th_comp)",
        # "model_name": "Models_MP_Classifier_Block_0_1_2_3_4_5(k_fold_separated_c_th_comp_NEW)",
        # "model_name": "Models_MP_Classifier_Block_0_1_2_3_4_5(k_fold_separated_c_th_comp_NEW_after_MP_Therapist_APPDA)",
        # "model_name": "Models_MP_Therapist_APPDA_Block_0(upper_body)",
        "model_name": "Models_MP_CADDIN_Upper_Body_Block_0(upper_body)",

        # TODO: Change every time you switch to the next model
        # Path to the pre-trained model in Pytorch format
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_Therapist_Classifier_Block_5/0730_1921_model_1/weights/Best_Model-ep300-trainloss0.31293-f10.54116.pt",
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_MP_Therapist_APPDA_Classifier_Block_0_1_2_3_4_5",
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_MP_Classifier_Block_0_1_2_3_4_5(k_fold_separated_c_th_comp)",  # Path to the pre-trained model for Therapies dataset in Pytorch format
        "pretrained_model_path": "./pretrained_models_Pytorch/Models_MP_Therapist_APPDA_Block_0(upper_body)",  # Path to the pre-trained model for Therapies dataset in Pytorch format
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_MP_CADDIN_Upper_Body_Block_0(upper_body)",  # Path to the pre-trained model for Therapies dataset in Pytorch format

        # TODO: Change every time you switch to the next model
        # Use a pre-trained model (Set True if you want to use a pre-trained model)
        "use_pretrained_model": True,  # Set to True if you want to use a pre-trained model
        
        # TODO: Change every time you switch to the next model
        # Convert Keras parameters to PyTorch equivalents (Set True if The model you want to fine tune is in TensorFlow/Keras format)
        "model_converter": False, # Set to True if you want to convert a Keras model to PyTorch or in case you want to change the 1st layer size
        "old_model_input_feature_size": 394, # Set to the number of input features that entered in the old model (e.g., 423 for Kinect, 394 for MP, 124 for upper body MP)
        
        "model_is_pytorch": True, # Set to True if the Model is in PyTorch format or False if it is in TensorFlow/Keras format
        
        # Path to the pre-trained model in TensorFlow/Keras format
        # "pretrained_model_path": "./ntu_benchmark_model/model",  # Path to the pre-trained model for NTU-120 one-shot benchmark
        # "pretrained_model_path": "./therapies_model_7/model",   # Path to the pre-trained model for the therapies dataset

        # TODO: Change every time you switch to the next model
        # "train_annotations": "./datasets_annotations/therapies_APPDA_MP_annotations.txt",
        "train_annotations": "./datasets_annotations/mp_train.txt",
        # "train_annotations": "./datasets_annotations/therapies_APPDA_MP_upper_body_annotations.txt",
        # "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        # "train_annotations": "./datasets_annotations/CADDIN_Final_Validation_MP_upper_body.txt",  # Set True to split the training data into K folds
        # "train_annotations": "./datasets_annotations/mp_train_upper_body.txt",
        
        # "val_annotations": "./datasets_annotations/mp_val.txt", # Set in case you don't use K-Fold Cross Validation
        # "final_validation_annotations": "./datasets_annotations/CADDIN_Final_MP_upper_body.txt",
        "final_validation_annotations": "./datasets_annotations/CADDIN_Final_Validation_MP_upper_body.txt",

        "train_compare_1": "./datasets_annotations/therapies_APPDA_MP_annotations.txt",
        "train_compare_2": "./datasets_annotations/mp_train.txt",

        # "eval_therapies": True,  # Therapy data needed for its evaluation
        "h_flip": True,
        # "skip_frames": [],
        "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        # "val_annotations": "",
        # "eval_therapies": False,
        # "h_flip": False,

        # TODO: Change every time you switch to the next model
        # Define which keys are *excluded from training* (i.e., frozen)
        "excluded_pt_keys": [
            # "encoder_net.encoder.0.residual_blocks.0.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.weight",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.bias",
            "encoder_net.encoder.0.residual_blocks.1.conv1.weight",
            "encoder_net.encoder.0.residual_blocks.1.conv1.bias",
            "encoder_net.encoder.0.residual_blocks.1.conv2.weight",
            "encoder_net.encoder.0.residual_blocks.1.conv2.bias",
            "encoder_net.encoder.0.residual_blocks.2.conv1.weight",
            "encoder_net.encoder.0.residual_blocks.2.conv1.bias",
            "encoder_net.encoder.0.residual_blocks.2.conv2.weight",
            "encoder_net.encoder.0.residual_blocks.2.conv2.bias",
            "encoder_net.encoder.0.residual_blocks.3.conv1.weight",
            "encoder_net.encoder.0.residual_blocks.3.conv1.bias",
            "encoder_net.encoder.0.residual_blocks.3.conv2.weight",
            "encoder_net.encoder.0.residual_blocks.3.conv2.bias",
            "encoder_net.encoder.0.residual_blocks.4.conv1.weight",
            "encoder_net.encoder.0.residual_blocks.4.conv1.bias",
            "encoder_net.encoder.0.residual_blocks.4.conv2.weight",
            "encoder_net.encoder.0.residual_blocks.4.conv2.bias",
            "encoder_net.encoder.0.residual_blocks.5.conv1.weight",
            "encoder_net.encoder.0.residual_blocks.5.conv1.bias",
            "encoder_net.encoder.0.residual_blocks.5.conv2.weight",
            "encoder_net.encoder.0.residual_blocks.5.conv2.bias",
            "clf_out.weight",
            "clf_out.bias",
        ],
        
        # Define which keys are replaced with new ones (i.e., re-initialized)
        "excluded_tf_prefixes": [
            # 'encoder_tcn/sequential/tcn/residual_block_0/conv1D_0/kernel:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/conv1D_0/bias:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/matching_conv1D/kernel:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/matching_conv1D/bias:0',
            
            'out_clf/kernel:0',
            'out_clf/bias:0',
        ],

        "in_memory_generator_train": False,
        "in_memory_generator_val": False,
        
        "joints_dim": 3,
        
        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,
        "max_seq_len": -32,
        "num_layers": 2,
        "num_neurons": 256,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True,
        "triplet": False,
        "decoder": False,
        "reverse_decoder": False,
        "clf_neurons": 0,
        "conv_params": [256, 4, 2, True, "causal", [4]],
        "is_tcn": False,
        "use_jcd_features": True,
        # "use_jcd_features": False,
        "use_speeds": False,
        "use_coords_raw": False,
        "use_coords": True,
        # "use_coords": True,
        "use_jcd_diff": False,
        "use_bone_angles": True,
        # "use_bone_angles": False,
        "use_bone_angles_cent": False,
        "average_wrong_skels": True,
        "average_wrong_skels_method": 'mean',
    }

    # Correct max_seq_len if negative for use in summary/testing
    if static_params['max_seq_len'] <= 0:
        print(
            f"Warning: static_params['max_seq_len'] is {static_params['max_seq_len']}. Using 32 as effective_seq_len for non-dataset parts.")
        static_params['effective_seq_len'] = 32
    else:
        static_params['effective_seq_len'] = static_params['max_seq_len']


    # --- Model Verification on unseen data ---
    # """
    # --- Path and Feature Calculation (Cleaned Up) ---
    # print("--- Initializing Parameters and Paths ---")
    static_params['path_model'] = train_utils.create_model_folder(
        static_params['path_results'], static_params['model_name']
    )
    os.makedirs(static_params['path_model'], exist_ok=True)

    # Calculate num_jcd_feats and add to static_params (for record-keeping and if get_num_feats uses it)
    try:
        static_params['num_jcd_feats'] = int(
            comb(static_params.get('joints_num', 23), 2))
    except Exception as e:
        print(
            f"Warning: Could not calculate num_jcd_feats using comb: {e}. Setting to None or a fallback.")
        static_params['num_jcd_feats'] = None  # Or a fallback value if critical

    # Calculate final num_feats using get_num_feats
    try:
        calculated_num_feats = get_num_feats(**static_params)
        # Check if a 'num_feats' was pre-set in static_params and if it differs.
        # The value from get_num_feats should be authoritative.
        if static_params.get('num_feats') != calculated_num_feats and static_params.get('num_feats') is not None:
            print(f"Warning: Initial static_params['num_feats'] ({static_params.get('num_feats')}) "
                  f"differs from get_num_feats() calculation ({calculated_num_feats}). "
                  f"Using calculated value from get_num_feats: {calculated_num_feats}.")
        static_params['num_feats'] = calculated_num_feats
    except Exception as e:
        print(
            f"CRITICAL Error calling get_num_feats: {e}. num_feats might be incorrect.")
        if 'num_feats' not in static_params or static_params['num_feats'] is None:
            # If get_num_feats fails and no num_feats is set, it's a critical issue.
            print("CRITICAL: num_feats is not set and get_num_feats failed. Aborting.")
            exit()  # Exit if essential num_feats cannot be determined

    # Save the complete static_params to JSON
    static_params_save_path = os.path.join(
        static_params['path_model'], 'static_params.json')
    try:
        with open(static_params_save_path, 'w') as f:
            json.dump(static_params, f, indent=4)
        print(f"Saved model parameters to {static_params_save_path}")
    except Exception as e:
        print(f"Error saving static_params.json: {e}")

    # print(' * Final Model params for this run:',
    #       json.dumps(static_params, indent=2))

    copy_scaler_if_needed(static_params)

    # --- Annotation File Counts (for informational purposes) ---
    num_train_files, num_val_files = 0, 0
    try:
        if static_params.get('train_annotations'):
            with open(static_params['train_annotations'], 'r') as f:
                num_train_files = len(f.read().splitlines())
        if static_params.get('val_annotations'):
            with open(static_params['val_annotations'], 'r') as f:
                num_val_files = len(f.read().splitlines())
        print(
            f"Num train annotation lines: {num_train_files}, Num val annotation lines: {num_val_files}")
    except FileNotFoundError as e:
        print(f"Warning: Annotation file not found: {e}. Counts will be 0.")
    except Exception as e:
        print(
            f"Warning: Error reading annotation files: {e}. Counts might be inaccurate.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_name = static_params['model_name']
    # Create a directory for saving metrics
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
    
    # Save best parameters to a JSON file
    json_path = os.path.join(metrics_save_dir, "best_hyperparams.json")
    with open(json_path, "r") as f:
        best_params = json.load(f)

    model_params = {**static_params, **best_params}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # export_dataset_to_txt(model_params['train_compare_1'], model_params, "full_dataset_1_converted.txt")
    # export_dataset_to_txt(model_params['train_compare_2'], model_params, "full_dataset_2_converted.txt")

    # visualize_pose_dataset_2d(model_params['train_compare_1'], model_params['train_compare_2'], model_params, max_samples=2000, method="pca")
    # animate_first_sample_skeleton_3d(model_params['train_compare_2'], model_params)
    print(f"model_params['train_annotations']: {model_params['train_annotations']}")
    animate_first_sample_skeleton_3d(model_params['train_annotations'], model_params)
    exit(0)
    
    # --- Data Loading Therapist ---
    # Load the validation dataset once
    val_dataset = TripletPoseDataset(
        pose_annotations_file=model_params['final_validation_annotations'],
        validation_mode=False,
        in_memory=model_params['in_memory_generator_train'],
        **model_params
    )

    # import pdb; pdb.set_trace()

    # Create DataLoaders for this Data ---
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_params['batch_size'],
        shuffle=False,
        num_workers=model_params.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )

    all_fold_preds = []  # store predictions from each fold
    all_fold_labels = []  # store labels from each fold
    all_fold_sims = []  # store similarities from each fold
    all_fold_embs = []  # store embeddings from each fold
    
    print(f"num_feats: {model_params['num_feats']}, num_jcd_feats: {model_params.get('num_jcd_feats', 'N/A')}")
    for fold in range(5):  # 5 folds
        print(f"\n===== Fold {fold} =====")
        # --- Init model for this fold ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'],
            conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'],
            masking=model_params['masking'],
            triplet=model_params.get('triplet', False),
            classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'],
            num_classes=14
        )

        checkpoint_path = model_params["pretrained_model_path"]
        fold_model_path = get_k_fold_model_path(checkpoint_path, fold)
        pytorch_model.load_state_dict(torch.load(fold_model_path))
        pytorch_model.eval().to(device)

        # evaluate_model_on_all_data_mp(pytorch_model, 12, model_params, device, fold, metrics_save_dir)
        # continue  # Skip the rest for now
    
        # --- Collect predictions instead of embeddings ---
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # import pdb; pdb.set_trace()
                
                print("Batch shape:", batch_x.shape)   # <- add this
                print("Batch y shape:", batch_y.shape)   # <- add this
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)    

                logits = pytorch_model(batch_x)        # (B, num_classes)
                preds = torch.argmax(logits, dim=1)    # (B,)

                all_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())

        # Concatenate across batches
        all_preds = torch.cat(all_preds, dim=0)   # (N,)
        all_labels = torch.cat(all_labels, dim=0) # (N,)

        # Save fold results
        all_fold_preds.append(all_preds)
        all_fold_labels.append(all_labels)

        # --- Collect embeddings ---
        all_embs = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                print("Batch shape:", batch_x.shape)   # <- add this
                batch_x = batch_x.to(device)
                emb = pytorch_model.get_embedding(batch_x)  # (B, D)
                print("Embedding shape:", emb.shape)        # <- add this
                all_embs.append(emb.cpu())
        all_embs = torch.cat(all_embs, dim=0)  # (N, D)

        # --- Compute similarities pairwise ---
        sims = []
        for i in range(0, all_embs.shape[0], 2):
            if i + 1 < all_embs.shape[0]:
                sim = F.cosine_similarity(
                    all_embs[i].unsqueeze(0), all_embs[i+1].unsqueeze(0)
                )
                sims.append(round(sim.item(), 2))
        
        # Store for later averaging
        all_fold_embs.append(all_embs)
        all_fold_sims.append(sims)

        # 🔍 Show results for this fold
        print(f"Fold {fold} similarities ({len(sims)} pairs): {sims}")
        
        print("Predictions:", all_preds.tolist())
        print("True labels:", all_labels.tolist())
    
        # 🔍 Pairwise correctness
        pair_correct = 0
        for i in range(0, len(all_preds), 2):
            if i + 1 < len(all_preds):
                if (all_preds[i] == all_labels[i]) and (all_preds[i+1] == all_labels[i+1]):
                    pair_correct += 1

        num_pairs = len(all_preds) // 2
        print(f"Pairs correct: {pair_correct}/{num_pairs}")

        # 🔍 Individual correctness
        ind_correct = (all_preds == all_labels).sum().item()
        print(f"Individual correct: {ind_correct}/{len(all_preds)}")

    # --- Aggregate results at the end ---
    all_fold_sims = torch.tensor(all_fold_sims)  # shape (num_folds, num_pairs)
    mean_sims = all_fold_sims.mean(dim=0)
    mean_sims_rounded = (mean_sims * 100).round() / 100   # 2 decimals
    overall_mean = round(mean_sims.mean().item(), 2)

    print("\n===== Final Averages =====")
    print("Per-pair mean similarity across folds:", [f"{x:.2f}" for x in mean_sims])
    print(f"Overall mean similarity: {overall_mean:.2f}")
    
    exit(0)
    # """

    # Create Optuna study
    study = optuna.create_study(direction="maximize")  # Or "minimize" for loss
    study.optimize(partial(objective, static_params=static_params), n_trials=static_params.get("n_trials", 40))  # Try 40 different combinations#   
    print("\n--- Hyperparameter Optimization Finished ---")

    best_trial = study.best_trial
    best_val_f1 = best_trial.value
    get_best_model_number = static_params.get("best_model_number", 'N/A')
    print("Best trial F1:", best_val_f1)
    print("Best model number:", get_best_model_number)

    # Determine folder one level up in order to save the best hyperparameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_name = static_params['model_name']

    # Create a directory for saving metrics
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
    os.makedirs(metrics_save_dir, exist_ok=True)
    
    # Save best parameters to a JSON file
    best_params = best_trial.params
    with open(os.path.join(metrics_save_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    
    # ----------------------------
    # Clean up unrelated files
    # ----------------------------
    for filename in os.listdir(metrics_save_dir):
        # Keep only .npz or .png that include the correct model number
        if filename.endswith(('.npz', '.png', '.npy')) and f"model_{get_best_model_number}_" in filename:
            continue  # Keep this file
        if filename == 'best_hyperparams.json':
            continue  # Keep the JSON too
        # Otherwise, delete it
        file_path = os.path.join(metrics_save_dir, filename)
        os.remove(file_path)
    # ----------------------------
    
    # ----------------------------
    # Clean up old model folders
    # ----------------------------
    model_dir = os.path.join(static_params['path_results'], static_params['model_name'])
    for foldername in os.listdir(model_dir):        
        folder_path = os.path.join(model_dir, foldername)
        # Only consider directories and ignore the best model folder
        if os.path.isdir(folder_path) and f"model_{get_best_model_number}" not in foldername:
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted old model folder: {folder_path}")
            except Exception as e:
                print(f"Error deleting folder {folder_path}: {e}")
    # ----------------------------
    
    """     
    # FOR LAST RUN
    
    # Set the best F1 score and K for the model parameters
    static_params["best_val_f1"] = study.best_value
    static_params["K"] = None

    # Combine the two
    model_params = {**static_params, **best_params}

    # Run the main training function with the best parameters
    print("\n--- Running main training with best parameters ---")
    _, model_number = main(model_params)
    print(f"\nBest model number: {model_number}")
    
    # Get the best model path for the given model number
    best_model_path = get_final_model_path(
        model_params["path_results"],
        model_params["model_name"],
        model_number
    )
    print(f"Best model path: {best_model_path}")
    """
    
    print("\n--- Training script finished ---")
    
    # TODO: Create an if function based on what param you want to load to train a new model
    # Determine folder one level up in order to save the best hyperparameters
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    # model_name = static_params['model_name']
    # # Create a directory for saving metrics
    # metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison', model_name)
    # os.makedirs(metrics_save_dir, exist_ok=True)
    # 
    # # Save best parameters to a JSON file
    # with open("best_hyperparams.json", "r") as f:
    #     best_params = json.load(f)
    # 
    # # Set the best F1 score and K for the model parameters
    # static_params["best_val_f1"] = 0
    # static_params["K"] = None
    # 
    # # Combine the two
    # model_params = {**static_params, **best_params}
    # 
    # main(model_params)
    # 
    # exit(0)    