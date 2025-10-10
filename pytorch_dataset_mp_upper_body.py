# pytorch_dataset.py

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
from scipy.special import comb # Used in helper get_jcd_features, get_num_feats
from scipy.spatial.distance import cdist # Used in helper get_jcd_features
import scipy.ndimage.interpolation as inter # Used in helper zoom_to_target_len

# =============================================================================
# Helper Functions (Ported or Adapted from Keras data_generator.py)
# These should be the same as in pytorch_dataset_py_01
# Ensure they are reviewed for TF/Keras dependencies.
# =============================================================================

"""
FLIP_CORRESPONDENCES_LEFT = [8, 7, 6, 2, 1, 0, 20, 21, 23]
FLIP_CORRESPONDENCES_RIGHT = [9, 10, 11, 3, 4, 5, 17, 20, 22]

# Pelvis → Mid Spine → Upper Spine → Neck → Head → Top Head
SPINE = [12, 14, 15, 16, 17, 18]
TORSO_CONNECTING_JOINTS = [14, 12]
# TORSO_CONNECTING_JOINTS = [14, 16, 
#                            16, 15, 
#                            15, 12]

LEFT_HIP = 2 # Old index 12
RIGHT_HIP = 3 # Old index 16
SPINE_CHEST = 12 # Old index 20

CONNECTING_JOINT = [s
    1, 0,       # Check CRMH skeleton
    1, 2,       #
    2, 14,      #
    3, 14,      #
    3, 4,       #
    4, 5,       #
    14, 16,     #
    15, 16,     #
    12, 15,     #
    6, 7,       #
    7, 8,       #
    8, 12,      #
    9, 12,      #
    9, 10,      #
    10, 11,     #
    12, 17,     #
    17, 18,     #
    18, 19,     #
    13, 19,     #
    21, 23,     #
    19, 21,     #
    19, 20,     #
    20, 22      #
]
"""

"""
# --- Constants (from Keras code) ---
# FLIP_CORRESPONDENCES_LEFT = [4, 5, 6, 7, 12, 13, 14, 15, 21, 22]
# FLIP_CORRESPONDENCES_RIGHT = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
# SPINE = [0, 1, 2, 3, 20]
# CONNECTING_JOINT = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9,
#                     10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11] # Used in get_body_spherical_angles
# 
# TORSO_CONNECTING_JOINTS = [20, 1,
#                            1, 0]
# 
# # TORSO_CONNECTING_JOINTS = [14, 16, 
# #                            16, 15, 
# #                            15, 12]
# 
# LEFT_HIP = 12 # Old index 12
# RIGHT_HIP = 16 # Old index 16
# SPINE_CHEST = 20 # Old index 20
"""

# --- Define dropped joints ---
drop_joints = {0, 1, 2, 3, 4, 5, 13, 19, 20, 21, 22, 23}
keep_indices = [j for j in range(24) if j not in drop_joints]
old_to_new = {old: new for new, old in enumerate(keep_indices)}

# print("Kept joints:", keep_indices)
# print("Old → New mapping:")
# for old, new in old_to_new.items():
    # print(f"  {old:2d} → {new:2d}")


# --- Remap flip correspondences ---
FLIP_CORRESPONDENCES_LEFT = [old_to_new[j] for j in [8, 7, 6] if j in old_to_new]
FLIP_CORRESPONDENCES_RIGHT = [old_to_new[j] for j in [9, 10, 11] if j in old_to_new]

# print("\nFlip correspondences:")
# for l, r in zip(FLIP_CORRESPONDENCES_LEFT, FLIP_CORRESPONDENCES_RIGHT):
    # print(f"  L {l} ↔ R {r}")

# Example: keep only existing spine joints
SPINE = [old_to_new[j] for j in [12, 14, 15, 16, 17, 18] if j in old_to_new]
# print("\nSpine joints (new indices):", SPINE)

# --- Connecting joints ---
CONNECTING_JOINT_OLD = [
    1, 0, 1, 2, 2, 14, 3, 14, 3, 4, 4, 5,
    14, 16, 15, 16, 12, 15, 6, 7, 7, 8, 8, 12,
    9, 12, 9, 10, 10, 11, 12, 17, 17, 18, 18, 19,
    13, 19, 21, 23, 19, 21, 19, 20, 20, 22
]

CONNECTING_JOINT = []
remap_pairs = []  # for printing pairs
for i in range(0, len(CONNECTING_JOINT_OLD), 2):
    j1, j2 = CONNECTING_JOINT_OLD[i], CONNECTING_JOINT_OLD[i+1]
    if j1 in old_to_new and j2 in old_to_new:
        CONNECTING_JOINT.extend([old_to_new[j1], old_to_new[j2]])
        remap_pairs.append((j1, j2, old_to_new[j1], old_to_new[j2]))

TORSO_CONNECTING_JOINTS = [old_to_new[j] for j in [14, 12] if j in old_to_new]
SPINE_CHEST = [old_to_new[j] for j in [12] if j in old_to_new]
LEFT_HIP = None # Old index 12
RIGHT_HIP = None # Old index 16

# print("\nConnecting joints (old → new):")
# for old1, old2, new1, new2 in remap_pairs:
#     print(f"  ({old1:2d}, {old2:2d}) → ({new1:2d}, {new2:2d})")

# print("\nFinal CONNECTING_JOINT:", CONNECTING_JOINT)


# --- Helper Function Definitions (Copied from pytorch_dataset_py_01) ---

def get_scaler_filename(**params):
    # ... (Implementation from your Keras data_generator.py, ensuring it's TF-free) ...
    # This function constructs the scaler filename based on model_params
    # Example from your Keras code:
    path_prefix = params.get('scaler_path_prefix', '/home/asabater/datasets/NTU-120/data_scalers/') # Make configurable
    filename = 'std_msl{}_jn{}_jd{}_cskl{}_strs{}'.format(
        params.get('max_seq_len', -1), params.get('joints_num',25), params.get('joints_dim',3),
        'T' if params.get('center_skels',True) else 'F',
        'T' if params.get('scale_by_torso',True) else 'F')
    filename += '_jcd{}_spds{}_coordsraw{}_coords{}_jcddiff{}_angs{}_angscent{}_numfeats{}.pckl'.format(
        'T' if params.get('use_jcd_features',False) else 'F',
        'T' if params.get('use_speeds',False) else 'F',
        'T' if params.get('use_coords_raw',False) else 'F',
        'T' if params.get('use_coords',True) else 'F',
        'T' if params.get('use_jcd_diff',False) else 'F',
        'T' if params.get('use_bone_angles',False) else 'F',
        'T' if params.get('use_bone_angles_cent',False) else 'F',
        params.get('num_feats', 0) # Default to 100 if not specified
    )
    return os.path.join(path_prefix, filename)

def get_num_feats(joints_num, joints_dim,
                use_jcd_features, use_speeds, use_coords_raw, use_coords, use_jcd_diff,
                use_bone_angles, use_bone_angles_cent, **kwargs):

    num_feats = 0
    if use_bone_angles:
        num_feats += (joints_num-1)*2
    if use_bone_angles_cent:
        num_feats += (joints_num-1)*2
    if use_jcd_features:
        num_feats += int(comb(joints_num, 2))
    if use_speeds:
        num_feats += joints_num * joints_dim
    if use_coords_raw:
        num_feats += joints_num * joints_dim
    if use_coords:
        num_feats += joints_num * joints_dim
    if use_jcd_diff:
        num_feats += int(comb(joints_num, 2))

    # print(f"Calculated number of features: {num_feats} based on parameters: "
    #         f"joints_num={joints_num}, joints_dim={joints_dim}, "
    #         f"use_jcd_features={use_jcd_features}, use_speeds={use_speeds}, "
    #         f"use_coords_raw={use_coords_raw}, use_coords={use_coords}, "
    #         f"use_jcd_diff={use_jcd_diff}, use_bone_angles={use_bone_angles}, "
    #         f"use_bone_angles_cent={use_bone_angles_cent}")
    return num_feats

def load_skeleton_data(file_path):
    """Loads skeleton data from a .npy file."""
    try:
        pose_raw = np.load(file_path, allow_pickle=True).item()
        return pose_raw
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading skeleton data from {file_path}: {e}")
        return None

def get_body_skel(pose_raw, validation, mode='var'):
    """Selects the primary skeleton from potentially multiple bodies."""
    if pose_raw is None: return None
    n_bodys_val = pose_raw.get('nbodys', [0]) # Default to [0] if 'nbodys' key is missing
    n_bodys = list(set(n_bodys_val if isinstance(n_bodys_val, list) else [n_bodys_val]))

    if not n_bodys or max(n_bodys) == 0: # Handles empty or [0]
        if 'skel_body0' in pose_raw:
            return pose_raw['skel_body0']
        else:
            print(f"Warning: No valid skeleton data found (no 'nbodys' or 'skel_body0') in file associated with {pose_raw.get('filename', 'unknown_file')}")
            return None

    valid_body_indices = [i for i in range(max(n_bodys) + 1) if f'skel_body{i}' in pose_raw]
    if not valid_body_indices:
        print(f"Warning: No 'skel_bodyX' keys found despite nbodys info in {pose_raw.get('filename', 'unknown_file')}")
        return None

    body_lens = []
    valid_skeletons = []
    for i in valid_body_indices:
        skel = pose_raw[f'skel_body{i}']
        if skel is None or skel.ndim < 3 or skel.shape[0] == 0: # Check if skel is valid
            continue
        non_zero_frames = skel[np.all(~np.all(skel == 0, axis=2), axis=1)]
        if non_zero_frames.shape[0] > 0:
            body_lens.append(non_zero_frames.shape[0])
            valid_skeletons.append(skel)

    if not body_lens:
        print(f"Warning: All detected skeletons have zero length after filtering in {pose_raw.get('filename', 'unknown_file')}")
        # Fallback: return the first skeleton found, even if it was all zeros initially,
        # average_wrong_frame_skels might fix it.
        if valid_skeletons: return valid_skeletons[0]
        return None


    max_len = max(body_lens)
    longest_indices_in_valid_list = [idx for idx, length in enumerate(body_lens) if length == max_len]

    chosen_valid_list_idx = 0 # Default
    if longest_indices_in_valid_list: # Ensure list is not empty
        if validation:
            if mode == 'var' and len(longest_indices_in_valid_list) > 0 :
                stds = [valid_skeletons[idx].std() for idx in longest_indices_in_valid_list if valid_skeletons[idx].size > 0] # Check size
                if stds: # Ensure stds is not empty
                    chosen_valid_list_idx = longest_indices_in_valid_list[np.argmax(stds)]
                else: # Fallback if all stds are zero or skeletons were empty
                    chosen_valid_list_idx = longest_indices_in_valid_list[0]
            elif longest_indices_in_valid_list:
                chosen_valid_list_idx = longest_indices_in_valid_list[0]
        elif longest_indices_in_valid_list:
            chosen_valid_list_idx = np.random.choice(longest_indices_in_valid_list)
        else: # Should not be reached if body_lens was not empty
            return valid_skeletons[0] if valid_skeletons else None
    else: # No longest skeletons found (e.g. all had zero length effectively)
        return valid_skeletons[0] if valid_skeletons else None


    return valid_skeletons[chosen_valid_list_idx]

def average_wrong_frame_skels(skels):
    
    if skels is None or len(skels) == 0: return skels
    if skels.ndim < 3: # Expects (frames, joints, dims)
        print(f"Warning: average_wrong_frame_skels received unexpected shape {skels.shape}. Skipping.")
        return skels
    
    good_frames_mask = np.any(np.any(skels != 0, axis=2), axis=1)
    if np.all(good_frames_mask): return skels # All frames are good
    
    bad_indices = np.where(~good_frames_mask)[0]

    for idx in bad_indices:
        prev_good_idx = -1
        # Find previous good frame
        for i in range(idx - 1, -1, -1):
            if good_frames_mask[i]:
                prev_good_idx = i
                break
        
        next_good_idx = -1
        # Find next good frame
        for i in range(idx + 1, len(skels)):
            if good_frames_mask[i]:
                next_good_idx = i
                break

        if prev_good_idx != -1 and next_good_idx != -1:
            skels[idx] = (skels[prev_good_idx] + skels[next_good_idx]) / 2.0
        elif prev_good_idx != -1: # Only previous good frame exists
            skels[idx] = skels[prev_good_idx]
        elif next_good_idx != -1: # Only next good frame exists
            skels[idx] = skels[next_good_idx]
        # If no good frames around, it remains as is (likely all zeros)
        # Or one could implement a more sophisticated fill, e.g., with a default pose.
    return skels

def zoom_to_target_len(p, target_len, joints_num, joints_dim):
    num_frames = p.shape[0]
    if num_frames == target_len:
        return p
    if num_frames == 0:
        return np.zeros([target_len, joints_num, joints_dim], dtype=p.dtype)

    p_new = np.zeros([target_len, joints_num, joints_dim], dtype=p.dtype)
    zoom_factor = target_len / num_frames
    # print(f"p: {p[0:10]}, len: {len(p)}")
    for m in range(joints_num):
        for n in range(joints_dim):
            if num_frames > 0: # Ensure not dividing by zero
                # order=0 for nearest, order=1 for linear
                # print(f"p data (joint {m}, dim {n}): {p[:, m, n]}, len: {len(p[:, m, n])}")
                zoomed_data = inter.zoom(p[:, m, n], zoom_factor, mode='nearest', order=1)
                # print(f"Zoomed data (joint {m}, dim {n}): {zoomed_data}, len: {len(zoomed_data)}")
                # Adjust length if zoom results in slightly different size due to float precision
                if len(zoomed_data) > target_len:
                    p_new[:, m, n] = zoomed_data[:target_len]
                elif len(zoomed_data) < target_len:
                    p_new[:len(zoomed_data), m, n] = zoomed_data
                    # Pad the rest if necessary (though zoom should ideally match target_len)
                    p_new[len(zoomed_data):, m, n] = zoomed_data[-1] if len(zoomed_data) > 0 else 0 # Fill with last value or 0
                else:
                    p_new[:, m, n] = zoomed_data

            else: # Should be caught by num_frames == 0 earlier
                p_new[:,m,n] = 0
    # print(f"p_new: {p_new[0:10]}, len: {len(p_new)}")
    
    return p_new

def flip_skeleton(skel, flip_axis=0):
    skel_flipped = skel.copy()
    aux = skel_flipped[..., FLIP_CORRESPONDENCES_LEFT, :].copy() # Ensure it's a copy for swap
    skel_flipped[..., FLIP_CORRESPONDENCES_LEFT, :] = skel_flipped[..., FLIP_CORRESPONDENCES_RIGHT, :]
    skel_flipped[..., FLIP_CORRESPONDENCES_RIGHT, :] = aux
    
    # Flip the specified axis for all relevant joints (left, right, spine)
    # Create a combined list of all joints to be flipped
    joints_to_flip_on_axis = list(set(FLIP_CORRESPONDENCES_LEFT + FLIP_CORRESPONDENCES_RIGHT + SPINE))
    
    # Ensure indices are within bounds
    valid_joints_to_flip = [j for j in joints_to_flip_on_axis if j < skel_flipped.shape[-2]] # skel_flipped.shape[-2] is joints_num

    if valid_joints_to_flip:
        skel_flipped[..., valid_joints_to_flip, flip_axis] = -skel_flipped[..., valid_joints_to_flip, flip_axis]
    return skel_flipped

def scale_skel_by_torso(skel):
    if skel.shape[0] == 0: return skel
    if skel.shape[1] <= 20: # joints_num
        # print(f"Warning: scale_by_torso requires at least 21 joints, found {skel.shape[1]}. Skipping scaling.")
        return skel

    # torso_dists = np.linalg.norm(skel[:, 20] - skel[:, 1], axis=1) + \
    #             np.linalg.norm(skel[:, 1] - skel[:, 0], axis=1)
    # print("\n".join(
    #     f"skel[:, TORSO_CONNECTING_JOINTS[{i}]]: {skel[:, TORSO_CONNECTING_JOINTS[i]]}"
    #     for i in range(len(TORSO_CONNECTING_JOINTS))
    # ))
    
    # Compute the total torso distance dynamically
    torso_dists = sum(
        np.linalg.norm(skel[:, TORSO_CONNECTING_JOINTS[i]] - skel[:, TORSO_CONNECTING_JOINTS[i+1]], axis=1)
        for i in range(0, len(TORSO_CONNECTING_JOINTS), 2)
    )
    
    for i in range(skel.shape[0]):
        rel = 0.4 / torso_dists[i] if torso_dists[i] != 0 else 1
        skel[i] = skel[i] * rel

    return skel

def matrix_unit_vector(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0 # Avoid division by zero for zero vectors, result will be zero vector
    return matrix / norms

def get_transformation_matrix_global(skel):
    if skel.shape[1] <= 20: # joints_num
        # print(f"Warning: get_transformation_matrix_global requires at least 21 joints. Returning identity.")
        return np.array([np.eye(4)] * skel.shape[0])

    # Origin: Midpoint between hips (indices 12 and 16 for NTU RGB+D)
    # Original code used 12 and 16. Let's stick to that.
    o = (skel[:, LEFT_HIP, :] + skel[:, RIGHT_HIP, :]) / 2.0

    # X-axis: From left hip (16) to right hip (12)
    x_vec = skel[:, RIGHT_HIP, :] - skel[:, LEFT_HIP, :] # Vector from left hip to right hip
    
    # print(f"x_vec sample: {x_vec}") # Debug # print
    
    x_axis = matrix_unit_vector(x_vec)

    # print(f"x_axis sample: {x_axis}") # Debug print

    # Y-axis: Approximate vertical axis. Vector from spine base (0) to neck (3) or mid-shoulders.
    # Original code used skel[:,20] - o for Z. Let's try to define Y as up.
    # A common choice for Y is (SpineNavel (1) - SpineChest (20)) or similar vertical component.
    # Let's use (Head (3) - SpineBase (0)) as a proxy for "up" relative to the body.
    # Or more robustly, use cross product.
    # Original code: z_vec = skel[:, 20] - o (SpineChest - origin)
    # Let's assume the original intent for z_vec was a forward-facing vector.
    # If x_axis is rightward, and we want y_axis upward, then z_axis is forward.
    # Let's try to define Y as orthogonal to a plane defined by hips and a point above.
    # For simplicity, let's use the original Z-axis definition and derive Y.
    # Original Z axis: SpineChest (20) - Origin (mid-hip)
    z_approx_vec = skel[:,SPINE_CHEST,:] - o # Vector from mid-hip to spine/chest
    
    # print(f"z_approx_vec sample: {z_approx_vec}") # Debug print
    
    # Y-axis: Orthogonal to X and Z_approx (cross product)
    # y_axis = np.cross(z_approx_vec, x_axis) # Order matters for right/left-handed system
    y_axis = np.cross(x_axis, z_approx_vec) # If x is right, z_approx is somewhat forward, y is up
    y_axis = matrix_unit_vector(y_axis)

    # Z-axis: Orthogonal to X and Y
    z_axis = np.cross(x_axis, y_axis)
    z_axis = matrix_unit_vector(z_axis) # Should already be unit if x and y are unit and orthogonal

    r_matrices = []
    for i in range(len(skel)):
        rotation_inv = np.eye(4)
        rotation_inv[0, :3] = x_axis[i]
        rotation_inv[1, :3] = y_axis[i]
        rotation_inv[2, :3] = z_axis[i]
        
        translation_inv = np.eye(4)
        translation_inv[:3, 3] = o[i] # We want to translate points BY -o. So matrix has o in translation part.
                                    # If P' = M * P, M = Rot * Trans(-o).
                                    # We want M such that P_new = M_inv * P_orig_homo
                                    # Or P_new_xyz = R^T * (P_orig_xyz - o)
                                    # So, R is [x_axis, y_axis, z_axis]^T. Trans(-o) applied first.
                                    # The matrix r should be T(-o) followed by R.
                                    # Let's build the matrix that transforms FROM world TO local.
                                    # 1. Translate by -o
                                    # 2. Rotate by R^T
        
        # Matrix to transform points from world to local: R_transpose * Translation(-origin)
        # R_transpose rows are x_axis, y_axis, z_axis
        # T_neg_origin sets origin at (0,0,0)
        # Combined matrix M:
        # [x_axis_x, x_axis_y, x_axis_z, -dot(x_axis, origin)]
        # [y_axis_x, y_axis_y, y_axis_z, -dot(y_axis, origin)]
        # [z_axis_x, z_axis_y, z_axis_z, -Num JCD features:(z_axis, origin)]
        # [0         , 0         , 0         , 1             ]
        
        transform_matrix = np.eye(4)
        transform_matrix[0,:3] = x_axis[i]
        transform_matrix[1,:3] = y_axis[i]
        transform_matrix[2,:3] = z_axis[i]
        transform_matrix[:3,3] = -np.dot(transform_matrix[:3,:3], o[i]) # R^T * (-o)

        r_matrices.append(transform_matrix)
        
    return np.stack(r_matrices)

def transform_skel_global(skel, r_matrices):
    skel_h = np.concatenate([skel, np.ones((*skel.shape[:-1], 1))], axis=-1) # (Frames, Joints, 4)
    # r_matrices is (Frames, 4, 4)
    # We want to apply r_matrices[f] to skel_h[f] for each frame f
    # skel_h[f] is (Joints, 4). We want (Joints, 4) @ (4, 4)^T -> (Joints, 4)
    # or (4,4) @ (4, Joints) -> (4, Joints) then transpose
    
    transformed_skel_h_frames = []
    for i in range(skel_h.shape[0]): # Iterate through frames
        # skel_h[i] is (Joints, 4). r_matrices[i] is (4,4)
        # Transformed_points = Points @ TransformMatrix.T (if points are rows)
        # Or Transformed_points = TransformMatrix @ Points (if points are columns)
        # Here, r_matrices[i] is the matrix that transforms world to local.
        # So, local_coords_homo = r_matrices[i] @ world_coords_homo.T
        # This means world_coords_homo should be (4, num_joints)
        frame_world_coords_homo_T = skel_h[i].T # Shape (4, num_joints)
        frame_local_coords_homo_T = np.matmul(r_matrices[i], frame_world_coords_homo_T) # Shape (4, num_joints)
        transformed_skel_h_frames.append(frame_local_coords_homo_T.T) # Back to (num_joints, 4)

    transformed_skel_h = np.stack(transformed_skel_h_frames)
    return transformed_skel_h[..., :3]

def get_jcd_features(p, joints_num):
    num_frames = p.shape[0]
    if num_frames == 0:
        num_jcd_feats = int(comb(joints_num, 2)) if joints_num >= 2 else 0
        return np.zeros((0, num_jcd_feats), dtype=np.float32)
    
    jcd_list = []
    # iu provides upper triangle indices (row_idx, col_idx)
    iu_rows, iu_cols = np.triu_indices(joints_num, k=1)
    
    for f_idx in range(num_frames):
        frame_coords = p[f_idx] # Shape (joints_num, joints_dim)
        # Calculate pairwise distances for this frame
        # cdist(frame_coords, frame_coords) gives a (joints_num, joints_num) matrix
        # We need to extract the upper triangle from this.
        dist_matrix = cdist(frame_coords, frame_coords, 'euclidean')
        jcd_list.append(dist_matrix[iu_rows, iu_cols])
        
    return np.array(jcd_list, dtype=np.float32) if jcd_list else np.zeros((0, int(comb(joints_num, 2))), dtype=np.float32)

def get_bone_spherical_angles(v): # v is (num_frames, dims) or (dims) for a single vector
    v = np.atleast_2d(v) # Ensure v is at least 2D for consistent indexing
    norm_xy = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    # Avoid division by zero or log of zero for elevation if norm_xy is zero
    elevation = np.arctan2(v[:, 2], norm_xy, out=np.zeros_like(v[:,2]), where=norm_xy!=0)
    azimuth = np.arctan2(v[:, 1], v[:, 0])
    return np.column_stack([elevation, azimuth])

def get_body_spherical_angles(body_coords): # body_coords is (frames, joints, dims)
    num_frames = body_coords.shape[0]
    # print("num_frames in get_body_spherical_angles:", num_frames)
    if num_frames == 0:
        # Estimate number of bones from CONNECTING_JOINT. This is tricky.
        # Assuming CONNECTING_JOINT defines pairs or a sequence.
        # The original Keras code: (len(connecting_joint)-1)*2. This implies connecting_joint has one more element than bones.
        # Let's assume CONNECTING_JOINT implies len(CONNECTING_JOINT) bones.
        num_angle_feats = len(CONNECTING_JOINT) * 2 # Each bone gives 2 angles
        return np.zeros((0, num_angle_feats), dtype=np.float32)

    all_bone_angles_list = []
    # This definition of bones needs to be accurate.
    # The Keras code iterates `range(len(connecting_joint)-1)` implying `connecting_joint` defines a sequence.
    # Let's assume it means bones between `joint[i]` and `joint[connecting_joint[i]]`
    # Or more simply, bones between `joint[i]` and `joint[j]` where (i,j) are defined pairs.
    # The original Keras code `body[:, i+1] - body[:, i]` is a sequential definition.
    # Let's use a more robust definition if NTU standard bones are known, or stick to sequential.
    # For now, using sequential bones based on the loop in Keras: body[:, bone_idx+1] - body[:, bone_idx]
    # This assumes joints are ordered to form meaningful sequential bones.
    num_joints = body_coords.shape[1]
    # Iterate through defined bone connections (pairs of joints)
    for i in range(0, len(CONNECTING_JOINT), 2):
        joint_a = CONNECTING_JOINT[i]
        joint_b = CONNECTING_JOINT[i + 1]

        # Ensure indices are within bounds
        if joint_a < num_joints and joint_b < num_joints:
            # Compute bone vector (joint_b - joint_a) for all frames
            bone_vectors = body_coords[:, joint_b, :] - body_coords[:, joint_a, :]

            # Only process if there are frames
            if bone_vectors.shape[0] > 0:
                spherical_angles = get_bone_spherical_angles(bone_vectors)
                all_bone_angles_list.append(spherical_angles)

    if not all_bone_angles_list:
        return np.zeros((num_frames, 0), dtype=np.float32)

    return np.concatenate(all_bone_angles_list, axis=1)

def get_pose_data_processed(body_raw, is_validation, model_params):
    if body_raw is None or body_raw.shape[0] == 0:
        print("Warning: get_pose_data_processed received None or empty body_raw.")
        # Return a zero array of expected feature dimension and a plausible sequence length
        _target_len_fallback = abs(model_params.get('max_seq_len', 32))
        if _target_len_fallback == 0: _target_len_fallback = 32
        _num_feats_fallback = model_params.get('num_feats', 100) # Get this from model_params or calculate
        return np.zeros((_target_len_fallback, _num_feats_fallback), dtype=np.float32)

    max_seq_len_param = model_params.get('max_seq_len', -32)
    joints_num = model_params.get('joints_num', 25)
    joints_dim = model_params.get('joints_dim', 3)
    center_skels = model_params.get('center_skels', True)
    h_flip_enabled = model_params.get('h_flip', False)
    scale_by_torso_enabled = model_params.get('scale_by_torso', True)
    temporal_scale_range = model_params.get('temporal_scale', False)
    skip_frames_options = model_params.get('skip_frames', [])
    average_wrong_skels_enabled = model_params.get('average_wrong_skels', True)
    scaler_obj = model_params.get('scaler_object', None) # Get scaler from params

    # print(f"max_seq_len_param: {max_seq_len_param}")
    # print(f"joints_num: {joints_num}, joints_dim: {joints_dim}")
    # print(f"center_skels: {center_skels}, h_flip_enabled: {h_flip_enabled}, scale_by_torso_enabled: {scale_by_torso_enabled}")
    # print(f"temporal_scale_range: {temporal_scale_range}, skip_frames_options: {skip_frames_options}, average_wrong_skels_enabled: {average_wrong_skels_enabled}, scaler_obj: {scaler_obj}")
    
    body = body_raw.copy()
    # print(f"Raw body shape: {body.shape}")
    # print(f"body.shape[0]: {body.shape[0]}")
    
    body = body[np.any(np.any(body != 0, axis=2), axis=1)]
    if body.shape[0] == 0:
        # print("Warning: Skeleton zero length after initial zero-frame removal.")
        _target_len_fallback = abs(max_seq_len_param) if max_seq_len_param != 0 else 32
        _num_feats_fallback = model_params.get('num_feats', 100)
        return np.zeros((_target_len_fallback, _num_feats_fallback), dtype=np.float32)

    if average_wrong_skels_enabled:
        body = average_wrong_frame_skels(body)
        if body is None or body.shape[0] == 0:
            # print("Warning: Skeleton zero length after averaging wrong frames.")
            _target_len_fallback = abs(max_seq_len_param) if max_seq_len_param != 0 else 32
            _num_feats_fallback = model_params.get('num_feats', 100)
            return np.zeros((_target_len_fallback, _num_feats_fallback), dtype=np.float32)

    if not is_validation:
        # print(f"isinstance(temporal_scale_range, (list, tuple)): {isinstance(temporal_scale_range, (list, tuple))}")
        if temporal_scale_range and isinstance(temporal_scale_range, (list, tuple)) and len(temporal_scale_range) == 2:
            orig_len = body.shape[0]
            min_scale, max_scale = temporal_scale_range
            # print(f"Temporal scaling: orig_len={orig_len}, min_scale={min_scale}, max_scale={max_scale}")
            if min_scale < max_scale and orig_len > 1 : # Need at least 2 frames to scale meaningfully
                scale_factor = np.random.uniform(min_scale, max_scale)
                # print(f"Chosen scale_factor: {scale_factor}")
                new_len = max(2, int(round(orig_len * scale_factor)))
                if new_len != orig_len:
                    body = zoom_to_target_len(body, new_len, joints_num, joints_dim)
        # print(f"Body shape after temporal scaling: {body.shape}")

        if skip_frames_options and body.shape[0] > 1:
            # Ensure skip_rate is at least 1 (no skip)
            skip_rate_choice = [s for s in skip_frames_options if isinstance(s, int) and s > 0]
            if skip_rate_choice:
                skip_rate = np.random.choice(skip_rate_choice)
                if skip_rate > 1 and body.shape[0] >= skip_rate : # Check if subsampling is possible
                    start_frame = np.random.randint(skip_rate)
                    body = body[start_frame::skip_rate]
                    if body.shape[0] == 0: # After skipping, it might become empty
                        body = body_raw.copy()[0:1] # Fallback to first frame of original raw
                        print("Warning: Body became empty after skip_frames, using fallback.")
        # print(f"Body shape after skip frames: {body.shape}")
        
        if h_flip_enabled and np.random.rand() > 0.5:
            body = flip_skeleton(body)

    # Determine final sequence length for padding/cropping
    final_target_len = abs(max_seq_len_param)
    if max_seq_len_param == 0: # Use actual length if max_seq_len is 0
        final_target_len = body.shape[0]
        if final_target_len == 0: final_target_len = 1 # Ensure at least 1 frame for feature extraction
    
    current_len = body.shape[0]
    # print(f"Body shape before length adjustment: {body.shape}, current_len: {current_len}, final_target_len: {final_target_len}")
    if max_seq_len_param > 0: # Zoom to fixed length (max_seq_len_param is positive)
        if current_len != final_target_len and final_target_len > 0:
            body = zoom_to_target_len(body, final_target_len, joints_num, joints_dim)
    elif max_seq_len_param < 0: # Crop if longer, Pad if shorter (max_seq_len_param is negative, use its abs value)
        target_crop_len = abs(max_seq_len_param)
        if current_len > target_crop_len:
            if not is_validation:
                start = np.random.randint(current_len - target_crop_len + 1) if (current_len - target_crop_len + 1) > 0 else 0
            else:
                start = (current_len - target_crop_len) // 2
            body = body[start : start + target_crop_len]
            # print(f"Cropped body from {current_len} to {target_crop_len}, start at {start}")
        elif current_len < target_crop_len and current_len > 0 : # Pad if shorter but not empty
            pad_width = target_crop_len - current_len
            padding = [(pad_width, 0), (0, 0), (0, 0)] # Pre-padding
            body = np.pad(body, padding, mode='constant', constant_values=0.0)
            # print(f"Padded body from {current_len} to {target_crop_len}")
        elif current_len == 0 and target_crop_len > 0: # If body became empty, pad to target_crop_len
            body = np.zeros((target_crop_len, joints_num, joints_dim), dtype=body_raw.dtype)

    if body.shape[0] == 0 and final_target_len > 0: # If still empty, create zeros
        print(f"Warning: Body is empty before feature extraction. Creating zeros of length {final_target_len}.")
        body = np.zeros((final_target_len, joints_num, joints_dim), dtype=body_raw.dtype)
    elif body.shape[0] == 0 and final_target_len == 0: # Should not happen if final_target_len defaults to 1
        _num_feats_fallback = model_params.get('num_feats', 100)
        return np.zeros((1, _num_feats_fallback), dtype=np.float32)

    if scale_by_torso_enabled:
        body = scale_skel_by_torso(body)

    body_uncentered_for_raw_coords = body.copy() if model_params.get('use_coords_raw', False) else None
    skels_to_process = body # Default to 'body' which might be uncentered

    if center_skels:
        transf_matrix = get_transformation_matrix_global(body) # Use original body for transform calc
        skels_to_process = transform_skel_global(body, transf_matrix) # Centered version for features

    # --- Feature Extraction ---
    num_frames_for_features = skels_to_process.shape[0]
    if num_frames_for_features == 0 and final_target_len > 0: # If skels_to_process became empty
        skels_to_process = np.zeros((final_target_len, joints_num, joints_dim), dtype=body.dtype)
        num_frames_for_features = final_target_len
    elif num_frames_for_features == 0 and final_target_len == 0:
        _num_feats_fallback = model_params.get('num_feats', 100)
        return np.zeros((1, _num_feats_fallback), dtype=np.float32)

    pose_features_list = []

    if model_params.get('use_bone_angles', False):
        # Use original body (before centering) for non-centered bone angles
        angles = get_body_spherical_angles(body)
        if angles.shape[0] == num_frames_for_features : pose_features_list.append(angles)
        elif angles.shape[0] > 0 : pose_features_list.append(zoom_to_target_len(angles.reshape(angles.shape[0], -1, 1), num_frames_for_features, angles.shape[1], 1).reshape(num_frames_for_features, -1))
        # print(f"Angles shape: {angles.shape}, num_frames_for_features: {num_frames_for_features}")

    if model_params.get('use_bone_angles_cent', False):
        angles_cent = get_body_spherical_angles(skels_to_process)
        if angles_cent.shape[0] == num_frames_for_features: pose_features_list.append(angles_cent)
        elif angles_cent.shape[0] > 0 : pose_features_list.append(zoom_to_target_len(angles_cent.reshape(angles_cent.shape[0], -1, 1), num_frames_for_features, angles_cent.shape[1], 1).reshape(num_frames_for_features, -1))
        # print(f"Angles_cent shape: {angles_cent.shape}, num_frames_for_features: {num_frames_for_features}")

    if model_params.get('use_coords_raw', False):
        if body_uncentered_for_raw_coords is not None:
            coords_raw_reshaped = body_uncentered_for_raw_coords.reshape(body_uncentered_for_raw_coords.shape[0], -1)
            if coords_raw_reshaped.shape[0] == num_frames_for_features: pose_features_list.append(coords_raw_reshaped)
            elif coords_raw_reshaped.shape[0] > 0 : pose_features_list.append(zoom_to_target_len(coords_raw_reshaped.reshape(coords_raw_reshaped.shape[0], -1, 1), num_frames_for_features, coords_raw_reshaped.shape[1], 1).reshape(num_frames_for_features, -1))
            # print(f"Raw coords shape: {coords_raw_reshaped.shape}, num_frames_for_features: {num_frames_for_features}")

    if model_params.get('use_coords', True): # Default to True if not specified
        coords_reshaped = skels_to_process.reshape(num_frames_for_features, -1)
        pose_features_list.append(coords_reshaped)
        # print(f"Coords shape: {coords_reshaped.shape}, num_frames_for_features: {num_frames_for_features}")

    jcd_calculated_feats = None
    if model_params.get('use_jcd_features', False) or model_params.get('use_jcd_diff', False):
        jcd_calculated_feats = get_jcd_features(skels_to_process, joints_num)
        if model_params.get('use_jcd_features', False):
            if jcd_calculated_feats.shape[0] == num_frames_for_features: pose_features_list.append(jcd_calculated_feats)
            elif jcd_calculated_feats.shape[0] > 0 : pose_features_list.append(zoom_to_target_len(jcd_calculated_feats.reshape(jcd_calculated_feats.shape[0], -1, 1), num_frames_for_features, jcd_calculated_feats.shape[1], 1).reshape(num_frames_for_features, -1))
            # print(f"JCD features shape: {jcd_calculated_feats.shape}, num_frames_for_features: {num_frames_for_features}")

    if model_params.get('use_jcd_diff', False):
        num_jcd_comb = int(comb(joints_num, 2)) if joints_num >= 2 else 0
        if jcd_calculated_feats is not None and jcd_calculated_feats.shape[0] > 1:
            jcd_diff_val = jcd_calculated_feats[1:] - jcd_calculated_feats[:-1]
            jcd_diff_val = np.concatenate([jcd_diff_val[0:1] if jcd_diff_val.shape[0] > 0 else np.zeros((1, num_jcd_comb)), jcd_diff_val], axis=0) # Prepend
        else:
            jcd_diff_val = np.zeros((num_frames_for_features, num_jcd_comb), dtype=skels_to_process.dtype)
        if jcd_diff_val.shape[0] == num_frames_for_features : pose_features_list.append(jcd_diff_val)
        elif jcd_diff_val.shape[0] > 0 : pose_features_list.append(zoom_to_target_len(jcd_diff_val.reshape(jcd_diff_val.shape[0], -1, 1), num_frames_for_features, jcd_diff_val.shape[1], 1).reshape(num_frames_for_features, -1))
        # print(f"JCD diff shape: {jcd_diff_val.shape}, num_frames_for_features: {num_frames_for_features}")

    if model_params.get('use_speeds', False):
        if num_frames_for_features > 1:
            speed_feats_val = skels_to_process[1:] - skels_to_process[:-1]
            speed_feats_val = np.concatenate([speed_feats_val[0:1] if speed_feats_val.shape[0] > 0 else np.zeros((1, joints_num, joints_dim)), speed_feats_val], axis=0) # Prepend
            speed_feats_val = speed_feats_val.reshape(num_frames_for_features, -1)
        else:
            speed_feats_val = np.zeros((num_frames_for_features, joints_num * joints_dim), dtype=skels_to_process.dtype)
        pose_features_list.append(speed_feats_val)
        # print(f"Speed features shape: {speed_feats_val.shape}, num_frames_for_features: {num_frames_for_features}")

    if not pose_features_list:
        # print("Warning: No features were extracted! Returning zeros.")
        # Ensure num_feats is available in model_params for fallback
        _num_feats_fallback = model_params.get('num_feats', 100)
        return np.zeros((final_target_len if final_target_len > 0 else 1, _num_feats_fallback), dtype=np.float32)

    # Ensure all feature arrays in list have the same number of frames (num_frames_for_features)
    # This should be guaranteed by processing steps if sequence length handling is correct.
    # If not, one might need to resize them here before concatenation.
    # For now, assume they are all num_frames_for_features long.

    pose_features_final = np.concatenate(pose_features_list, axis=1).astype(np.float32)
    # print(f"Final concatenated features shape before scaling: {pose_features_final.shape}")
    
    if scaler_obj is not None:
        try:
            pose_features_final = scaler_obj.transform(pose_features_final)
        except Exception as e:
            print(f"Warning: Error applying scaler: {e}. Features not scaled.")

    # Final check on sequence length and feature dimension
    expected_num_feats = model_params.get('num_feats', -1)
    if expected_num_feats != -1 and pose_features_final.shape[1] != expected_num_feats:
        print(f"Warning: Mismatch in feature dimension! Expected {expected_num_feats}, got {pose_features_final.shape[1]}. This can cause errors.")
        # Option: Pad/truncate features, or error out, or re-calculate expected_num_feats based on selected features.
        # This often indicates an issue in get_num_feats or feature selection logic.

    # Ensure final output matches final_target_len (especially if max_seq_len was 0)
    if pose_features_final.shape[0] != final_target_len and final_target_len > 0 :
        print(f"Warning: Final processed features length ({pose_features_final.shape[0]}) != final_target_len ({final_target_len}). Resizing features.")
        # Reshape to (frames, features, 1) for zoom, then back
        current_feat_dim = pose_features_final.shape[1]
        pose_features_final_reshaped = pose_features_final.reshape(pose_features_final.shape[0], current_feat_dim, 1)
        pose_features_final = zoom_to_target_len(pose_features_final_reshaped, final_target_len, current_feat_dim, 1).reshape(final_target_len, current_feat_dim)

    elif final_target_len == 0 and pose_features_final.shape[0] == 0: # Handle case where max_seq_len=0 and input was empty
        _num_feats_fallback = model_params.get('num_feats', 100)
        return np.zeros((1, _num_feats_fallback), dtype=np.float32) # Return at least one frame

    return pose_features_final

def zoom_to_max_len(p, max_seq_len, joints_num, joints_dim, force=False):
    # Resize movement
    num_frames = p.shape[0]
    if force or num_frames > max_seq_len:
        # Zoom -> crop movement
        p_new = np.zeros(
            [max_seq_len, joints_num, joints_dim], dtype="float32")
        for m in range(joints_num):
            for n in range(joints_dim):
                # smooth coordinates
                # Zoom coordinates to fit the max_seq_len_shape
                p_new[:, m, n] = inter.zoom(
                    # , mode='nearest'
                    p[:, m, n], max_seq_len/num_frames)[:max_seq_len]
    else:
        p_new = p
    return p_new

def get_jcd_features1(p, joints_num, max_seq_len):
    # Get joint distances
    jcd = []
    iu = np.triu_indices(joints_num, 1, joints_num)
    for f in range(max_seq_len):
        d_m = cdist(p[f], p[f], 'euclidean')
        d_m = d_m[iu]
        jcd.append(d_m)
    jcd = np.stack(jcd)

    return jcd

def get_pose_data_v2(body, max_seq_len, joints_num, joints_dim, center_skels,
                     h_flip, scale_by_torso, temporal_scale, scaler,
                     validation,
                     use_jcd_features, use_speeds,
                     use_coords_raw, use_coords, use_jcd_diff,
                     use_bone_angles,
                     use_bone_angles_cent,

                     skip_frames=[],
                     **kwargs):

    # Remove frames without predictions
    body = body[np.all(~np.all(body == 0, axis=2), axis=1)]
    # body = body[body.sum(axis=1).sum(axis=1)!=0]

    # Crop or extend the movement by interpolation
    # If extension is longer than max_seq_len, crop to max_seq_len
    if not validation and temporal_scale is not False:
        orig_new_frames = len(body)
        temporal_scale = list(temporal_scale)
        temporal_scale[0] = int(temporal_scale[0]*orig_new_frames)
        temporal_scale[1] = int(temporal_scale[1]*orig_new_frames)
        new_num_frames = np.random.randint(*temporal_scale)
        new_num_frames = max(new_num_frames, 2)

        zoom_factor = new_num_frames/orig_new_frames
        body = inter.zoom(body, (zoom_factor, 1, 1), mode='nearest')

    # Reduce frame rate
    if len(skip_frames) > 0:
        # print('aaaa', len(body))
        sk = np.random.choice(skip_frames)
        if validation:
            sk_init = 0
        else:
            sk_init = np.random.randint(sk)
        body = body[sk_init::sk]
        # print('bbbb', len(body))

    if max_seq_len > 0:
        # If movement is longer than max_seq_lenght -> crop to max_seq_length
        body = zoom_to_max_len(body, max_seq_len, joints_num, joints_dim)

    elif max_seq_len < 0:
        if not validation:
            # Crop randomly the movement to -max_seq_length
            start = np.random.randint(max(len(body)-abs(max_seq_len)+1, 1))
            end = start + abs(max_seq_len)
            body = body[start:end]
        else:
            # Crop to the last part of the movement
            start = max(0, (len(body) - abs(max_seq_len)) // 2)
            end = start + abs(max_seq_len)
            body = body[start:end]

    if scale_by_torso:
        body = scale_skel_by_torso(body)

    num_frames = len(body)
    # jcd_features, speed_features = [], []

    if not validation and h_flip and np.random.rand() > 0.5:
        body = flip_skeleton(body)

    body_before_center = body.copy()

    if center_skels:
        # Get transformation matrix

        r = get_transformation_matrix_global(body)

        skels = transform_skel_global(body, r)
        if use_speeds:
            skels_next = transform_skel_global(body[1:], r[:-1])

    else:
        skels = body
        if use_speeds:
            skels_next = body[1::]

    pose_features = []

    if use_bone_angles:     # 24*4
        # Elevation and azimuth for each bone (vector of consecutive joints)
        pose_features.append(get_body_spherical_angles(body))
    if use_bone_angles_cent:     # 24*4
        # Elevation and azimuth for each bone (vector of consecutive joints)
        pose_features.append(get_body_spherical_angles(skels))
    if use_coords_raw:  # 75 = 25*3
        # Raw coordinates
        pose_features.append(np.reshape(
            body_before_center, (num_frames, joints_num * joints_dim)))
    if use_coords:  # 75 = 25*3
        # Raw coordinates
        pose_features.append(np.reshape(
            skels, (num_frames, joints_num * joints_dim)))

    if use_jcd_diff or use_jcd_features:
        jcd_features = get_jcd_features1(skels, joints_num, num_frames)

        if use_jcd_diff:  # 300 = comb(25,2)
            # Distance difference between frames per each pair of joints
            jcd_diff = jcd_features[1:] - jcd_features[:-1]
            jcd_diff = np.reshape(
                jcd_diff, (num_frames-1, jcd_features.shape[-1]))
            jcd_diff = np.concatenate(
                [np.expand_dims(jcd_diff[0], axis=0), jcd_diff], axis=0)
            # print('Adding: use_jcd_diff')
            pose_features.append(jcd_diff)

        if use_jcd_features:  # 300 = comb(25,2)
            # Per-frame Joint distances
            pose_features.append(jcd_features)

    if use_speeds:  # 75 = 25*3
        # Frame-to-frame speeds
        speed_features = skels_next - skels[:-1]
        speed_features = np.reshape(
            speed_features, (num_frames-1, joints_num*joints_dim))

        speed_features = np.concatenate(
            [np.expand_dims(speed_features[0], axis=0), speed_features], axis=0)
        pose_features.append(speed_features)

    # pose_features = np.concatenate([jcd_features, speed_features], axis=1)
    pose_features = np.concatenate(pose_features, axis=1).astype('float32')

    if scaler is not None:
        pose_features = scaler.transform(pose_features)

    return pose_features



# =============================================================================
# PyTorch Dataset Class (Revised for Single Sample Output)
# =============================================================================

class TripletPoseDataset(Dataset):
    def __init__(self, pose_annotations_file, validation_mode, in_memory, **model_params_kwargs):
        self.pose_annotations_file = pose_annotations_file
        self.is_validation = validation_mode
        self.in_memory_data = in_memory
        self.model_params = model_params_kwargs

        print(f"\nInitializing PoseDataset (Single Sample Output):")
        print(f"  Annotations: {self.pose_annotations_file}")
        print(f"  Validation Mode: {self.is_validation}")
        print(f"  In Memory: {self.in_memory_data}")

        self.samples = self._read_annotations()
        if not self.samples:
            raise ValueError(f"No samples found or loaded from annotation file: {pose_annotations_file}")

        self.scaler = None
        if self.model_params.get('scale_data', False):
            try:
                scaler_path = self.model_params.get('scaler_path_override', None) # Allow override for testing
                if not scaler_path:
                    # Attempt to use get_scaler_filename if available and TF-free
                    if 'get_scaler_filename' in globals() and callable(get_scaler_filename):
                        scaler_path = get_scaler_filename(**self.model_params)
                    else: # Fallback or error if get_scaler_filename isn't usable
                        print("  Warning: get_scaler_filename not available. Cannot determine scaler path.")
                        raise FileNotFoundError("Scaler path determination failed.")

                print(f"  Attempting to load scaler from: {scaler_path}")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.model_params['scaler_object'] = self.scaler # Make it available to get_pose_data_processed
                print(f"  Scaler loaded successfully from {scaler_path}")
            except FileNotFoundError:
                print(f"  Warning: Scaler file not found at '{scaler_path}'. Proceeding without feature scaling.")
                self.model_params['scale_data'] = False
            except Exception as e:
                print(f"  Warning: Could not load or use scaler: {e}. Proceeding without feature scaling.")
                self.model_params['scale_data'] = False
        
        self.loaded_raw_cache = {}
        if self.in_memory_data:
            print(f"  Loading all raw skeleton data into memory...")
            num_loaded = 0
            for sample_info in self.samples:
                file_path = sample_info['file_path'] # Changed from 'anchor_path'
                if file_path not in self.loaded_raw_cache:
                    raw_data = load_skeleton_data(file_path)
                    if raw_data is not None:
                        self.loaded_raw_cache[file_path] = raw_data
                        num_loaded +=1
            print(f"  Loaded raw data for {num_loaded} unique files into memory.")


    def _read_annotations(self):
        """Reads annotation file, expects 'file_path label' per line."""
        samples_list = []
        print(f"  Reading annotations from: {self.pose_annotations_file}")
        try:
            with open(self.pose_annotations_file, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        file_path = parts[0]
                        try:
                            label = int(parts[1])
                            samples_list.append({
                                'id': i,
                                'file_path': file_path, # Renamed from 'anchor_path'
                                'class_id': label - 1 # Adjusted to 0-indexed       
                            })
                        except ValueError:
                            print(f"    Warning: Invalid label format on line {i+1}: {line.strip()}")
                    else:
                        print(f"    Warning: Skipping line {i+1} due to incorrect format: {line.strip()}")
        except FileNotFoundError:
            print(f"  CRITICAL Error: Annotation file not found at {self.pose_annotations_file}")
            return []
        print(f"  Found {len(samples_list)} samples in annotations.")
        return samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches and processes one single data sample and its class label.
        Triplet mining will happen in the training loop based on batches of these.
        """
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        class_id = sample_info['class_id']

        raw_data_sample = None
        if self.in_memory_data:
            raw_data_sample = self.loaded_raw_cache.get(file_path)
        else:
            raw_data_sample = load_skeleton_data(file_path)
        if raw_data_sample is None:
            print(f"Warning: Failed to load raw data for sample at index {idx}, path {file_path}. Returning dummy data.")
            _seq_len = abs(self.model_params.get('max_seq_len', 32))
            if _seq_len == 0: _seq_len = 32
            _num_feats = self.model_params.get('num_feats', 423)
            dummy_features = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long) # Default label
            return dummy_features, dummy_label
        body_selected = get_body_skel(raw_data_sample, self.is_validation, mode=self.model_params.get('get_body_skel_mode', 'var'))
        if body_selected is None or body_selected.shape[0] == 0:
            print(f"Warning: Failed to select a valid body skeleton for sample {file_path}. Returning dummy data.")
            _seq_len = abs(self.model_params.get('max_seq_len', 32))
            if _seq_len == 0: _seq_len = 32
            _num_feats = self.model_params.get('num_feats', 423)
            dummy_features = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
            dummy_label = torch.tensor(class_id, dtype=torch.long) # Still return correct label if possible
            return dummy_features, dummy_label


        # Process the single selected body
        # The scaler object is now passed via model_params if loaded in __init__
        processed_features_np = get_pose_data_processed(body_selected, self.is_validation, self.model_params)

        if processed_features_np is None:
            print(f"Warning: Failed to process skeleton for sample {file_path}. Returning dummy data.")
            _seq_len = abs(self.model_params.get('max_seq_len', 32))
            if _seq_len == 0: _seq_len = 32
            _num_feats = self.model_params.get('num_feats', 423)
            dummy_features = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
            dummy_label = torch.tensor(class_id, dtype=torch.long)
            return dummy_features, dummy_label


        # Convert to PyTorch Tensors
        features_tensor = torch.from_numpy(processed_features_np.astype(np.float32))
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return features_tensor, label_tensor

class TherapyDataset(Dataset):
    def __init__(self, data_df, video_skels, in_memory=False, validation=False,
                 max_seq_len=100, joints_num=25, joints_dim=3, num_jcd_feats=300,
                 scale_data=None,
                 center_skels=True, h_flip=False, scale_by_torso=False, temporal_scale=False,
                 use_jcd_features=False, use_speeds=False,
                 use_coords_raw=False, use_coords=False, use_jcd_diff=False,
                 use_bone_angles=False, use_bone_angles_cent=False,
                 classification=True, num_classes=None,
                 **kwargs):
        self.data_df = data_df
        self.video_skels = video_skels
        self.in_memory = in_memory
        self.validation = validation

        # Store all params needed for get_pose_data_v2
        self.max_seq_len = max_seq_len
        self.joints_num = joints_num
        self.joints_dim = joints_dim
        self.num_jcd_feats = num_jcd_feats
        self.scale_data = scale_data
        self.center_skels = center_skels
        self.h_flip = h_flip
        self.scale_by_torso = scale_by_torso
        self.temporal_scale = temporal_scale
        self.use_jcd_features = use_jcd_features
        self.use_speeds = use_speeds
        self.use_coords_raw = use_coords_raw
        self.use_coords = use_coords
        self.use_jcd_diff = use_jcd_diff
        self.use_bone_angles = use_bone_angles
        self.use_bone_angles_cent = use_bone_angles_cent
        self.classification = classification
        self.num_classes = num_classes if num_classes else len(self.data_df['action'].unique())

        # Map actions to indexes for classification labels
        self.actions = sorted(self.data_df['action'].unique())
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        if self.in_memory:
            print("Loading all samples into memory...")
            self.preloaded_data = [self._load_sample(i) for i in range(len(self.data_df))]
        else:
            self.preloaded_data = None

    def __len__(self):
        return len(self.data_df)

    def _load_sample(self, idx):
        row = self.data_df.iloc[idx]

        filename = row['preds_filename']
        action = row['action']
        start = row['preds_init']
        end = row['preds_end']

        if filename not in self.video_skels:
            raise RuntimeError(f"File {filename} not found in video_skels")

        timestamps, frame_indices, skeletons = self.video_skels[filename]

        skel_clip = skeletons[start:end + 1]

        # Call your existing pose feature extractor
        sample_np = get_pose_data_v2(
            skel_clip,
            self.max_seq_len,
            self.joints_num,
            self.joints_dim,
            self.center_skels,
            self.h_flip,
            self.scale_by_torso,
            self.temporal_scale,
            None,  # scaler (pass None or implement if you have one)
            self.validation,
            self.use_jcd_features,
            self.use_speeds,
            self.use_coords_raw,
            self.use_coords,
            self.use_jcd_diff,
            self.use_bone_angles,
            self.use_bone_angles_cent,
            skip_frames=[],
        )

        # Convert numpy array to torch tensor
        sample_tensor = torch.tensor(sample_np, dtype=torch.float32)

        if self.classification:
            label_idx = self.action_to_idx[action]
            return sample_tensor, torch.tensor(label_idx, dtype=torch.long)
        else:
            return sample_tensor

    def __getitem__(self, idx):
        if self.in_memory:
            return self.preloaded_data[idx]
        else:
            return self._load_sample(idx)

if __name__ == "__main__":
    print("Testing TripletPoseDataset...")
    # Create a dummy annotation file for testing
    dummy_ann_file = "./ntu_annotations/dummy_annotations.txt"
    with open(dummy_ann_file, "w") as f:
        # Create a few dummy .npy files that load_skeleton_data expects
        # For example, save a dict {'skel_body0': np.random.rand(50, 25, 3)}
        # to "dummy_skel_0.npy" and "dummy_skel_1.npy"
        np.save("dummy_skel_0.npy", {'skel_body0': np.random.rand(np.random.randint(20,60), 25, 3).astype(np.float32)})
        np.save("dummy_skel_1.npy", {'skel_body0': np.random.rand(np.random.randint(20,60), 25, 3).astype(np.float32)})
        f.write("dummy_skel_0.npy 0\n")
        f.write("dummy_skel_1.npy 1\n")
        f.write("dummy_skel_0.npy 0\n") # Another sample of class 0

    test_model_params = {
        # --- Fill with relevant parameters from your train.py model_params ---
        "max_seq_len": -32, "joints_num": 25, "joints_dim": 3,
        "center_skels": True, "h_flip": True, "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2], "skip_frames": [2],
        "use_jcd_features": True, "use_coords": True, "use_bone_angles": True,
        # ... other feature flags ...
        "num_feats": get_num_feats( # Use your actual get_num_feats
            joints_num=25, 
            joints_dim=3, 
            use_jcd_features=True, 
            use_speeds=False,
            use_coords_raw=False,
            use_coords=True,
            use_jcd_diff=False,
            use_bone_angles=True,
            use_bone_angles_cent=False
        ),
        "scale_data": False, # Set to True if you have a test scaler
        # "scaler_path_override": "path/to/your/test_scaler.pckl" # If testing scaler
    }
    
    print(f"Test model parameters: {test_model_params}")
    
    try:
        dataset = TripletPoseDataset(
            pose_annotations_file=dummy_ann_file,
            validation_mode=False,
            in_memory=False,
            **test_model_params
        )
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            features, label = dataset[0]
            # print(f"Sample 0 features shape: {features.shape}, dtype: {features.dtype}")
            # print(f"Sample 0 label: {label}, dtype: {label.dtype}")
            assert features.shape[1] == test_model_params["num_feats"], "Feature dimension mismatch!"
            assert features.shape[0] == abs(test_model_params["max_seq_len"]), "Sequence length mismatch!"

            features_val, label_val = dataset[1] # Test another sample
            # print(f"Sample 1 features shape: {features_val.shape}")
            # print(f"Sample 1 label: {label_val}")

        # Test with DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch_features, batch_labels in loader:
            # print(f"Batch features shape: {batch_features.shape}")
            # print(f"Batch labels: {batch_labels}")
            break # Just test one batch
    except Exception as e_test:
        print(f"Error during Dataset test: {e_test}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_ann_file): os.remove(dummy_ann_file)
        if os.path.exists("dummy_skel_0.npy"): os.remove("dummy_skel_0.npy")
        if os.path.exists("dummy_skel_1.npy"): os.remove("dummy_skel_1.npy")