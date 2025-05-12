# pytorch_dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from scipy.special import comb
from scipy.spatial.distance import cdist
import scipy.ndimage.interpolation as inter
# Note: Keras pad_sequences is replaced later, often manually or via DataLoader collate_fn
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# Note: Keras to_categorical is replaced later, usually labels are integers for CrossEntropyLoss
# from tensorflow.keras.utils import to_categorical

# =============================================================================
# Helper Functions (Ported or Adapted from Keras data_generator.py)
# =============================================================================
# IMPORTANT: Review these functions carefully.
# - Ensure they operate primarily on NumPy arrays.
# - Remove any TensorFlow/Keras specific dependencies (like tf.one_hot, etc.)
# - Ensure file paths and parameter names match how they'll be used in the Dataset.

# --- Constants (from Keras code) ---
FLIP_CORRESPONDENCES_LEFT = [4, 5, 6, 7, 12, 13, 14, 15, 21, 22]
FLIP_CORRESPONDENCES_RIGHT = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
SPINE = [0, 1, 2, 3, 20]
CONNECTING_JOINT = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9,
                    10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]

# --- Helper Function Definitions ---

def load_skeleton_data(file_path):
    """Loads skeleton data from a .npy file."""
    try:
        # Assuming the .npy file contains a dictionary as saved by the Keras version
        pose_raw = np.load(file_path, allow_pickle=True).item()
        return pose_raw
    except Exception as e:
        print(f"Error loading skeleton data from {file_path}: {e}")
        return None

def get_body_skel(pose_raw, validation, mode='var'):
    """Selects the primary skeleton from potentially multiple bodies."""
    # (Copy the implementation from your Keras data_generator.py)
    # This function selects which body's skeleton to use if multiple are detected.
    # Ensure it returns a NumPy array: [num_frames, joints_num, joints_dim]
    n_bodys = list(set(pose_raw.get('nbodys', [0]))) # Use .get for safety
    if not n_bodys or max(n_bodys) == 0 or 'skel_body0' not in pose_raw:
         # Handle cases with no bodies or only body 0 incorrectly labelled
         if 'skel_body0' in pose_raw:
              return pose_raw['skel_body0']
         else: # Cannot find any skeleton data
              print(f"Warning: No valid skeleton data found in file associated with {pose_raw.get('filename', 'unknown')}")
              # Return a dummy array or raise an error
              # Returning dummy might hide issues, consider raising error
              # For now, return None to be handled later
              return None

    # Determine which bodies are present and valid
    valid_body_indices = [i for i in range(max(n_bodys) + 1) if f'skel_body{i}' in pose_raw]
    if not valid_body_indices:
         print(f"Warning: No 'skel_bodyX' keys found despite nbodys info in {pose_raw.get('filename', 'unknown')}")
         return None

    # Calculate lengths of valid skeletons
    body_lens = []
    valid_skeletons = []
    indices_map = [] # Map index in body_lens back to original body index

    for i in valid_body_indices:
        skel = pose_raw[f'skel_body{i}']
        # Calculate length based on non-zero frames
        non_zero_frames = skel[np.all(~np.all(skel == 0, axis=2), axis=1)]
        if non_zero_frames.shape[0] > 0: # Only consider skeletons with actual movement
             body_lens.append(non_zero_frames.shape[0])
             valid_skeletons.append(skel) # Store the full skeleton
             indices_map.append(i)

    if not body_lens: # No skeletons with non-zero frames found
         print(f"Warning: All detected skeletons have zero length after filtering in {pose_raw.get('filename', 'unknown')}")
         return None # Or return the first valid one found earlier?

    max_len = max(body_lens)
    longest_indices_in_valid_list = [idx for idx, length in enumerate(body_lens) if length == max_len]

    if validation:
        if mode == 'var':
            # Calculate variance (or std dev) only on the longest skeletons
            stds = [valid_skeletons[idx].std() for idx in longest_indices_in_valid_list]
            chosen_valid_list_idx = longest_indices_in_valid_list[np.argmax(stds)]
        else: # Default to 'normal' or just take the first longest
            chosen_valid_list_idx = longest_indices_in_valid_list[0]
    else: # Training: random choice among the longest
        chosen_valid_list_idx = np.random.choice(longest_indices_in_valid_list)

    # Return the chosen skeleton
    return valid_skeletons[chosen_valid_list_idx]


def average_wrong_frame_skels(skels):
    """Interpolates frames where all joint coordinates are zero."""
    # (Copy the implementation from your Keras data_generator.py)
    if skels is None or len(skels) == 0: return skels # Handle None or empty input

    good_frames_mask = np.any(np.any(skels != 0, axis=2), axis=1)
    bad_indices = np.where(~good_frames_mask)[0]

    for idx in bad_indices:
        prev_good = -1
        for i in range(idx - 1, -2, -1): # Search backwards for a good frame
            if i == -1 or good_frames_mask[i]:
                prev_good = i
                break

        next_good = -1
        for i in range(idx + 1, len(skels) + 1): # Search forwards for a good frame
            if i == len(skels) or good_frames_mask[i]:
                next_good = i
                break

        if prev_good != -1 and next_good != len(skels): # Interpolate
            skels[idx] = (skels[prev_good] + skels[next_good]) / 2
        elif prev_good != -1: # Extrapolate from previous
            skels[idx] = skels[prev_good]
        elif next_good != len(skels): # Extrapolate from next
            skels[idx] = skels[next_good]
        else: # All frames might be bad, leave as is or handle differently
             print(f"Warning: Could not interpolate frame {idx}, possibly all frames are bad.")
             # skels[idx] remains zero or you could assign a default pose

    return skels


def zoom_to_target_len(p, target_len, joints_num, joints_dim):
    """Zooms or crops a sequence to a target length."""
    # Based on Keras zoom_to_max_len, but generalized
    num_frames = p.shape[0]
    if num_frames == target_len:
        return p
    elif num_frames == 0:
         # Handle empty input sequence
         print(f"Warning: zoom_to_target_len received empty sequence. Returning zeros of target length.")
         return np.zeros([target_len, joints_num, joints_dim], dtype=p.dtype)

    # Use interpolation (zoom)
    zoom_factor = target_len / num_frames
    p_new = np.zeros([target_len, joints_num, joints_dim], dtype=p.dtype)
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:, m, n] = inter.zoom(p[:, m, n], zoom_factor, mode='nearest', order=1)[:target_len] # order=1 for linear interp.
    return p_new


def flip_skeleton(skel, flip_axis=0):
    """Flips skeleton horizontally."""
    # (Copy the implementation from your Keras data_generator.py)
    skel_flipped = skel.copy() # Work on a copy
    # Swap left and right joints
    aux = skel_flipped[..., FLIP_CORRESPONDENCES_LEFT, :]
    skel_flipped[..., FLIP_CORRESPONDENCES_LEFT, :] = skel_flipped[..., FLIP_CORRESPONDENCES_RIGHT, :]
    skel_flipped[..., FLIP_CORRESPONDENCES_RIGHT, :] = aux
    # Flip the specified axis coordinate for relevant joints
    relevant_joints = FLIP_CORRESPONDENCES_LEFT + FLIP_CORRESPONDENCES_RIGHT + SPINE
    skel_flipped[..., relevant_joints, flip_axis] = -skel_flipped[..., relevant_joints, flip_axis]
    return skel_flipped


def scale_skel_by_torso(skel):
    """Scales skeleton based on torso length."""
    # (Copy the implementation from your Keras data_generator.py)
    if skel.shape[0] == 0: return skel # Handle empty sequence
    # Ensure indices 20, 1, 0 are valid for joints_num
    if skel.shape[1] <= 20:
         print(f"Warning: scale_by_torso requires at least 21 joints, found {skel.shape[1]}. Skipping scaling.")
         return skel

    # Calculate torso distance frame by frame
    torso_dists = np.linalg.norm(skel[:, 20] - skel[:, 1], axis=1) + \
                  np.linalg.norm(skel[:, 1] - skel[:, 0], axis=1)

    # Avoid division by zero and apply scaling
    # Use a small epsilon or check for zero. Apply scaling factor 1 if torso_dist is near zero.
    epsilon = 1e-6
    scale_factors = np.where(torso_dists > epsilon, 0.4 / torso_dists, 1.0)

    # Apply scaling factor to each frame
    skel_scaled = skel * scale_factors[:, np.newaxis, np.newaxis]
    return skel_scaled


def get_transformation_matrix_global(skel):
    """Calculates global transformation matrix for centering."""
    # (Copy the implementation from your Keras data_generator.py)
    # Ensure indices 16, 12, 20 are valid
    if skel.shape[1] <= 20:
         print(f"Warning: get_transformation_matrix_global requires at least 21 joints, found {skel.shape[1]}. Returning identity matrices.")
         return np.array([np.eye(4)] * skel.shape[0]) # Return identity matrices

    o = (skel[:, 16, :] + skel[:, 12, :]) / 2 # Origin (mid-hip)
    # Calculate x-axis (vector from origin to right hip, index 12)
    x_vec = skel[:, 12, :] - o
    # Calculate z-axis (vector from origin to spine base, index 20) - Check if this is correct axis
    z_vec = skel[:, 20, :] - o # Original code uses spine base

    # Normalize vectors, handle potential zero vectors
    x = matrix_unit_vector(x_vec)
    z = matrix_unit_vector(z_vec)

    # Ensure orthogonality using cross product for y-axis
    y = np.cross(z, x) # Changed order for right-hand rule if z is forward, x is right
    y = matrix_unit_vector(y)

    # Recalculate z or x to ensure orthogonality if needed (Gram-Schmidt like)
    z = np.cross(x, y) # z is now guaranteed orthogonal to x and y
    z = matrix_unit_vector(z)

    # Construct transformation matrices
    r_matrices = []
    for i in range(len(skel)):
        # Create rotation part
        rotation = np.eye(4)
        rotation[0, :3] = x[i]
        rotation[1, :3] = y[i]
        rotation[2, :3] = z[i]
        # Create translation part
        translation = np.eye(4)
        translation[:3, 3] = -o[i] # Translate origin to (0,0,0)

        # Transformation matrix (apply translation then rotation's inverse/transpose)
        # To transform points TO the new coordinate system: R^T * (P - o)
        # We want the inverse transform matrix to apply via matmul
        # Inverse of rotation is transpose: R^T
        # Inverse of translation is -T applied first
        # Combined inverse: R^T * T(-o)
        inv_transform = np.dot(rotation.T, translation) # Check matrix multiplication order
        r_matrices.append(inv_transform)

    return np.stack(r_matrices)

def matrix_unit_vector(matrix):
    """ Normalizes rows of a matrix to unit vectors, handles zero vectors. """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero, result will be zero vector
    norms[norms == 0] = 1.0
    return matrix / norms

def transform_skel_global(skel, r):
    """Applies global transformation matrix to skeleton."""
    # (Copy the implementation from your Keras data_generator.py)
    # Add homogeneous coordinate
    skel_h = np.concatenate([skel, np.ones((*skel.shape[:-1], 1))], axis=-1)
    # Apply transformation matrix for each frame
    # Matrix r is expected to be the inverse transform: P' = P * r^T (if P is row vector)
    # Or P' = r * P (if P is column vector)
    # PyTorch/NumPy matmul: assumes last 2 dims are matrices. skel_h (N, F, J, 4), r (N, 4, 4)
    # We need to apply r[f] to skel_h[f] for each frame f.
    # Result = np.einsum('nfjc,ncd->nfjd', skel_h, r) # Maybe too complex
    transformed_skel_h = np.zeros_like(skel_h)
    for i in range(skel.shape[0]): # Iterate through frames
         # skel_h[i] is (J, 4). r[i] is (4, 4). Result should be (J, 4)
         transformed_skel_h[i] = np.matmul(skel_h[i], r[i].T) # Apply transpose of inv_transform

    # Return only the first 3 coordinates
    return transformed_skel_h[..., :3]


def get_jcd_features(p, joints_num):
    """Calculates Joint-to-Joint distances for each frame."""
    # (Adapted from Keras data_generator.py)
    num_frames = p.shape[0]
    jcd = []
    iu = np.triu_indices(joints_num, k=1) # k=1 to exclude diagonal
    if num_frames == 0:
        num_jcd_feats = int(comb(joints_num, 2)) if joints_num >= 2 else 0
        return np.zeros((0, num_jcd_feats), dtype=np.float32) # Return empty array with correct feature dim

    for f in range(num_frames):
        d_m = cdist(p[f], p[f], 'euclidean')
        jcd.append(d_m[iu])
    return np.stack(jcd) if jcd else np.zeros((0, int(comb(joints_num, 2))), dtype=np.float32)


def get_bone_spherical_angles(v):
    """Calculates spherical angles (elevation, azimuth) for bone vectors."""
    # (Copy from Keras data_generator.py)
    # Handle potential zero vectors to avoid NaN in arctan2
    norm_xy = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
    elevation = np.arctan2(v[:, 2], norm_xy)
    azimuth = np.arctan2(v[:, 1], v[:, 0])
    return np.column_stack([elevation, azimuth])

def get_body_spherical_angles(body):
    """Calculates spherical angles for all defined bones."""
    # (Copy from Keras data_generator.py, ensure CONNECTING_JOINT is defined)
    angles_list = []
    num_frames = body.shape[0]
    if num_frames == 0:
         num_angle_feats = (len(CONNECTING_JOINT)) * 2 # Approx number of features
         return np.zeros((0, num_angle_feats), dtype=np.float32)

    # Ensure connecting joints are valid indices
    max_joint_idx = body.shape[1] - 1
    valid_bones = []
    for i in range(len(CONNECTING_JOINT)):
         j1_idx = CONNECTING_JOINT[i]
         # Need the *next* joint to form the bone vector. The original code seems to imply
         # iterating through connecting_joint pairs, but the list isn't pairs.
         # Let's assume it connects joint i to CONNECTING_JOINT[i]. This needs verification.
         # A common definition: Bone i connects joint `connecting_joint[i]` to joint `i`.
         # Let's redefine based on NTU common bones if possible, or stick to original intent if clear.
         # Assuming original intent was bone = joint[i+1] - joint[i] for i in range(num_joints-1)? No, uses CONNECTING_JOINT.
         # Let's assume bone i = joint[i] - joint[connecting_joint[i]]
         # This needs clarification based on the original paper/intent.
         # Using a simple sequential bone definition for placeholder: joint[i+1] - joint[i]
         if i + 1 <= max_joint_idx:
              bone_vec = body[:, i+1] - body[:, i]
              angles_list.append(get_bone_spherical_angles(bone_vec))

    if not angles_list: # No valid bones found
         return np.zeros((num_frames, 0), dtype=np.float32)

    return np.concatenate(angles_list, axis=1)


def get_pose_data_processed(body_raw, is_validation, model_params):
    """
    Processes a single raw skeleton sequence.
    Encapsulates augmentation, normalization, feature extraction, padding.
    Returns a NumPy array [L, C_feats].
    """
    if body_raw is None: # Handle case where get_body_skel returned None
         print("Warning: get_pose_data_processed received None for body_raw. Returning None.")
         return None

    # --- Parameters ---
    max_seq_len = model_params.get('max_seq_len', -32)
    joints_num = model_params.get('joints_num', 25)
    joints_dim = model_params.get('joints_dim', 3)
    center_skels = model_params.get('center_skels', True)
    h_flip_enabled = model_params.get('h_flip', False)
    scale_by_torso = model_params.get('scale_by_torso', True)
    temporal_scale_range = model_params.get('temporal_scale', False) # e.g., [0.8, 1.2] or False
    skip_frames_options = model_params.get('skip_frames', []) # e.g., [2, 3]
    average_wrong = model_params.get('average_wrong_skels', True)

    use_jcd = model_params.get('use_jcd_features', False)
    use_speeds = model_params.get('use_speeds', False)
    use_coords_raw = model_params.get('use_coords_raw', False)
    use_coords = model_params.get('use_coords', True)
    use_jcd_diff = model_params.get('use_jcd_diff', False)
    use_bone_ang = model_params.get('use_bone_angles', False)
    use_bone_ang_cent = model_params.get('use_bone_angles_cent', False)
    # scaler = model_params.get('scaler_object', None) # Scaler object needs to be loaded and passed

    body = body_raw.copy() # Work on a copy

    # --- Preprocessing ---
    # Remove frames with all zeros (already done in Keras version?) - Optional redundancy
    body = body[np.any(np.any(body != 0, axis=2), axis=1)]
    if body.shape[0] == 0:
        print("Warning: Skeleton has zero length after initial filtering.")
        # Return None or handle appropriately
        return None

    if average_wrong:
        body = average_wrong_frame_skels(body)
        if body is None or body.shape[0] == 0:
             print("Warning: Skeleton has zero length after averaging wrong frames.")
             return None


    # --- Augmentations (Applied only during training) ---
    if not is_validation:
        # Temporal Scaling
        if temporal_scale_range and isinstance(temporal_scale_range, (list, tuple)) and len(temporal_scale_range) == 2:
            orig_len = body.shape[0]
            min_scale, max_scale = temporal_scale_range
            if min_scale < max_scale and orig_len > 0:
                 # Calculate target length based on random scale factor
                 scale_factor = np.random.uniform(min_scale, max_scale)
                 new_len = max(2, int(orig_len * scale_factor)) # Ensure at least 2 frames
                 body = zoom_to_target_len(body, new_len, joints_num, joints_dim)

        # Skip Frames (Subsampling)
        if skip_frames_options:
            skip_rate = np.random.choice(skip_frames_options) # e.g., chooses 2 or 3
            if skip_rate > 1 and body.shape[0] > skip_rate: # Ensure skipping is meaningful
                 start_frame = np.random.randint(skip_rate)
                 body = body[start_frame::skip_rate]

        # Horizontal Flip
        if h_flip_enabled and np.random.rand() > 0.5:
            body = flip_skeleton(body)

    # --- Sequence Length Handling (Padding/Cropping/Zooming) ---
    target_seq_len = abs(max_seq_len)
    if target_seq_len == 0: # Use actual length if max_seq_len is 0
        final_len = body.shape[0]
    else:
        final_len = target_seq_len

    current_len = body.shape[0]

    if max_seq_len > 0: # Zoom or Pad to fixed length
        if current_len != final_len:
            body = zoom_to_target_len(body, final_len, joints_num, joints_dim)
    elif max_seq_len < 0: # Crop if longer, Pad if shorter
        if current_len > final_len:
            # Random crop for training, center crop for validation
            if not is_validation:
                start = np.random.randint(current_len - final_len + 1)
            else:
                start = (current_len - final_len) // 2
            body = body[start : start + final_len]
        elif current_len < final_len:
            # Pad (pre-padding is common for TCNs)
            pad_width = final_len - current_len
            # Create padding array: (pad_before, pad_after) for each axis
            # We pad only the time axis (axis 0) at the beginning
            padding = [(pad_width, 0), (0, 0), (0, 0)]
            body = np.pad(body, padding, mode='constant', constant_values=0)

    # --- Normalization ---
    if scale_by_torso:
        body = scale_skel_by_torso(body)

    # Store body before centering if needed for raw coords feature
    body_uncentered = body.copy() if use_coords_raw else None

    # Centering (Global Transformation)
    skels_centered = body # Default if not centering
    if center_skels:
        transf_matrix = get_transformation_matrix_global(body)
        skels_centered = transform_skel_global(body, transf_matrix)

    # --- Feature Extraction ---
    num_frames_final = skels_centered.shape[0]
    pose_features_list = []

    if use_bone_ang:
        pose_features_list.append(get_body_spherical_angles(body)) # Use original body for non-centered angles
    if use_bone_ang_cent:
        pose_features_list.append(get_body_spherical_angles(skels_centered))
    if use_coords_raw:
        if body_uncentered is None: body_uncentered = body # Fallback if not stored
        pose_features_list.append(body_uncentered.reshape(num_frames_final, -1))
    if use_coords:
        pose_features_list.append(skels_centered.reshape(num_frames_final, -1))

    jcd_feats = None
    if use_jcd or use_jcd_diff:
        jcd_feats = get_jcd_features(skels_centered, joints_num)
        if use_jcd:
            pose_features_list.append(jcd_feats)

    if use_jcd_diff:
        if jcd_feats is not None and jcd_feats.shape[0] > 1:
            jcd_diff_val = jcd_feats[1:] - jcd_feats[:-1]
            # Prepend first difference or zero to maintain length
            jcd_diff_val = np.concatenate([jcd_diff_val[0:1], jcd_diff_val], axis=0)
        else: # Handle sequences too short for diff
            num_jcd_feats = int(comb(joints_num, 2)) if joints_num >= 2 else 0
            jcd_diff_val = np.zeros((num_frames_final, num_jcd_feats), dtype=skels_centered.dtype)
        pose_features_list.append(jcd_diff_val)

    if use_speeds:
        if num_frames_final > 1:
             # Calculate speed based on centered skeleton
             speed_feats_val = skels_centered[1:] - skels_centered[:-1]
             # Prepend first speed or zero to maintain length
             speed_feats_val = np.concatenate([speed_feats_val[0:1], speed_feats_val], axis=0)
             speed_feats_val = speed_feats_val.reshape(num_frames_final, -1)
        else: # Handle single-frame sequences
             speed_feats_val = np.zeros((num_frames_final, joints_num * joints_dim), dtype=skels_centered.dtype)
        pose_features_list.append(speed_feats_val)

    if not pose_features_list:
        print("Warning: No features selected! Returning empty array.")
        return np.zeros((final_len, 0), dtype=np.float32)

    # Concatenate all features
    pose_features_final = np.concatenate(pose_features_list, axis=1).astype(np.float32)

    # --- Scaling (Optional) ---
    # The scaler object needs to be loaded once in __init__ and stored in self.scaler
    # if scaler is not None:
    #     pose_features_final = scaler.transform(pose_features_final)

    # Final check for sequence length (should match final_len due to padding/cropping/zooming)
    if pose_features_final.shape[0] != final_len:
         print(f"Warning: Final features length ({pose_features_final.shape[0]}) != target length ({final_len}). Resizing.")
         # This might indicate an issue in padding/cropping logic
         # Force resize as a fallback
         pose_features_final = zoom_to_target_len(pose_features_final.reshape(pose_features_final.shape[0], -1, 1), # Reshape to fit zoom
                                                 final_len, pose_features_final.shape[1], 1).reshape(final_len, -1)


    return pose_features_final


# =============================================================================
# PyTorch Dataset Class
# =============================================================================

class TripletPoseDataset(Dataset):
    def __init__(self, pose_annotations_file, validation_mode, in_memory, **model_params_kwargs):
        """
        Args:
            pose_annotations_file (string): Path to the annotation file (e.g., "filename label").
            validation_mode (bool): True if this dataset is for validation (no shuffle, no augmentations).
            in_memory (bool): Whether to load all data into RAM during initialization.
            **model_params_kwargs: Dictionary containing all other parameters needed for processing
                                   (e.g., max_seq_len, joints_num, feature flags, augmentation params).
        """
        self.pose_annotations_file = pose_annotations_file
        self.is_validation = validation_mode
        self.in_memory_data = in_memory
        self.model_params = model_params_kwargs

        print(f"\nInitializing TripletPoseDataset:")
        print(f"  Annotations: {self.pose_annotations_file}")
        print(f"  Validation Mode: {self.is_validation}")
        print(f"  In Memory: {self.in_memory_data}")

        # --- Load Annotations ---
        self.samples = self._read_annotations()
        if not self.samples:
            raise ValueError(f"No samples found or loaded from annotation file: {pose_annotations_file}")

        # --- Load Scaler (if applicable) ---
        self.scaler = None
        if self.model_params.get('scale_data', False):
            try:
                # Ensure get_scaler_filename and loading logic are TF-free
                # scaler_filename = get_scaler_filename(**self.model_params) # Ensure this helper is defined/imported
                # print(f"  Loading scaler: {scaler_filename}")
                # with open(scaler_filename, 'rb') as f:
                #     self.scaler = pickle.load(f)
                # self.model_params['scaler_object'] = self.scaler # Pass to processing func
                 print("  Note: Scaler loading logic needs verification/implementation.")
            except Exception as e:
                print(f"  Warning: Could not load scaler: {e}. Proceeding without scaling.")
                self.model_params['scale_data'] = False # Disable scaling if load fails

        # --- Load Data to Memory (if requested) ---
        self.loaded_samples_cache = {} # Use dict for faster lookup if needed, or list
        if self.in_memory_data:
            print(f"  Loading all raw data into memory...")
            # Store raw loaded data to apply augmentations on the fly in __getitem__
            num_loaded = 0
            for sample_info in self.samples:
                file_path = sample_info['anchor_path']
                if file_path not in self.loaded_samples_cache:
                    raw_data = load_skeleton_data(file_path)
                    if raw_data is not None:
                         self.loaded_samples_cache[file_path] = raw_data
                         num_loaded += 1
                    else:
                         print(f"    Warning: Failed to load {file_path} for in-memory cache.")
            print(f"  Loaded raw data for {num_loaded} unique files into memory.")


    def _read_annotations(self):
        """Reads the annotation file and stores sample information."""
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
                            # Basic check if file exists (optional, can slow down init)
                            # if not os.path.exists(file_path):
                            #     print(f"    Warning: File not found for sample {i}: {file_path}")
                            #     continue
                            samples_list.append({
                                'id': i, # Unique ID for this sample line
                                'anchor_path': file_path,
                                'class_id': label
                            })
                        except ValueError:
                            print(f"    Warning: Invalid label format on line {i+1}: {line.strip()}")
                        except Exception as e:
                             print(f"    Warning: Error processing line {i+1} ('{line.strip()}'): {e}")

                    else:
                        print(f"    Warning: Skipping line {i+1} due to incorrect format: {line.strip()}")
        except FileNotFoundError:
            print(f"  Error: Annotation file not found at {self.pose_annotations_file}")
            return [] # Return empty list

        print(f"  Found {len(samples_list)} samples in annotations.")
        # Create a mapping from class_id to list of sample indices for faster triplet selection
        self.class_to_indices = {}
        for i, sample in enumerate(samples_list):
            class_id = sample['class_id']
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(i) # Store index in self.samples

        return samples_list

    def __len__(self):
        """Returns the total number of anchor samples."""
        return len(self.samples)

    def _find_positive(self, anchor_idx, anchor_class_id):
        """Finds a positive sample index (different from anchor, same class)."""
        possible_indices = self.class_to_indices.get(anchor_class_id, [])
        if not possible_indices or len(possible_indices) == 1:
            # No other samples of the same class, return anchor itself
            return anchor_idx
        # Exclude anchor itself and choose randomly
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = np.random.choice(possible_indices)
        return positive_idx

    def _find_negative(self, anchor_class_id):
        """Finds a negative sample index (different class)."""
        possible_classes = list(self.class_to_indices.keys())
        if not possible_classes or len(possible_classes) == 1:
             # No other classes available, something is wrong or only one class in dataset
             # Fallback: maybe return a random sample from the same class? Or raise error?
             print(f"Warning: Cannot find negative sample, only class {anchor_class_id} available.")
             # Returning a random index from the same class as a poor fallback
             return np.random.choice(self.class_to_indices.get(anchor_class_id, [0]))


        negative_class_id = anchor_class_id
        while negative_class_id == anchor_class_id:
            negative_class_id = np.random.choice(possible_classes)

        possible_indices = self.class_to_indices.get(negative_class_id, [])
        if not possible_indices:
             # Should not happen if class was chosen from keys, but safety check
             print(f"Warning: No samples found for chosen negative class {negative_class_id}. Falling back.")
             # Fallback to a random sample from the anchor class
             return np.random.choice(self.class_to_indices.get(anchor_class_id, [0]))

        return np.random.choice(possible_indices)


    def __getitem__(self, idx):
        """
        Fetches and processes one triplet (Anchor, Positive, Negative)
        and the anchor's classification label.
        """
        # 1. Get Anchor Info
        anchor_info = self.samples[idx]
        anchor_path = anchor_info['anchor_path']
        anchor_class_id = anchor_info['class_id']

        # 2. Find Positive and Negative Sample Indices
        # (Only needed if triplet loss is active, but simplifies structure to always find)
        positive_idx = self._find_positive(idx, anchor_class_id)
        negative_idx = self._find_negative(anchor_class_id)

        positive_info = self.samples[positive_idx]
        negative_info = self.samples[negative_idx]

        positive_path = positive_info['anchor_path']
        negative_path = negative_info['anchor_path']

        # 3. Load Raw Data (from cache or disk)
        raw_A, raw_P, raw_N = None, None, None
        if self.in_memory_data:
            raw_A = self.loaded_samples_cache.get(anchor_path)
            raw_P = self.loaded_samples_cache.get(positive_path)
            raw_N = self.loaded_samples_cache.get(negative_path)
        else:
            raw_A = load_skeleton_data(anchor_path)
            raw_P = load_skeleton_data(positive_path)
            raw_N = load_skeleton_data(negative_path)

        # Handle loading failures
        if raw_A is None or raw_P is None or raw_N is None:
             print(f"Warning: Failed to load raw data for triplet at index {idx}. Returning dummy data.")
             # Fallback: return dummy data or raise error
             _seq_len = abs(self.model_params.get('max_seq_len', 32))
             if _seq_len == 0: _seq_len = 32
             _num_feats = self.model_params.get('num_feats', 423) # Need num_feats here
             dummy_data = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
             dummy_label = torch.tensor(0, dtype=torch.long)
             return dummy_data, dummy_data, dummy_data, dummy_label


        # 4. Select the main body skeleton for each
        body_A = get_body_skel(raw_A, self.is_validation)
        body_P = get_body_skel(raw_P, self.is_validation)
        body_N = get_body_skel(raw_N, self.is_validation)

        # Handle cases where body selection fails
        if body_A is None or body_P is None or body_N is None:
             print(f"Warning: Failed to select body skeleton for triplet at index {idx}. Returning dummy data.")
             _seq_len = abs(self.model_params.get('max_seq_len', 32))
             if _seq_len == 0: _seq_len = 32
             _num_feats = self.model_params.get('num_feats', 423)
             dummy_data = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
             dummy_label = torch.tensor(0, dtype=torch.long)
             return dummy_data, dummy_data, dummy_data, dummy_label


        # 5. Process each skeleton (Augmentation, Normalization, Features, Padding)
        # Pass the scaler object if it was loaded
        # self.model_params['scaler_object'] = self.scaler
        processed_A = get_pose_data_processed(body_A, self.is_validation, self.model_params)
        processed_P = get_pose_data_processed(body_P, self.is_validation, self.model_params)
        processed_N = get_pose_data_processed(body_N, self.is_validation, self.model_params)

        # Handle processing failures
        if processed_A is None or processed_P is None or processed_N is None:
             print(f"Warning: Failed to process skeleton for triplet at index {idx}. Returning dummy data.")
             _seq_len = abs(self.model_params.get('max_seq_len', 32))
             if _seq_len == 0: _seq_len = 32
             _num_feats = self.model_params.get('num_feats', 423)
             dummy_data = torch.zeros(_seq_len, _num_feats, dtype=torch.float32)
             dummy_label = torch.tensor(0, dtype=torch.long)
             return dummy_data, dummy_data, dummy_data, dummy_label

        # 6. Convert to PyTorch Tensors
        anchor_tensor = torch.from_numpy(processed_A)
        positive_tensor = torch.from_numpy(processed_P)
        negative_tensor = torch.from_numpy(processed_N)
        # Classification target is the integer class ID of the anchor
        clf_target_tensor = torch.tensor(anchor_class_id, dtype=torch.long)

        return anchor_tensor, positive_tensor, negative_tensor, clf_target_tensor

# Example of how to potentially define/import helpers if not in the main Keras file
# def load_scaler(**params):
#    filename = get_scaler_filename(**params)
#    with open(filename, 'rb') as f:
#        scaler = pickle.load(f)
#    return scaler
#
# def get_scaler_filename(**params):
#    # ... implementation from Keras file ...
#    pass
