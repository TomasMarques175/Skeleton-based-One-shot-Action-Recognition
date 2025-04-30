import os
import numpy as np
from scipy.ndimage import zoom as inter_zoom

def parse_skeleton_file(filepath):
    data = {
        'nbodys': [],
    }
    with open(filepath, 'r') as f:
        n_frames = int(f.readline())
        for frame_idx in range(n_frames):
            n_bodies = int(f.readline())
            data['nbodys'].append(n_bodies)
            for body_id in range(n_bodies):
                _ = [f.readline() for _ in range(5)]  # skip body metadata
                n_joints = int(f.readline())
                joints = []
                for _ in range(n_joints):
                    values = list(map(float, f.readline().split()))
                    joints.append(values[:3])  # Only use x, y, z
                joints = np.array(joints)
                if f'skel_body{body_id}' not in data:
                    data[f'skel_body{body_id}'] = []
                data[f'skel_body{body_id}'].append(joints)
    
    for k in data:
        if k.startswith('skel_body'):
            data[k] = np.array(data[k])  # shape: [frames, 25, 3]
    return data

def get_body_skel(pose_raw, validation, mode='var'):
    n_bodys = list(set(pose_raw['nbodys']))
    if len(n_bodys) == 0:
        return pose_raw['skel_body0']

    body_lens = np.array([len(pose[np.all(~np.all(pose == 0, axis=2), axis=1)]) for pose in
                         [pose_raw.get(f'skel_body{i}', np.zeros((1, 25, 3))) for i in range(max(n_bodys))]])
    body_lens = np.where(body_lens == max(body_lens))[0]

    if validation:
        if mode == 'normal':
            return pose_raw[f'skel_body{body_lens[0]}']
        elif mode == 'var':
            stds = [pose_raw[f'skel_body{i}'].std() for i in body_lens]
            return pose_raw[f'skel_body{body_lens[stds.index(max(stds))]}']
        else:
            raise ValueError('Unknown mode')
    else:
        p_ind = np.random.choice(body_lens)
        return pose_raw[f'skel_body{p_ind}']

def zoom_to_max_len(body, max_len, joints_num, joints_dim):
    if len(body) == max_len:
        return body
    zoom_factor = max_len / len(body)
    return inter_zoom(body, (zoom_factor, 1, 1), mode='nearest')

def get_pose_data_v2(body, max_seq_len, joints_num, joints_dim, center_skels,
                     h_flip, scale_by_torso, temporal_scale, scaler,
                     validation,
                     use_jcd_features, use_speeds,
                     use_coords_raw, use_coords, use_jcd_diff,
                     use_bone_angles,
                     use_bone_angles_cent,
                     skip_frames=[], **kwargs):

    body = body[np.all(~np.all(body == 0, axis=2), axis=1)]
    if max_seq_len > 0:
        body = zoom_to_max_len(body, max_seq_len, joints_num, joints_dim)

    num_frames = len(body)
    pose_features = []

    if use_coords:
        pose_features.append(np.reshape(body, (num_frames, joints_num * joints_dim)))

    pose_features = np.concatenate(pose_features, axis=1).astype('float32')

    if scaler is not None:
        pose_features = scaler.transform(pose_features)

    return pose_features

def convert_folder_skeletons_to_npy(input_folder, output_folder, model_params):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.endswith('.skeleton'):
            continue

        in_path = os.path.join(input_folder, fname)
        out_name = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_folder, out_name)

        try:
            pose_raw = parse_skeleton_file(in_path)
            body = get_body_skel(pose_raw, validation=True)

            if model_params.get('average_wrong_skels', False):
                body = average_wrong_frame_skels(body)  # define this if needed

            pose_data = get_pose_data_v2(body, validation=True, **model_params)
            np.save(out_path, pose_data)

            print(f"[✓] Saved: {out_name} - shape {pose_data.shape}")

        except Exception as e:
            print(f"[!] Failed on {fname}: {e}")

if __name__ == "__main__":
    model_params = {
        'max_seq_len': 300,
        'joints_num': 25,
        'joints_dim': 3,
        'center_skels': False,
        'h_flip': False,
        'scale_by_torso': False,
        'temporal_scale': False,
        'scaler': None,

        'use_jcd_features': False,
        'use_speeds': False,
        'use_coords_raw': False,
        'use_coords': True,
        'use_jcd_diff': False,
        'use_bone_angles': False,
        'use_bone_angles_cent': False,

        'average_wrong_skels': False
    }

    input_dir = "./datasets/NTU-120/raw_npy"
    output_dir = "./datasets/NTU-120/raw_npy"

    convert_folder_skeletons_to_npy(input_dir, output_dir, model_params)
