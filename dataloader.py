from torch.utils.data import Dataset
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import os

def pc2vol(points, vsize=64, radius=1.0):
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol

def pm40_symmetry_mapping(cls, gt_rot):
    # PartialModelNet40
    I = np.eye(3)
    Rx_pi = quat2mat([0, 1, 0, 0])
    Ry_pi = quat2mat([0, 0, 1, 0])
    Rz_pi = quat2mat([0, 0, 0, 1])
    if cls in [5, 6, 9, 10, 15, 19, 26, 32, 37]:# Z-inf
        alpha = np.arctan2(gt_rot[1, 0] - gt_rot[0, 1], gt_rot[0, 0] + gt_rot[1, 1])
        S_map = np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
                            [np.sin(alpha), np.cos(alpha) , 0.0],
                            [0.0          , 0.0           , 1.0]])
        gt_rot = np.dot(gt_rot, S_map.T)
    if cls in [4, 11, 13, 14, 16, 18, 27, 38, 39]:# X-180
        if np.linalg.norm(gt_rot-I, axis=(0,1)) < np.linalg.norm(np.dot(gt_rot,Rx_pi.T)-I, axis=(0,1)):
            S_map = I
        else:
            S_map = Rx_pi
        gt_rot = np.dot(gt_rot, S_map.T)
    if cls in [4, 11, 13, 14, 16, 17, 18, 27, 38, 39]:# Y-180
        if np.linalg.norm(gt_rot-I, axis=(0,1)) < np.linalg.norm(np.dot(gt_rot,Ry_pi.T)-I, axis=(0,1)):
            S_map = I  
        else:
            S_map = Ry_pi
        gt_rot = np.dot(gt_rot, S_map.T)
    if cls in [1, 4, 11, 13, 14, 16, 18, 23, 27, 33, 34, 36, 38, 39]:# Z-180
        if np.linalg.norm(gt_rot-I, axis=(0,1)) < np.linalg.norm(np.dot(gt_rot,Rz_pi.T)-I, axis=(0,1)):
            S_map = I  
        else:
            S_map = Rz_pi
        gt_rot = np.dot(gt_rot, S_map.T)
    return gt_rot

def rot_add_noise(gt_rot, delta=45):
    np.random.seed()
    angles = (delta*2/180.0)*np.pi*np.random.rand(3) - (delta/180.0)*np.pi
    Rx = np.array([[1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]),np.cos(angles[2]),0],
        [0,0,1]])
    noi_rot = np.dot(Rz, np.dot(Ry, Rx))
    gt_noi = noi_rot @ gt_rot
    gt_noi = mat2quat(gt_noi)
    return gt_noi


def random_plane_occlusion(
    batch_pc: np.ndarray,
    min_occlude: float = 0.50,
    max_occlude: float = 0.90,
) -> np.ndarray:
    """
    Apply random plane-based occlusion to a batch of point clouds.

    Args:
        batch_pc: (B, N, D) array. First 3 dims are xyz, remaining dims
                  (e.g. normals) are kept aligned.
        min_occlude: minimum fraction of points to occlude.
        max_occlude: maximum fraction of points to occlude.

    Returns:
        (B, N, D) array with each cloud occluded
            and resampled back to N points.
    """
    assert batch_pc.ndim == 3, "batch_pc must be (B, N, D)"
    B, N, D = batch_pc.shape
    out = np.empty_like(batch_pc)

    rng = np.random.default_rng()

    for b in range(B):
        pc = batch_pc[b]           # (N, D)
        coords = pc[:, :3]         # xyz

        # Sample occlusion fraction
        frac_occ = rng.uniform(min_occlude, max_occlude)
        k_drop = int(round(frac_occ * N))
        k_drop = max(1, min(N - 1, k_drop))  # drop at least 1, keep at least 1

        # Random plane normal
        n = rng.normal(size=3)
        n /= (np.linalg.norm(n) + 1e-8)
        # Project points onto this direction
        proj = coords @ n  # (N,)
        # Sort along projection axis
        sorted_idx = np.argsort(proj)

        # Randomly choose which side to occlude: near or far
        if rng.integers(0, 2) == 0:
            # Occlude smallest projections
            keep_idx = sorted_idx[k_drop:]
        else:
            # Occlude largest projections
            keep_idx = sorted_idx[:N - k_drop]

        kept = pc[keep_idx]        # (N_keep, D)
        N_keep = kept.shape[0]

        # Resample remaining points back to N with replacement
        if N_keep == N:
            out[b] = kept
        else:
            extra_idx = rng.integers(0, N_keep, size=N - N_keep)
            all_idx = np.concatenate([np.arange(N_keep), extra_idx], axis=0)
            out[b] = kept[all_idx]

    return out


class DataLoader(Dataset):
    def __init__(self, dataset, root, split='train', partiality="", occlusion_setting=None):
        """Custom Dataset for loading point cloud data and corresponding labels and rotations.
        
        This class loads data from three files located in the specified root directory:
        - '<split>_points<_partiality>.npy': Contains point cloud data.
        - '<split>_labels.npy': Contains labels for the point clouds.
        - '<split>_gt_rot.npy': Contains ground truth rotations.

        Args:
            dataset (str): Name of the dataset ('pm40' or 'ps15').
            root (str): Root directory where the dataset files are stored.
            split (str): Dataset split to use ('train' or 'test').
            partiality (str): Partiality setting for the dataset.
                              Default is an empty string.
            occlusion_setting (dict, optional): Settings for applying occlusion during training.
                                               If None, no occlusion is applied.

        Returns:
            object: An instance of the DataLoader class.

        """
        self.dataset = dataset
        self.root = root
        self.split = split
        self.partiality = partiality

        if split == 'train':
            self.points = np.load(os.path.join(root, f'train_points{partiality}.npy'))
            self.labels = np.load(os.path.join(root, 'train_labels.npy'))
            self.gt_rot = np.load(os.path.join(root, 'train_gt_rot.npy'))

            # Optional occlusion setting for training data
            if occlusion_setting is not None:
                assert occlusion_setting.get('min_occlude', 0) >= 0.0 and occlusion_setting.get('max_occlude', 1.0) <= 1.0, \
                    "Occlusion fractions must be between 0 and 1."
                assert occlusion_setting['min_occlude'] <= occlusion_setting['max_occlude'], \
                    "min_occlude must be less than or equal to max_occlude."
                self.points = random_plane_occlusion(
                    self.points,
                    min_occlude=occlusion_setting.get('min_occlude', 0.50),
                    max_occlude=occlusion_setting.get('max_occlude', 0.90),
                )
        else:
            self.points = np.load(os.path.join(root, f'test_points{partiality}.npy'))
            self.labels = np.load(os.path.join(root, 'test_labels.npy'))
            self.gt_rot = np.load(os.path.join(root, 'test_gt_rot.npy'))

        print('The size of %s data is %d'%(split, len(self.points)))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):

        pts = self.points[index][:, 0:3]        
        cls = self.labels[index]
        gt_rot = self.gt_rot[index]        

        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        radius = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / radius

        if self.dataset == 'pm40':
            gt_rot = pm40_symmetry_mapping(cls, gt_rot)
        gt_noi = rot_add_noise(gt_rot)

        vol = pc2vol(pts) # N*3 -> 64*64*64
        gt_rot = mat2quat(gt_rot)

        return vol[None, :, :, :].astype(np.float32), \
               cls.astype(np.int32), \
               gt_rot.astype(np.float32), \
               gt_noi.astype(np.float32)
