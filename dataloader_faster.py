import torch
from torch.utils.data import Dataset
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import os
import tqdm

from lib.sample_rotations import sample_rotations_60

def pc2vol(points, vsize=64, radius=1.0):
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol
    
def pc2vol_torch(points, vsize=64, radius=1.0):
    """ Convert point cloud to volumetric representation using PyTorch
    
    Use this function to leverage GPU acceleration for faster conversion during training.
    
    Args:
        points: (N, 3) torch tensor of point cloud
        vsize: int, size of the volumetric grid
        radius: float, radius of the sphere within which points are considered

    Returns:
        vol: (vsize, vsize, vsize) torch tensor of volumetric data
    """
    vol = torch.zeros((vsize, vsize, vsize)).to(points.device)
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.long()
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol

def Rs_to_bin_delta_batch(Rs, R_bin_ctrs, knn=False):
    def R_to_bin_delta(R=None, R_bin_ctrs=None, theta1=0.4, theta2=0.2, knn=False):
        def geodesic_dists(R_bin_ctrs, R):
            internal = 0.5 * (torch.diagonal(torch.matmul(R_bin_ctrs, torch.transpose(R, -1, -2)), 
                dim1=-1, dim2=-2).sum(-1) - 1.0)
            internal = torch.clamp(internal, -1.0, 1.0)
            return torch.acos(internal)
        dists = geodesic_dists(R_bin_ctrs, R)
        if knn:
            bin_R = torch.zeros(R_bin_ctrs.shape[0]).cuda()
            delta_R = torch.zeros(R_bin_ctrs.shape).cuda()
            _, nn4 = torch.topk(dists, k=4, largest=False)
            bin_R[nn4] = theta2
            bin_R[nn4[0]] = theta1
        else:
            bin_R = torch.argmin(dists)
            delta_R = R[..., :3, : 3].matmul(R_bin_ctrs[bin_R].t())
        return bin_R, delta_R

    bin_Rs = []
    for i in tqdm.tqdm(range(len(Rs)), desc='Binning rotations'):
        bin_R, _ = R_to_bin_delta(Rs[i], R_bin_ctrs, knn=knn)
        bin_Rs.append(bin_R)
    return torch.stack(bin_Rs)

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

def quat2mat_torch(q):
    B = q.size(0)
    R = torch.cat(((1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)).view(B, 1), \
            (2.0*q[:, 1]*q[:, 2] - 2.0*q[:, 0]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 0]*q[:, 2] + 2.0*q[:, 1]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 1]*q[:, 2] + 2.0*q[:, 3]*q[:, 0]).view(B, 1), \
            (1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)).view(B, 1), \
            (-2.0*q[:, 0]*q[:, 1] + 2.0*q[:, 2]*q[:, 3]).view(B, 1), \
            (-2.0*q[:, 0]*q[:, 2] + 2.0*q[:, 1]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 0]*q[:, 1] + 2.0*q[:, 2]*q[:, 3]).view(B, 1), \
            (1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)).view(B, 1)), dim=1).view(B, 3, 3)
    return R 

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

class DataLoader(Dataset):
    def __init__(self, dataset, root, split='train'):
        self.dataset = dataset
        self.root = root
        self.split = split

        if split == 'train':
            self.points = np.load(os.path.join(root, 'train_points.npy'))
            self.labels = np.load(os.path.join(root, 'train_labels.npy'))
            self.gt_rot = np.load(os.path.join(root, 'train_gt_rot.npy'))
        else:
            self.points = np.load(os.path.join(root, 'test_points.npy'))
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

class FastDataLoader(Dataset):
    """
    PAPNet faster dataloader. 

    Preprocesses the entire dataset during initialization to speed up training.
    With the exception of converting point clouds to volumetric representation (voxels)
    this includes normalization of point clouds and processing of rotations
    with symmetry mapping, noise addition, and binning.
    Reason for excluding voxel conversion is that voxels occupy a lot of memory
    and it is more efficient to convert them on-the-fly during training using GPU.
    """

    # Load rotation bins
    R_bin_ctrs_torch = torch.tensor(sample_rotations_60("matrix")).float().cuda()

    def __init__(self, dataset, root, split='train'):
        self.dataset = dataset
        self.root = root

        if split == 'train':
            self.points = np.load(os.path.join(root, 'train_points.npy'))
            self.labels = np.load(os.path.join(root, 'train_labels.npy'))
            self.gt_rot = np.load(os.path.join(root, 'train_gt_rot.npy'))
        else:
            self.points = np.load(os.path.join(root, 'test_points.npy'))
            self.labels = np.load(os.path.join(root, 'test_labels.npy'))
            self.gt_rot = np.load(os.path.join(root, 'test_gt_rot.npy'))
        self.split = split

        print('The size of %s data is %d. Preprocessing...'%(split, len(self.points)))

        # Batch process rotations
        self.points = self.batch_normalize_points(self.points)
        self.gt_rot_bin, self.gt_Rmat_noi = self.batch_process_rotations(self.labels, self.gt_rot)
        print('Preprocessing done.')

    def batch_normalize_points(self, batch_points: np.ndarray) -> np.ndarray:
        """ Normalize batch point clouds
        
        Args:
            batch_points: (N, P, 3) numpy array of point clouds

        Returns:
            normed_points: (N, P, 3) numpy array of normalized point clouds
        """
        N = batch_points.shape[0]
        normed_points = np.zeros(batch_points.shape, dtype=np.float32)
        for i in tqdm.tqdm(range(N), desc='Normalizing point clouds'):
            centroid = np.mean(batch_points[i], axis=0)
            pts = batch_points[i] - centroid
            radius = np.max(np.linalg.norm(pts, axis=1))
            normed_points[i] = pts / radius
        return normed_points

    # def batch_convert_to_vol(self, batch_points: np.ndarray) -> np.ndarray:
    #     """ Convert batch point cloud to volumetric representation 
        
    #     Args:
    #         batch_points: (N, P, 3) numpy array of point clouds

    #     Returns:
    #         vol: (N, 64, 64, 64) numpy array of volumetric data
    #     """
    #     N = batch_points.shape[0]
    #     vsize = 64
    #     radius = 1.0
    #     vol = np.zeros((N, vsize, vsize, vsize), dtype=np.float32)
    #     voxel = 2 * radius / float(vsize)
    #     locations = (batch_points + radius) / voxel
    #     locations = locations.astype(int)
    #     for i in range(N):
    #         vol[i, locations[i, :, 0], locations[i, :, 1], locations[i, :, 2]] = 1.0
    #     return vol
        
    def batch_process_rotations(self, batch_cls: np.ndarray, batch_gt_rot: np.ndarray) -> np.ndarray:
        """ Process batch rotations with symmetry mapping, noise addition, and binning
        
        Args:
            batch_cls: (B,) numpy array of class labels
            batch_gt_rot: (B, 3, 3) numpy array of ground truth rotations

        Returns:
            batch_gt_rot_bin: (B,) torch tensor of binned rotations
            batch_gt_Rmat_noi: (B, 3, 3) torch tensor of noisy rotations in matrix form
        """
        B = batch_gt_rot.shape[0]
                
        # Apply symmetry mapping
        for i in tqdm.tqdm(range(B), desc='Applying symmetry mapping to rotations'):
            if self.dataset == 'pm40':
                batch_gt_rot[i] = pm40_symmetry_mapping(batch_cls[i], batch_gt_rot[i])
        
        # Add noise to rotations
        batch_gt_noi = np.zeros((B, 4), dtype=np.float32)
        for i in tqdm.tqdm(range(B), desc='Adding noise to rotations'):
            batch_gt_noi[i] = rot_add_noise(batch_gt_rot[i])
        
        # Convert to tensors
        batch_gt_rot_torch = torch.from_numpy(batch_gt_rot).float()
        batch_gt_noi_torch = torch.from_numpy(batch_gt_noi).float()
        
        # Bin the ground truth rotations
        R_bin_ctrs_torch = self.R_bin_ctrs_torch
        if torch.cuda.is_available():
            batch_gt_rot_torch = batch_gt_rot_torch.cuda()
            batch_gt_noi_torch = batch_gt_noi_torch.cuda()
            R_bin_ctrs_torch = R_bin_ctrs_torch.cuda()
        
        # Convert the noise rotations to matrix form since the add noise function is in numpy
        batch_gt_noi_torch = quat2mat_torch(batch_gt_noi_torch)
        
        batch_gt_noi_bin = Rs_to_bin_delta_batch(batch_gt_noi_torch, R_bin_ctrs_torch)
        batch_gt_rot_bin = Rs_to_bin_delta_batch(batch_gt_rot_torch, R_bin_ctrs_torch, knn=True)
        
        # Convert noisy quaternions to rotation matrices
        batch_gt_Rmat_noi = torch.gather(R_bin_ctrs_torch, 0, 
            batch_gt_noi_bin[:, None, None].repeat(1, 3, 3))
        
        # Back to CPU and numpy
        batch_gt_rot_bin = batch_gt_rot_bin.cpu().numpy()
        batch_gt_Rmat_noi = batch_gt_Rmat_noi.cpu().numpy()

        return batch_gt_rot_bin, batch_gt_Rmat_noi

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):

        pts = self.points[index]
        cls = self.labels[index]
        gt_rot_bin = self.gt_rot_bin[index]
        gt_Rmat_noi = self.gt_Rmat_noi[index]
        
        return pts.astype(np.float32), \
               cls.astype(np.int32), \
               gt_rot_bin.astype(np.int64), \
               gt_Rmat_noi.astype(np.float32)