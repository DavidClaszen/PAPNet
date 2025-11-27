import argparse
import torch
import torch.nn as nn
import numpy as np
from model import Model
from dataloader import DataLoader
from tqdm import tqdm

from lib.sample_rotations import sample_rotations_60
R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()

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
    for i in range(len(Rs)):
        bin_R, _ = R_to_bin_delta(Rs[i], R_bin_ctrs, knn=knn)
        bin_Rs.append(bin_R)
    return torch.stack(bin_Rs)

def quat2mat(q):
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

def eval_papnet(model_path, dataset, data_path, rot_k, batch_size=16):
    """Evaluate a given PAPNet model on the specified dataset.
    
    Args:
        model_path (str): Path to the trained PAPNet model.
        data_path (str): Path to the dataset for evaluation.
        dataset (str): Name of the dataset to evaluate on.
        rot_k (int): Number of rotation bins to consider by the classifier.
        batch_size (int): Batch size for evaluation.
    
    Returns:
        y_pred_cls (ndarray): Predicted classification labels. (N,)
        y_true_cls (ndarray): True classification labels. (N,)
        y_pred_rot_k (ndarray): Top-k rotation classification bins. (N, k)
        y_true_rot (ndarray): True rotation classification bin. (N,)

        Estimated memory usage:
            For batch_size B, memory usage is approximately:
            Memory (in GB) = B * (4 + k) * size_of_float32(4 bytes)
    """

    if dataset == 'pm40':
        num_class = 40
    elif dataset == 'ps15':
        num_class = 15
    
    torch.backends.cudnn.benchmark = True
    TEST_DATASET = DataLoader(dataset=dataset, root=data_path, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    classifier = nn.DataParallel(Model(num_class=num_class, num_k=rot_k).cuda())
    classifier.load_state_dict(torch.load(model_path), strict=False)
    classifier = classifier.eval()
    print('# classifier parameters:', sum(param.numel() for param in classifier.parameters()))

    y_pred_cls = np.empty((0))
    y_true_cls = np.empty((0))
    y_pred_rot_k = np.empty((0,rot_k))
    y_true_rot = np.empty((0))

    with torch.no_grad():
        total_correct = np.zeros(rot_k)
        total_bin = np.zeros(rot_k)         
        total_seen = 0
        for _, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
            vol, gt_cls, gt_rot, gt_noi = data
            vol, gt_cls, gt_rot, gt_noi = \
                vol.cuda(), gt_cls.cuda().long(), gt_rot.cuda(), gt_noi.cuda()
            gt_rot_bin = Rs_to_bin_delta_batch(quat2mat(gt_rot), R_bin_ctrs)# (B, 4) -> B

            cand_log, cand_cls, pred_rot_bin = classifier(vol)# (B, k), (B, k)
            for i in range(rot_k):
                final_cls = torch.gather(cand_cls[:, 0:i+1], 1, torch.argmax(cand_log[:, 0:i+1], 1)[:, None]).view(-1) # (B, )
                total_correct[i] += torch.sum(final_cls == gt_cls).item()

                pred_rot_bin_k = torch.topk(pred_rot_bin, k=i+1, dim=1)[1]# (B, 60) -> (B, i+1)
                total_bin[i] += (pred_rot_bin_k == gt_rot_bin[:, None]).any(1).sum()
            total_seen += final_cls.shape[0]
            y_pred_cls = np.concatenate((y_pred_cls, final_cls.cpu().numpy()), axis=0)
            y_true_cls = np.concatenate((y_true_cls, gt_cls.cpu().numpy()), axis=0)
            y_pred_rot_k = np.concatenate((y_pred_rot_k, pred_rot_bin_k.cpu().numpy()), axis=0)
            y_true_rot = np.concatenate((y_true_rot, gt_rot_bin.cpu().numpy()), axis=0)

        test_ins_acc = total_correct / float(total_seen)
        test_bin_acc = total_bin / float(total_seen)
        for i in range(rot_k):
            print('k=%d, Test Ins Acc: %f, Test Bin Top-k Acc: %f' % (i+1, test_ins_acc[i], test_bin_acc[i]))

    return y_pred_cls, y_true_cls, y_pred_rot_k, y_true_rot
