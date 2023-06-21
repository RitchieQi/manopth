import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from manopth import argutils
from manopth.manolayer import ManoLayer
from manopth.demo import display_hand
torch.set_printoptions(sci_mode=False)
offset = torch.tensor([[[0.24701, 0, 0.01]]],dtype=torch.float32) * 1000

def rearrange(tensor):
    reindex = [0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1,2,3,4]
    return tensor[:,reindex,:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument(
        '--no_display',
        action='store_true',
        help="Disable display output of ManoLayer given random inputs")
    parser.add_argument('--side', default='left', choices=['left', 'right'])
    parser.add_argument('--random_shape', action='store_true', help="Random hand shape")
    parser.add_argument('--rand_mag', type=float, default=1, help="Controls pose variability")
    parser.add_argument(
        '--flat_hand_mean',
        action='store_true',
        help="Use flat hand as mean instead of average hand pose")
    parser.add_argument(
        '--iters',
        type=int,
        default=1,
        help=
        "Use for quick profiling of forward and backward pass accross ManoLayer"
    )
    parser.add_argument('--mano_root', default='mano/models')
    parser.add_argument('--root_rot_mode', default='axisang', choices=['rot6d', 'axisang'])
    parser.add_argument('--no_pca', action='store_true', help="Give axis-angle or rotation matrix as inputs instead of PCA coefficients")
    parser.add_argument('--joint_rot_mode', default='axisang', choices=['rotmat', 'axisang'], help="Joint rotation inputs")
    parser.add_argument(
        '--mano_ncomps', default=6, type=int, help="Number of PCA components")
    args = parser.parse_args()

    argutils.print_args(args)

    layer = ManoLayer(
        flat_hand_mean=False,
        side='right',
        mano_root=args.mano_root,
        ncomps=args.mano_ncomps,
        use_pca=not args.no_pca,
        root_rot_mode=args.root_rot_mode,
        joint_rot_mode=args.joint_rot_mode)
    if args.root_rot_mode == 'axisang':
        rot = 3
    else:
        rot = 6
    if args.no_pca:
        args.mano_ncomps = 45

    # Generate random pose coefficients
    #pose_params = args.rand_mag * torch.rand(args.batch_size, args.mano_ncomps + rot)
    # Random pose coefficients cause inrealistic hand poses, using a fixed pose from DexYCB instead
    # pose_params = torch.tensor([[ 0.22901694, -0.8658326,   2.056979,    0.7533836,  -0.27545434, -0.3340937,
    #                             0.34736416, -0.15854122,  0.4078512,   0.87308675,  0.06270637,  0.97369653,
    #                             0.0849302,  -1.0316252,  -0.09225126,  0.66329986,  0.8585661,  -1.0765529,
    #                             0.16443409, -0.5211599, -1.8718586,   0.6009506,   0.9062733,  -0.18175313,
    #                             0.06957736, -0.76746035, -0.20473613,  0.5317386,  -0.09889563,  0.01427455,
    #                             0.21685456,  0.18911918,  0.30047104,  0.36480734,  0.1723292,  -0.3447407,
    #                             -0.44491622, -0.02265068, -0.35290015, -0.395249,   -0.47771075, -0.11828388,
    #                             -0.08889918,  0.27671254, -0.10235115,  0.08282909, -0.16244628,  0.12293345,
    #                             -0.15703154,  0.06008038,  0.8856846 ]])
    pose_params = torch.tensor([[ 0.,  0.,   0.,    0.7533836,  -0.27545434, -0.3340937,
                                0.34736416, -0.15854122,  0.4078512,   0.87308675,  0.06270637,  0.97369653,
                                0.0849302,  -1.0316252,  -0.09225126,  0.66329986,  0.8585661,  -1.0765529,
                                0.16443409, -0.5211599, -1.8718586,   0.6009506,   0.9062733,  -0.18175313,
                                0.06957736, -0.76746035, -0.20473613,  0.5317386,  -0.09889563,  0.01427455,
                                0.21685456,  0.18911918,  0.30047104,  0.36480734,  0.1723292,  -0.3447407,
                                -0.44491622, -0.02265068, -0.35290015, -0.395249,   -0.47771075, -0.11828388,
                                -0.08889918,  0.27671254, -0.10235115,  0.08282909, -0.16244628,  0.12293345,
                                0.,  0.,  0. ]])
    globalRot = torch.tensor(R.from_euler('xyz', [0.22901694, -0.8658326,   2.056979]).as_matrix(), dtype=torch.float32)
    print('globalRot', globalRot)
    print('pose_params: shape&content')
    print(pose_params.shape)
    print(pose_params)
    pose_params.requires_grad = True
    if args.random_shape:
        shape = torch.rand(args.batch_size, 10)
    else:
        shape = torch.zeros(1)  # Hack to act like None for PyTorch JIT
    if args.cuda:
        pose_params = pose_params.cuda()
        shape = shape.cuda()
        layer.cuda()

    # Loop for forward/backward quick profiling
    for idx in tqdm(range(args.iters)):
        # Forward pass
        verts, Jtr = layer(pose_params, shape)
        
        # Backward pass
        loss = torch.norm(verts)
        loss.backward()

    if not args.no_display:
        verts, Jtr = layer(th_pose_coeffs=pose_params, th_betas=shape, th_trans=pose_params[:,48:51])
        joints = Jtr.cpu().detach()
        verts = verts.cpu().detach()
        # Draw obtained vertices and joints
        print(joints)

        jt = joints - joints[0,0,:]
        vt = verts - joints[0,0,:]
        rot_mat = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],dtype=torch.float32)
        rot_mat2 = torch.tensor([[1, 0, 0],[0, 0, 1],[0, -1, 0]],dtype=torch.float32)
        jt = torch.matmul(jt, rot_mat)
        jt = torch.matmul(jt, rot_mat2)
        vt = torch.matmul(vt, rot_mat)
        vt = torch.matmul(vt, rot_mat2)
        jt_reorder = rearrange(jt)
        print(jt_reorder + offset[0,0,:])
        #print(jt.size())
        #jt_inv = torch.matmul(jt, torch.inverse(rot_mat))
        #jt_check = torch.matmul(jt_inv, globalRot) + joints[0,0,:]
        #print(jt_check-joints)
        display_hand({
            'verts': vt,
            'joints': jt
        },
                     mano_faces=layer.th_faces)
