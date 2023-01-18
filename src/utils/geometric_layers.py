"""
Useful geometric operations, e.g. Orthographic projection and a differentiable Rodrigues formula

Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
import math

import torch

def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    
    
def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d

def camera_calibration(X, camera):
    global x, y, z
    camera = camera.view(-1,7)
    rot_matrix = camera[:,:3].detach().cpu().numpy()
    x_t, y_t, z_t = rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2]
    rot_list = []
    for i in range(len(x_t)):
        x, y, z = x_t[i], y_t[i], z_t[i]
        rot = [
            [math.cos(math.radians(z)) * math.cos(math.radians(y)),
             -math.sin(math.radians(z)) * math.cos(math.radians(y)),
             math.sin(math.radians(y))],
            [math.cos(math.radians(z)) * math.sin(math.radians(x)) * math.sin(math.radians(y)) + math.sin(
                math.radians(z)) * math.cos(math.radians(x)),
             - math.sin(math.radians(z)) * math.sin(math.radians(x)) * math.sin(math.radians(y)) + math.cos(
                 math.radians(z)) * math.cos(math.radians(x)),
             - math.sin(math.radians(x)) * math.cos(math.radians(y))],
            [math.sin(math.radians(z)) * math.sin(math.radians(x)) - math.cos(math.radians(z)) * math.cos(
                math.radians(x)) * math.sin(math.radians(y)),
             math.sin(math.radians(z)) * math.cos(math.radians(x)) * math.sin(math.radians(y)) + math.cos(
                 math.radians(z)) * math.sin(math.radians(x)),
             math.cos(math.radians(x)) * math.cos(math.radians(y))]
        ]
        rot_list.append(rot)
    trans = camera[:,3:6].view(-1, 1, 3)
    focal_length = camera[:,-1].view(-1, 1,1)
    rot = torch.tensor(rot_list)
    new_X = (X-trans).transpose(1, -1)
    intrinsiced_X = (rot.cuda() @ new_X).transpose(1,-1)
    X_2d = (intrinsiced_X[:,:,:2]/(intrinsiced_X[:,:,2]).view(-1,21,1)) * (focal_length) * 224 + 112


    return X_2d
