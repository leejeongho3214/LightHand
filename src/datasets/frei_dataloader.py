"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import cv2
import json
import os.path as op
import numpy as np

from src.utils.tsv_file import TSVFile, CompositeTSVFile
from src.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml
from src.utils.image_ops import img_from_base64, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, p):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(p):
            if pt[0] > 0: 
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms


class HandMeshTSVDataset(object):
    def __init__(self, args, img_file, label_file=None, hw_file=None,
                 linelist_file=None, is_train=True, cv2_output=False, scale_factor=1):

        self.args = args
        self.img_file = img_file
        self.label_file = label_file
        self.hw_file = hw_file
        self.linelist_file = linelist_file
        self.img_tsv = self.get_tsv_file(img_file)
        self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
        self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)

        if self.is_composite:
            assert op.isfile(self.linelist_file)
            self.line_list = [i for i in range(self.hw_tsv.num_rows())]
        else:
            self.line_list = load_linelist_file(linelist_file)

        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 90 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224
        self.image_keys = self.prepare_image_keys()
        self.joints_definition = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 
                                'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.root_index = self.joints_definition.index('Wrist')

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.linelist_file,
                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def get_valid_tsv(self):
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.label_tsv:
            return self.label_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise

        rot = 0 # rotation
        sc = 1.0 # scaling


        if self.is_train:
            sc = 1.0 
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp


    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_image(self, idx): 
        line_no = self.get_line_no(idx)
        row = self.img_tsv[line_no]
        # use -1 to support old format with multiple columns.
        cv2_im = img_from_base64(row[-1])
        if self.cv2_output:
            return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        return cv2_im

    def get_annotations(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv[line_no]
            annotations = json.loads(row[1])
            return annotations
        else:
            return []

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to 
        # decode the labels to specific formats for each task. 
        return annotations

    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.hw_tsv:
            return self.hw_tsv[line_no][0]
        elif self.label_tsv:
            return self.label_tsv[line_no][0]
        else:
            return self.img_tsv[line_no][0]

    def __len__(self):
        if self.line_list is None:
            return self.img_tsv.num_rows() 
        else:
            return len(self.line_list)

    def __getitem__(self, idx):

        img = self.get_image(idx)
        img_key = self.get_img_key(idx)
        annotations = self.get_annotations(idx)

        annotations = annotations[0]
        center = annotations['center']
        scale = annotations['scale']
        has_2d_joints = annotations['has_2d_joints']
        has_3d_joints = annotations['has_3d_joints']
        joints_2d = np.asarray(annotations['2d_joints'])
        joints_3d = np.asarray(annotations['3d_joints'])

        if joints_2d.ndim==3:
            joints_2d = joints_2d[0]
        if joints_3d.ndim==3:
            joints_3d = joints_3d[0]

        # Get SMPL parameters, if available
        has_smpl = np.asarray(annotations['has_smpl'])
        pose = np.asarray(annotations['pose'])
        betas = np.asarray(annotations['betas'])

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        
        size= 256
        img = transforms.Resize((size, size))(img)
        
        # Store image before normalization to use it in visualization
        transfromed_img = self.normalize_img(img)

        # normalize 3d pose by aligning the wrist as the root (at origin)
        root_coord = joints_3d[self.root_index,:-1]
        joints_3d[:,:-1] = joints_3d[:,:-1] - root_coord[None,:]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
        # 2d pose augmentation
        joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)
        
        ###################################
        # Masking percantage
        # We observe that 0% or 5% works better for 3D hand mesh
        # We think this is probably becasue 3D vertices are quite sparse in the down-sampled hand mesh 
        mvm_percent = 0.0 # or 0.05
        ###################################

        mjm_mask = np.ones((21,1))
        if self.is_train:
            num_joints = 21
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()

        mvm_mask = np.ones((195,1))
        if self.is_train:
            num_vertices = 195
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()
        meta_data = {}
        meta_data['ori_img'] = img
        meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_smpl'] = has_smpl
        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()

        meta_data['scale'] = float(sc * scale)
        meta_data['center'] = np.asarray(center).astype(np.float32)
        # transfromed_img = transforms.Resize((224, 224))(transfromed_img)

        joint_2d = (meta_data['joints_2d'][:,:-1] * 100 + 112) * (size / 224)
        heatmap= self.generate_target(joint_2d)
        
        return transfromed_img[(2,1,0),:,:], joint_2d, heatmap

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((21, 1), dtype=np.float32)
        target = np.zeros((21,
                        64,
                        64),
                        dtype=np.float32)

        tmp_size = 2 * 3

        for joint_id in range(21):
            feat_stride = [4, 4]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= 64 or ul[1] >= 64 \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 2 ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], 64) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], 64) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], 64)
            img_y = max(0, ul[1]), min(br[1], 64)

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        # if self.use_different_joints_weight:
        #     target_weight = np.multiply(target_weight, 1)

        return torch.tensor(target)

def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred



def vector_to_heatmaps(keypoints):
    """
    Creates 2D heatmaps from keypoint locations for a single image
    Input: array of size N_KEYPOINTS x 2
    Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    """
    N_KEYPOINTS = 21
    MODEL_IMG_SIZE = 224
    heatmaps = np.zeros([N_KEYPOINTS, MODEL_IMG_SIZE, MODEL_IMG_SIZE])
    for k, (x, y) in enumerate(keypoints[:,:2]):
        x, y = int(x), int(y)
        if x > 223:
            x = 223
        if y > 223:
            y = 223

        heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


class HandMeshTSVYamlDataset(HandMeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, args, yaml_file, is_train=True, cv2_output=False, scale_factor=1):
        self.cfg = load_from_yaml_file(yaml_file)
        self.is_composite = self.cfg.get('composite', False)
        self.root = op.dirname(yaml_file)
        
        if self.is_composite==False:
            img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
            label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                                self.root)
            hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)
        else:
            img_file = self.cfg['img']
            hw_file = self.cfg['hw']
            label_file = self.cfg.get('label', None)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)

        super(HandMeshTSVYamlDataset, self).__init__(
            args, img_file, label_file, hw_file, linelist_file, is_train, cv2_output=cv2_output, scale_factor=scale_factor)
