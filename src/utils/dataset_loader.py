import os
import pickle
import cv2
from matplotlib import pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from src.utils.transforms import world2cam, cam2pixel
from src.utils.preprocessing import load_skeleton, process_bbox
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os.path as op
from torch.utils.data import random_split, ConcatDataset
import numpy as np



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
        hms = np.zeros(shape=(self.num_parts, self.output_res,
                       self.output_res), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(p):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms



class Dataset_interhand(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        root_path = "/mnt/SSD01_1tb/jeongho/dataset"
        self.args = args
        self.mode = mode  # train, test, val
        self.img_path = os.path.join(root_path, 'InterHand2.6M_5fps_batch1/images')
        self.annot_path = os.path.join(root_path, 'InterHand2.6M_5fps_batch1/annotations')
        # if self.mode == 'val':
        #     self.rootnet_output_path = os.path.join(root_path, 'interhand2.6m/rootnet_output/rootnet_interhand2.6m_output_val.json')
        # else:
        #     self.rootnet_output_path = os.path.join(root_path, 'interhand2.6m/rootnet_output/rootnet_interhand2.6m_output_test.json')
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(
            0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num*2)}
        self.skeleton = load_skeleton(
            op.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        # load annotation
        print("Load annotation from  " + op.join(self.annot_path, self.mode))
        db = COCO(op.join(self.annot_path, self.mode,
                  'InterHand2.6M_' + self.mode + '_data.json'))
        with open(op.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(op.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
        # if (self.mode == 'val' or self.mode == 'test'):
        #     print("Get bbox and root depth from " + self.rootnet_output_path)
        #     rootnet_result = {}
        #     with open(self.rootnet_output_path) as f:
        #         annot = json.load(f)
        #     for i in range(len(annot)):
        #         rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        # else:
        #     print("Get bbox and root depth from groundtruth annotation")

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = op.join(self.img_path, self.mode, img['file_name'])

            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(
                frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(
                1, 0), camrot, campos.reshape(3, 1)).transpose(1, 0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']
                        ] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']
                        ] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array(
                (ann['hand_type_valid']), dtype=np.float32)

            # if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            # if (self.mode == 'val' or self.mode == 'test'):
            #     bbox = np.array(
            #         rootnet_result[str(aid)]['bbox'], dtype=np.float32)
            #     abs_depth = {'right': rootnet_result[str(
            #         aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            # else:
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'],
                                            2], 'left': joint_cam[self.root_joint_idx['left'], 2]}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam,
                     'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint, 'hand_type': hand_type,
                    'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            # if hand_type == 'right' or hand_type == 'left':
            if hand_type == 'right':
                if np.array(Image.open(img_path)).ndim != 3:
                    continue
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        # self.datalist = self.datalist_sh + self.datalist_ih
        self.datalist = self.datalist_sh
        print('Number of annotations in single hand sequences: ' +
              str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' +
              str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data[
            'bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        # 3rd dimension means depth-relative value
        joint = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)

        image_size = 256

        
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ]) 

        ori_img = Image.open(img_path)
        
        bbox = list(map(int, bbox))
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[0] < 0:
            bbox[0] = 0        
        if bbox[2] % 2 == 1: bbox[2] - 1
        if bbox[3] % 2 == 1: bbox[3] - 1
        space_l = int(224 - bbox[3]) / 2; space_r = int(224 - bbox[2]) / 2
        if (bbox[1] - space_l) < 0: space_l = bbox[1]
        if (bbox[1] + bbox[3] + space_l) > ori_img.height: space_l = ori_img.height - (bbox[1] + bbox[3]) - 1
        if (bbox[0] - space_r) < 0: space_r = bbox[0]
        if (bbox[0] +  bbox[2] + space_r) > ori_img.width: space_r = ori_img.width - (bbox[0] + bbox[2]) - 1
        
        # img = img[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        joint[:, 0] = (joint[:, 0] - bbox[0] + space_r) * (ori_img.width/(bbox[2] + 2*space_r))
        joint[:, 1] = (joint[:, 1] - bbox[1] + space_l) * (ori_img.height/(bbox[3] + 2*space_l))
        
        img = np.array(ori_img)[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        img = Image.fromarray(img); img = trans(img)
    
        # reorganize interhand order to ours
        joint = joint[(
            20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16), :]
        # joint[:, 0] = joint[:, 0] - bbox[0]; joint[:, 1] = joint[:, 1] - bbox[1]
        
        joint[:, 0] = joint[:, 0] * (image_size / ori_img.width)
        joint[:, 1] = joint[:, 1] * (image_size / ori_img.height)
        targets = torch.tensor(joint[:21, :-1])
        heatmap = self.generate_target(targets)

        return img, targets, heatmap
    
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
    
class RHD(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.path = "../../dataset/RHD_published_v2"
        self.phase = phase
        with open(f"../../dataset/RHD_published_v2/{phase}/anno_{phase}.pickle", "rb") as st_json:
            self.p_anno = pickle.load(st_json)
        self.sorting()

    def __len__(self):
        return len(self.anno)
    
    def sorting(self):
        sorted_list = []
        for idx in tqdm(self.p_anno.keys()):
            seg_img = cv2.imread(os.path.join(
            self.path, self.phase, "mask", f"{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        
            # 17보다 큰 픽셀들의 좌표 찾기
            indices = np.where(seg_img > 17)
            
            if (len(indices[0]) == 0 or len(indices[1]) == 0):
                sorted_list.append(idx)
            else:
                # Bounding Box의 좌표 계산 (최소값, 최대값)
                x_min, y_min = np.min(indices[1]), np.min(indices[0])
                x_max, y_max = np.max(indices[1]), np.max(indices[0])
                if (x_max - x_min) < 30 or (y_max - y_min) < 30:
                    sorted_list.append(idx)
                
        self.anno = [[idx, self.p_anno[idx]] for idx in self.p_anno.keys() if idx not in sorted_list]
        # self.anno = {idx: item for idx, item in self.anno.items() if int(idx) not in sorted_list}

    def __getitem__(self, idx):
        ori_img = Image.open(os.path.join(
            self.path, self.phase, "color", f"{self.anno[idx][0]:05d}.png"))
        ori_img = np.array(ori_img)

        joint_z = np.matmul(self.anno[idx][1]["K"], self.anno[idx][1]["xyz"].T).T
        joint = joint_z / joint_z[:, -1].reshape(-1, 1)
        joint = joint[21:]
        
        # Bounding Box의 좌표 계산 (최소값, 최대값)
        h_min, w_min = np.min(joint[:,1]), np.min(joint[:,0])
        h_max, w_max = np.max(joint[:,1]), np.max(joint[:,0])
        

        spare = int(max(w_max-w_min, h_max-h_min) * 0.4)
        s_h_max = max(int(h_max + spare), 0)
        s_h_min = min(int(h_min - spare), ori_img.shape[0])
        s_w_max = max(int(w_max + spare), 0)
        s_w_min = min(int(w_min - spare), ori_img.shape[1])
        img = ori_img[s_h_min:s_h_max, s_w_min:s_w_max]

        joint[:, 1] = (joint[:, 1] - s_h_min) / (s_h_max - s_h_min)
        joint[:, 0] = (joint[:, 0] - s_w_min) / (s_w_max - s_w_min)

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        
        joint_order = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12,
                       11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])]
                                   )
        joint = joint[joint_order, :]
        joint[:, 0] = joint[:, 0] * size
        joint[:, 1] = joint[:, 1] * size
        joint = torch.tensor(joint)

        # heatmap = GenerateHeatmap(64, 21)(joint/4)
        heatmap = self.generate_target(joint)
        img = Image.fromarray(img)
        img = trans(img)

        return img, joint[:, :2].float(), heatmap
    
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
    
class STB(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../../../../../data1/STB"
        anno_list = os.listdir("../../../../../../data1/STB/labels")
        self.meta = {}
        for a_path in anno_list:
            anno = sio.loadmat(os.path.join("../../../../../../data1/STB/labels", a_path))
            # self.meta[a_path] = anno
            
            image = np.array(Image.open(os.path.join(self.img_path, "images", "B1Counting", "BB_right_0.png")))
            parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
            
            # base line = 120.054 fx = 822.79041 fy = 822.79041 tx = 318.47345 ty = 250.31296
            colorKmat = np.array([[607.92271, 0, 314.78337],  [0, 607.88192, 236.42484],  [0, 0, 1]])
            
            anno["handPara"][:, :, 0] = anno["handPara"][:, :, 0] - np.expand_dims(np.array([318-120, 250-120, 0]), axis =1).repeat(21, axis = 1)
            anno["handPara"][:, :, 0] = np.dot(colorKmat, anno["handPara"][:, :, 0])
            anno["handPara"][:, :, 0][:-1] = anno["handPara"][:, :, 0][:-1] / anno["handPara"][:, :, 0][2]
            
            # anno["handPara"][:, :, 0] = anno["handPara"][:, :, 0] + np.expand_dims(np.array([ 318 - 120, 250 - 120, 0]), axis =1).repeat(21, axis = 1)
            
            for i in range(21):
                cv2.circle(image, (int(anno["handPara"][0][i][0]), int(anno["handPara"][1][i][0])), 2, [0, 1, 0],
                        thickness=-1)
                if i != 0:
                    cv2.line(image, (int(anno["handPara"][0][i][0]), int(anno["handPara"][1][i][0])),
                            (int(anno["handPara"][0][parents[i]][0]), int(anno["handPara"][1][parents[i]][0])),
                            [0, 0, 1], 1)
            plt.imshow(image)
            plt.savefig('sample2.png')
            print()

    def __len__(self):
        return len(self.anno['annotations'])

    def __getitem__(self, idx):
        print()


class GAN(Dataset):
    def __init__(self, args,):
        self.args = args
        self.img_path = "../../dataset/GANeratedHands_Release/data/noObject"
        img_folder= os.listdir(self.img_path)
        self.meta = list()
        
        for i_folder in img_folder:
            t_list = os.listdir(os.path.join(self.img_path, i_folder))
            for name in t_list:
                if name.split(".")[-1] == "png":
                    img_num = name.split("_")[0]
                    name = os.path.join(i_folder, name)
                    anno_name = os.path.join(i_folder, img_num + "_joint2D.txt")
                    self.meta.append((name, anno_name))

    def __len__(self):
            
        return len(self.meta)

    def __getitem__(self, idx):
        
        f = open(os.path.join(self.img_path, self.meta[idx][1])).read()
        try:
            anno = f.split(',')
            anno[-1] = anno[-1][:-1
                                ]
            anno = list(map(float, anno))
            anno = np.array(anno, dtype = int).reshape(21, -1)
            joint_2d = torch.tensor(anno)
        except:
            print()
        
        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        
        image = Image.open(os.path.join(self.img_path, self.meta[idx][0]))

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])])

        trans_image = trans(image)

        heatmap = GenerateHeatmap(64, 21)(joint_2d/4)
        
        return trans_image, joint_2d, heatmap
        
     
    
def add_our(args, dataset, folder_num, path):
    from src.tools.dataset import CustomDataset
    ratio  = (117216 * args.ratio_of_other) / 1375570

     ## it divides the length of dataset into totla_len
    for iter, degree in enumerate(folder_num):
        dataset = CustomDataset(args, degree, path, ratio_of_dataset= ratio)

        if iter == 0:
            train_dataset, test_dataset = random_split(
                dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

        else:
            train_dataset_other, test_dataset_other = random_split(
                dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
            train_dataset = ConcatDataset(
                [train_dataset, train_dataset_other])
            test_dataset = ConcatDataset(
                [test_dataset, test_dataset_other])
                
    trainset_dataset = ConcatDataset([train_dataset, trainset_dataset])
    testset_dataset = ConcatDataset([test_dataset, testset_dataset])
    
    return trainset_dataset, test_dataset

def our_cat(args, folder_num, path):
    from src.tools.dataset import CustomDataset
    for iter, degree in enumerate(folder_num):
        if iter == 0 :
            train_dataset = CustomDataset(args, degree, path,
                                ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= args.ratio_of_our)
        else:
            train_dataset_other = CustomDataset(args, degree, path, 
                                ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= args.ratio_of_our)
            train_dataset = ConcatDataset(
                [train_dataset, train_dataset_other])
    return train_dataset
