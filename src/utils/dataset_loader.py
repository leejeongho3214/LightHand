import os
import sys
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


class HIU_Dataset(Dataset):
    def __init__(self, args):
        image_list = []
        for (root, _, files) in os.walk("../../datasets/HIU_DMTL"):
            for file in files:
                if not file.endswith('.json') and not file.endswith('_mask.png') and not file.endswith('_mask.jpg'):
                    file_path = os.path.join(root, file)
                    anno_name = file_path[:-4] + '.json'
                    if os.path.isfile(os.path.join(root, anno_name)):
                        image_list.append((file_path, anno_name))
        self.image = image_list
        self.args = args

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        image = Image.open(self.image[idx][0])
        scale_x = size / image.width
        scale_y = size / image.height

        with open(self.image[idx][1], "r") as st_json:
            annotation = json.load(st_json)

        if annotation['hand_type'][0] == 0:
            joint = annotation['pts2d_2hand'][21:]
        else:
            joint = annotation['pts2d_2hand'][:21]
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        joint_2d = torch.tensor(joint)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d / 2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)

        return trans_image, joint_2d, heatmap, torch.ones(21, 3)
    
class Frei(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/frei_test/evaluation/rgb"
        with open("../../datasets/frei_test/evaluation_K.json", "r") as st_json:
            self.anno_K = json.load(st_json)
        with open("../../datasets/frei_test/evaluation_xyz.json", "r") as st_json:
            self.anno_xyz = json.load(st_json)
        # with open("../../datasets/frei_test/evaluation_mano.json", "r") as st_json:
        #     self.anno_mano = json.load(st_json)
            
    def __len__(self):
        return len(self.anno_K)
    
    def __getitem__(self, idx):
        anno_K = torch.tensor(self.anno_K[idx])
        anno_xyz = torch.tensor(self.anno_xyz[idx])
        # anno_mano = torch.tensor(self.anno_mano[idx][0][:-3])
        
        joint_2d = torch.matmul(anno_K, anno_xyz.T).T
        joint_2d = (joint_2d[:, :2].T / joint_2d[:, -1]).T
        
        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
        
        image = Image.open(os.path.join(self.img_path, f"{str(idx).zfill(8)}.jpg"))
        scale_x = size / image.width
        scale_y = size / image.height

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_image = trans(image)
        joint_2d[:, 0] = joint_2d[:, 0] * scale_x
        joint_2d[:, 1] = joint_2d[:, 1] * scale_y

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint_2d/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint_2d/4)

        return trans_image, joint_2d, heatmap, anno_xyz
    
class Dataset_interhand(torch.utils.data.Dataset):
    def __init__(self, transform, mode, args):
        self.args = args
        self.mode = mode  # train, test, val
        self.img_path = '../../datasets/InterHand2.6M/images'
        self.annot_path = '../../datasets/InterHand2.6M/annotations'
        if self.mode == 'val':
            self.rootnet_output_path = '../../datasets/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = '../../datasets/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transform
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
        if (self.mode == 'val' or self.mode == 'test'):
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

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
            if (self.mode == 'val' or self.mode == 'test'):
                bbox = np.array(
                    rootnet_result[str(aid)]['bbox'], dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(
                    aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
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

        if self.args.model == "ours":
            size = 224
        else:
            size = 256
        

        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])

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
        
        joint[:, 0] = joint[:, 0] * (size / ori_img.width)
        joint[:, 1] = joint[:, 1] * (size / ori_img.height)
        targets = torch.tensor(joint[:21, :-1])

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(targets/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(targets/4)

        return img, targets, heatmap, torch.ones(21, 3)
    
class Rhd(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = "../../datasets/rhd"
        with open("../../datasets/rhd/annotations/rhd_train.json", "r") as st_json:
            self.anno = json.load(st_json)
        print()

    def __len__(self):
        return len(self.anno['annotations'])

    def __getitem__(self, idx):
        ori_img = Image.open(os.path.join(
            self.img_path, self.anno['images'][idx]['file_name']))
        joint = np.array(self.anno['annotations'][idx]['keypoints'])
        bbox = list(map(int, self.anno['annotations'][idx]['bbox']))
        img = np.array(ori_img)

        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
            
        if bbox[2] % 2 == 1: bbox[2] - 1
        if bbox[3] % 2 == 1: bbox[3] - 1
        space_l = int(224 - bbox[3]) / 2; space_r = int(224 - bbox[2]) / 2
        if (bbox[1] - space_l) < 0: space_l = bbox[1]
        if (bbox[1] + bbox[3] + space_l) > ori_img.height: space_l = ori_img.height - (bbox[1] + bbox[3]) - 1
        if (bbox[0] - space_r) < 0: space_r = bbox[0]
        if (bbox[0] +  bbox[2] + space_r) > ori_img.width: space_r = ori_img.width - (bbox[0] + bbox[2]) - 1
        
        img = img[int(bbox[1] - space_l): int(bbox[1] + bbox[3] + space_l), int(bbox[0] -  space_r) : int(bbox[0] + bbox[2] +  space_r)]
        joint[:, 0] = (joint[:, 0] - bbox[0] + space_r) * (self.anno['images'][idx]['width']/(bbox[2] + 2*space_r))
        joint[:, 1] = (joint[:, 1] - bbox[1] + space_l) * (self.anno['images'][idx]['width']/(bbox[3] + 2*space_l))
        
        joint_order = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12,
                       11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        joint = joint[joint_order, :]
        joint[:, 0] = joint[:, 0] * (size / self.anno['images'][idx]['width'])
        joint[:, 1] = joint[:, 1] * (size / self.anno['images'][idx]['height'])
        joint = torch.tensor(joint)

        if self.args.model == "hrnet":
            heatmap = GenerateHeatmap(128, 21)(joint/2)
        else:
            heatmap = GenerateHeatmap(64, 21)(joint/4)
        img = Image.fromarray(img)
        img = trans(img)

        return img, joint[:, :2].float(), heatmap, torch.ones(21, 3)
    
    
def add_our(args, dataset, folder_num, path):
    from src.tools.dataset import CustomDataset
    for iter, degree in enumerate(folder_num):
        ratio  = (len(dataset) * args.ratio_of_other) / 1528415     ## it divides the length of dataset into totla_len
        dataset = CustomDataset(args, degree, path, color=args.color,
                                ratio_of_aug=args.ratio_of_aug, ratio_of_dataset= ratio)

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
