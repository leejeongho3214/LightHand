import sys
from matplotlib import pyplot as plt
import os
sys.path.append(os.path.abspath('../..'))
from tqdm import tqdm
from src.utils.miscellaneous import mkdir
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import random_split, ConcatDataset
import random
import os.path as op
import torch
import math
import json
import pickle


class Pkl_transform(Dataset):
    def __init__(self, phase, ratio = 'val'):
        self.phase = phase
        self.root = "../../datasets/ArmHand"
        self.folder_num = os.listdir(
            self.root + "/annotations/train/wrist_angles")
        self.dict = list()
        self.ratio = ratio

    def set_path(self, num):
        if self.phase != "train":
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
            self.img_root = os.path.join(self.root, f"images/{self.phase}")    
            
        else:
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_{num}_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_{num}_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_{num}_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
            self.img_root = os.path.join(self.root, f"images/train/wrist_angles/{num}")
        
        self.store_path = os.path.join(
            self.root, f"annotations/{self.phase}/{self.ratio}_revision_data.pkl")

    def processing(self, num, ratio = 1, img_store = False):
        self.set_path(num)
        dict = list()
        pbar = tqdm(total=len(self.meta['images']))

        for idx, j in enumerate(self.meta['images']):
            if idx == int(len(self.meta['images']) * ratio):     ## This means 0.1 M images
                break
            
            pbar.update(1)
            if j['camera'] == '0':
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint_3d = torch.tensor(
                self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = torch.tensor(
                self.camera['0']['focal'][f'{camera}'][0])
            translation = torch.tensor(self.camera['0']['campos'][f'{camera}'])
            rot = torch.tensor(self.camera['0']['camrot'][f'{camera}'])

            calibrationed_joint = torch.einsum(
                'ij, kj -> ki', rot, (joint_3d - translation))
            calibrationed_joint[:, :2] = calibrationed_joint[:,
                                                             :2]/(calibrationed_joint[:, 2][:, None].repeat(1, 2))
            calibrationed_joint = calibrationed_joint[:,
                                                      :2] * focal_length + 256

            if any(joint[idx] < 50 or joint[idx] > 460 for joint in calibrationed_joint for idx in range(2)):
                continue

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            # If wrist was rotated, it happend black area under rotated wrist
            lowest_wrist_left, lowest_wrist_right = [
                79-256, -256], [174-256, -256]
            rot_lowest_wrist_left = math.cos(
                rad) * lowest_wrist_left[1] - math.sin(rad) * lowest_wrist_left[0] + 256
            rot_lowest_wrist_right = math.cos(
                rad) * lowest_wrist_right[1] - math.sin(rad) * lowest_wrist_right[0] + 256

            if rot_lowest_wrist_left > 0:
                lift_y = rot_lowest_wrist_left

            elif rot_lowest_wrist_right > 0:
                lift_y = rot_lowest_wrist_right

            else:
                lift_y = 0

            translation_y = random.uniform(0, 40)

            calibrationed_joint[:, 0] = math.cos(
                rad) * (calibrationed_joint[:, 0] - 256) + math.sin(rad) * (calibrationed_joint[:, 1] - 256) + 256
            calibrationed_joint[:, 1] = math.cos(rad) * (calibrationed_joint[:, 1] - 256) - math.sin(
                rad) * (calibrationed_joint[:, 0] - 256) + 256 + lift_y + translation_y

            if any(joint[idx] < 50 or joint[idx] > 460 for joint in calibrationed_joint for idx in range(2)):
                continue
            
            if img_store:
                image_path = os.path.join(self.img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:]))
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rot_image = i_rotate(image, degrees, 0, (lift_y+ translation_y))
                
                bg_path = "../../datasets/ArmHand/background"
                bg_list = os.listdir(bg_path)
                bg_len = len(bg_list)
                bg_img = cv2.imread(os.path.join(bg_path, bg_list[idx%bg_len]))
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = cv2.resize(bg_img, (512, 512))
                
                index = np.where((rot_image[:, :, 0] == 0) & (rot_image[:, :, 1] == 0) & (rot_image[:, :, 2] == 0))
                rot_image[index] = bg_img[index]
                
                img_root_list = self.img_root.split('/')
                img_root_list[3] = f"new_Armo_{self.ratio}"
                img_root = '/'.join(img_root_list)
                new_img_name = os.path.join(img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:]))
                new_img_fold_path = os.path.join(img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:-1]))
                
                calibrationed_joint = np.array(calibrationed_joint).astype(np.float64)
                bbox = [(min(calibrationed_joint[:, 1]),  min(calibrationed_joint[:, 0])) , (max(calibrationed_joint[:, 1]), max(calibrationed_joint[:, 0]))] ## ((min_row, min_col), (max_row, max_col))
                
                if not os.path.isdir(new_img_fold_path):
                    mkdir(new_img_fold_path)
                
                cv2.imwrite(new_img_name, rot_image[:, :, (2, 1, 0)])
                dict.append({'file_name': self.meta['images'][idx]['file_name'], 'joint_2d': calibrationed_joint.tolist(
                ), 'joint_3d': joint_3d.tolist(), 'bbox': bbox})

                
            else:
                self.dict.append({'file_name': self.meta['images'][idx]['file_name'], 'joint_2d': calibrationed_joint.tolist(
                ), 'joint_3d': joint_3d.tolist(), 'move': int(lift_y + translation_y), 'degree': int(degrees), 'angle': num})
                
        return dict
            
    def get_json(self, ratio = 1):
        if self.phase == "train":
            for num in self.folder_num:
                _ = self.processing(num, ratio)
        else:
            _ = self.processing(self.phase)

        with open(self.store_path, 'wb') as f:
            pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Done ===> {self.store_path}")
        
    def get_image(self, ratio = 1):
        if self.phase == "train":
            for num in self.folder_num:
                dict = self.processing(num, ratio, img_store = True)
                store_path = f"../../datasets/new_Armo_{self.ratio}/annotations/train/wrist_angels/{num}/CISLAB_{num}_rot_data.json"
                if not os.path.isdir(f"../../datasets/new_Armo_{self.ratio}/annotations/train/wrist_angels/{num}"):
                    mkdir(f"../../datasets/new_Armo_{self.ratio}/annotations/train/wrist_angels/{num}")
                with open(store_path, 'w') as f:
                    json.dump(dict, f)
        else:
            dict = self.processing(self.phase, ratio, img_store = True)
            store_path = f"../../datasets/new_Armo_{self.ratio}/annotations/val/CISLAB_val_rot_data.json"
            if not os.path.isdir(f"../../datasets/new_Armo_{self.ratio}/annotations/val"):
                mkdir(f"../../datasets/new_Armo_{self.ratio}/annotations/val")
            with open(store_path, 'w') as f:
                json.dump(dict, f)

        print(f"Done ===> {self.store_path}")
        
def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    result = cv2.warpAffine(result, translation, (new_w, new_h))

    return result

def main():
    # num_images = {'0.1M': 0.26, '0.2M': 0.52, '0.3M': 0.78}
    # for ratio in num_images:
    #     Pkl_transform(phase="train", ratio = ratio).get_image(num_images[ratio])
    Pkl_transform(phase="val").get_image()
    print("ENDDDDDD")


if __name__ == '__main__':
    main()