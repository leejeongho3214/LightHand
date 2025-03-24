import sys
from matplotlib import pyplot as plt
import os
sys.path.append(os.getcwd())
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
    def __init__(self, phase, input_size):
        self.input_size = input_size / 2
        self.phase = phase
        self.root = "../../dataset/ArmHand"
        self.dict = list()

    def set_path(self):
        with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_camera.json"), "r") as st_json:
            self.camera = json.load(st_json)
        with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_joint_3d.json"), "r") as st_json:
            self.joint = json.load(st_json)
        with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_data.json"), "r") as st_json:
            self.meta = json.load(st_json)
        self.img_root = os.path.join(self.root, f"images/{self.phase}/Capture0")    


    def processing(self):
        self.set_path()
        joint_list = list()
        pbar = tqdm(total=len(self.meta['images']))

        for idx, j in enumerate(self.meta['images']):
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
                                                      :2] * focal_length + self.input_size

            # if any(joint[idx] < 50 or joint[idx] > 460 for joint in calibrationed_joint for idx in range(2)):
            #     continue
            
            if any(joint[idx] < 20 or joint[idx] > 200 for joint in calibrationed_joint for idx in range(2)):
                continue

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            # If wrist was rotated, it happend black area under rotated wrist
            lowest_wrist_left, lowest_wrist_right = [
                79-self.input_size, -self.input_size], [174-self.input_size, -self.input_size]
            rot_lowest_wrist_left = math.cos(
                rad) * lowest_wrist_left[1] - math.sin(rad) * lowest_wrist_left[0] + self.input_size
            rot_lowest_wrist_right = math.cos(
                rad) * lowest_wrist_right[1] - math.sin(rad) * lowest_wrist_right[0] + self.input_size

            if rot_lowest_wrist_left > 0:
                lift_y = rot_lowest_wrist_left

            elif rot_lowest_wrist_right > 0:
                lift_y = rot_lowest_wrist_right

            else:
                lift_y = 0

            # translation_y = random.uniform(0, 40)
            translation_y = random.uniform(0, 17)

            calibrationed_joint[:, 0] = math.cos(
                rad) * (calibrationed_joint[:, 0] - self.input_size) + math.sin(rad) * (calibrationed_joint[:, 1] - self.input_size) + self.input_size
            calibrationed_joint[:, 1] = math.cos(rad) * (calibrationed_joint[:, 1] - self.input_size) - math.sin(
                rad) * (calibrationed_joint[:, 0] - self.input_size) + self.input_size + lift_y + translation_y

            if any(joint[idx] < 20 or joint[idx] > 200 for joint in calibrationed_joint for idx in range(2)):
                continue
            
            image_path = os.path.join(self.img_root, '/'.join(self.meta['images'][idx]['file_name'].split('/')[1:]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rot_image = i_rotate(image, degrees, 0, (lift_y+ translation_y))
            
            new_img_path = os.path.join("../../dataset/LightHand", f"images/{self.phase}/{'/'.join(self.meta['images'][idx]['file_name'].split('/')[1:])}")
            mkdir(os.path.dirname(new_img_path))
            
            joint_list.append({'file_name': new_img_path, 'joint_2d': calibrationed_joint.tolist()})
            cv2.imwrite(new_img_path, rot_image[:, :, (2, 1, 0)])
            
        return joint_list

        
    def save_dataset(self):
        dict = self.processing()
        store_path = f"../../dataset/LightHand/annotations/{self.phase}/CISLAB_{self.phase}_data.json"
        mkdir(os.path.dirname(store_path))
        with open(store_path, 'w') as f:
            json.dump(dict, f)

        print(f"Done ===> {store_path}")
        
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
    Pkl_transform(phase="train2", input_size=224).save_dataset()
    print("ENDDDDDD")


if __name__ == '__main__':
    main()