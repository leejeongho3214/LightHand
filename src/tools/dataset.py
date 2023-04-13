import pickle
from src.utils.miscellaneous import mkdir
from src.utils.comm import is_main_process
from src.datasets.build import make_hand_data_loader
import json
import math
import torch
from src.utils.dataset_loader import Dataset_interhand, STB, RHD, GAN, GenerateHeatmap, add_our, our_cat
import os.path as op
import random
from torch.utils.data import random_split, ConcatDataset
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import sys
from tqdm import tqdm
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


def build_dataset(args):
    path = "../../datasets/ArmHand"

    if args.eval:
        test_dataset = eval_set(args)
        return test_dataset, test_dataset

    if args.test:
        test_dataset = test_set(args, path)
        return test_dataset, test_dataset

    assert args.name.split("/")[0] in ["simplebaseline", "hourglass", "hrnet",
                                       "ours"], "Please write down the model name in [simplebaseline, hourglass, hrnet, ours], not %s" % args.name.split("/")[0]
    assert args.name.split("/")[1] in ["rhd", "stb", "frei", "interhand", "gan",
                                       "ours"], "Please write down the dataset name in [rhd, stb, frei, interhand, gan, ours], not %s" % args.name.split("/")[1]

    args.model = args.name.split("/")[0]
    args.dataset = args.name.split("/")[1]

    if args.dataset == "interhand":
        train_dataset = Dataset_interhand("train", args)
        eval_dataset = Dataset_interhand("val", args)

    elif args.dataset == "frei":  # Frei don't provide 2d anno in valid-set
        dataset = make_hand_data_loader(
            args, args.train_yaml,  is_train=True)
        train_dataset1 = CustomDataset(args,  path, "train")
        eval_dataset1 = val_set(args, path, "val")
        # This function's name is random_split but i change it to split the dataset by sqeuntial order
        train_dataset2, eval_dataset2 = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        eval_dataset = ConcatDataset([eval_dataset1, eval_dataset2])

    elif args.dataset == "rhd":
        train_dataset = RHD(args, "train")
        eval_dataset = RHD(args, "test")

    elif args.dataset == "stb":
        train_dataset = STB(args)
        eval_dataset = STB(args)

    elif args.dataset == "gan":
        dataset = GAN(args)     # same reason as above
        train_dataset, eval_dataset = random_split(
            dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])

    elif args.dataset == "ours":
        train_dataset = CustomDataset(args,  path, "train")
        eval_dataset = val_set(args, path, "val")

    return train_dataset, eval_dataset


class CustomDataset(Dataset):
    def __init__(self, args, path, phase):
        self.args = args
        self.path = path
        self.phase = phase
        self.ratio_of_aug = args.ratio_of_aug
        
        self.bg_path = "../../datasets/ArmHand/background"
        self.bg_list = os.listdir(self.bg_path)
        self.bg_len = len(self.bg_list)
        if phase == "train":
            with open(f"{path}/annotations/{phase}/{args.ratio_of_our}M_revision_data.pkl", "rb") as st_json:
                self.meta = pickle.load(st_json)

    def __len__(self):
        return int(len(self.meta) * self.args.ratio_of_other)

    def __getitem__(self, idx):
        name = '/'.join(self.meta[idx]['file_name'].split("/")[1:])
        move = self.meta[idx]['move']
        degrees = self.meta[idx]['degree']

        if self.phase == "train":
            num = self.meta[idx]["angle"]
            # Color order of cv2 is BGR
            image = cv2.imread(os.path.join(
                self.path, "images/train", 'wrist_angles', num, name))
        else:
            image = cv2.imread(os.path.join(self.path, name))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = 256
        
        rot_image = i_rotate(image, degrees, 0, move)
        joint_2d = torch.tensor(
            self.meta[idx]['joint_2d']) * (256/512)

        bg_img = cv2.imread(os.path.join(self.bg_path, self.bg_list[idx%self.bg_len]))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = cv2.resize(bg_img, (512, 512))
        
        index = np.where((rot_image[:, :, 0] == 0) & (rot_image[:, :, 1] == 0) & (rot_image[:, :, 2] == 0))
        rot_image[index] = bg_img[index]

        image = Image.fromarray(rot_image)
        if idx < len(self.meta) * self.ratio_of_aug:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

        else:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            
        image = trans(image)
        heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)

        return image, joint_2d, heatmap


class val_set(CustomDataset):
    def __init__(self,  *args):
        super().__init__(*args)
        self.ratio_of_aug = 0
        self.args.ratio_of_dataset = 1
        with open(os.path.join(f"{self.path}/annotations/{self.phase}", "revision_data.pkl"), "rb") as st_json:
            self.meta = pickle.load(st_json)
        self.path = os.path.join(self.path, "images/val")


class test_set(Dataset):
    def __init__(self, args, path):
        self.args = args
        args.phase = 'test'
        path = os.path.join("/".join(path.split("/")[:-1]), "new_Armo_val")
        self.image_path = os.path.join(f'{path}', "images/val")
        anno_path = os.path.join(
            f'{path}', "annotations/val/CISLAB_val_rot_data.json")
        with open(anno_path, "r") as st_json:
            self.meta = json.load(st_json)



class eval_set(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_path = f'../../datasets/test/rgb'
        self.anno_path = f'../../datasets/test/annotations.json'
        self.list = os.listdir(self.image_path)
        with open(self.anno_path, "r") as st_json:
            self.json_data = json.load(st_json)
        with open('../../datasets/test/annotations.json', "r") as st_json:
            self.json_data = json.load(st_json)

        list_del = list()
        for num in self.json_data:
            if len(self.json_data[f"{num}"]['coordinates']) < 21 or len(self.json_data[f"{num}"]['visible']) < 21:
                list_del.append(num)
        for i in list_del:
            del self.json_data[i]

        self.num = list(self.json_data)

    def __len__(self):
        return len(self.num)

    def __getitem__(self, idx):
        idx = self.num[idx]
        joint = self.json_data[f"{idx}"]['coordinates']
        pose_type = self.json_data[f"{idx}"]['pose_ctgy']
        file_name = self.json_data[f"{idx}"]['file_name']
        visible = self.json_data[f"{idx}"]['visible']
        try:
            joint_2d = torch.tensor(joint)[:, :2]
        except:
            print(file_name)
            print("EROOORROORR")
            
        visible = torch.tensor(visible)
        joint_2d_v = torch.concat([joint_2d, visible[:, None]], axis=1)
        assert len(joint) == 21, f"{file_name} have joint error"
        assert len(visible) == 21, f"{file_name} have visible error"

        img_size = 256 if not self.args.model == 'mediapipe' else 224
        
        
        trans = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image = Image.open(f"../../datasets/{file_name}")
        trans_image = trans(image)
        if self.args.model == 'mediapipe': trans_image = f"../../datasets/{file_name}"
        joint_2d_v[:, 0] = joint_2d_v[:, 0] * img_size
        joint_2d_v[:, 1] = joint_2d_v[:, 1] * img_size
        joint_2d[:, 0] = joint_2d[:, 0] * img_size
        joint_2d[:, 1] = joint_2d[:, 1] * img_size

        return trans_image, joint_2d_v, [pose_type, idx]


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apply(img, aug, num=1, scale=1.5):
    Y = [aug(img) for _ in range(num)]
    return Y


def i_rotate(img, degree, move_x, move_y):
    h, w = img.shape[:-1]

    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    result = cv2.warpAffine(result, translation, (new_w, new_h))

    return result


def save_checkpoint(model, args, epoch, optimizer, best_loss, count, ment, num_trial=10, logger=None):

    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}'.format(
        ment))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer,
                'best_loss': best_loss,
                'count': count,
                'model_state_dict': model_to_save.state_dict()}, op.join(checkpoint_dir, 'state_dict.bin'))

            break
        except:
            pass

    return model_to_save, checkpoint_dir
