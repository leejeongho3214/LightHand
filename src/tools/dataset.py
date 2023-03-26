import sys
from tqdm import tqdm
import os
sys.path.insert(0, os.path.abspath(
os.path.join(os.path.dirname(__file__), '../..')))

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import random_split, ConcatDataset
import random
import os.path as op
from src.utils.dataset_loader import Dataset_interhand, STB, RHD, GAN, GenerateHeatmap, add_our, our_cat
import torch
import math
import json
from src.datasets.build import make_hand_data_loader
from src.utils.comm import is_main_process
from src.utils.miscellaneous import mkdir
import pickle


def build_dataset(args):
    path = "../../datasets/Armo"

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
        eval_path = "/".join(path.split('/')[:-1]) + "/annotations/evaluation"
        train_dataset1 = CustomDataset(args,  path, "train")
        eval_dataset1 = val_set(args, eval_path, "val")
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
        eval_path = os.path.join(path, "/annotations/evaluation")
        train_dataset = CustomDataset(args,  path, "train")
        eval_dataset = val_set(args, eval_path, "val")

    return train_dataset, eval_dataset


class CustomDataset(Dataset):
    def __init__(self, args, path, phase):
        self.args = args
        self.path = path
        self.phase = phase
        self.ratio_of_aug = args.ratio_of_aug
        if phase == "train":
            with open(f"{path}/revision_data.pkl", "rb") as st_json:
                self.meta = pickle.load(st_json)
                self.meta = self.meta[: int(
                    len(self.meta) * args.ratio_of_our)]
                self.new_meta = [{} for i in range(len(self.meta))]
                self.count = 0

    def __len__(self):
        # if self.phase == 'train':
        #     return int((self.args.ratio_of_other) * 117000)
        # else:
        #     return int((self.args.ratio_of_other) * 11700)
        return len(self.meta) - 1

    def __getitem__(self, idx):
        name = self.meta[idx]['file_name']
        move = self.meta[idx]['move']
        degrees = self.meta[idx]['degree']

        if self.phase == "train":
            num = self.meta[idx]["num"]
            # Color order of cv2 is BGR
            image = cv2.imread(os.path.join(
                self.path, num, "images/train", name))
        else:
            image = cv2.imread(os.path.join(self.path, name))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = 224

        if self.phase == "train":
            image = i_rotate(image, degrees, 0, move)
            joint_2d = torch.tensor(
                self.meta[idx]['rot_joint_2d'])
        else:
            joint_2d = torch.tensor(
                self.meta[idx]['joint_2d'])

        image = Image.fromarray(image)
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

        # plt.imshow(image.permute(1, 2, 0))
        # dir_path = os.path.join("new_dataset",'/'.join(name.split('/')[:-1]))
        # if not os.path.isdir(dir_path):
        #     mkdir(dir_path)
        # plt.savefig(os.path.join("new_dataset", name))

        # self.new_meta[idx]['joint_2d'] = joint_2d
        # self.new_meta[idx]['file_name'] = name
        # self.count += 1
        # if self.count == len(self.meta):
        #     with open('annotation.json', 'w') as f:
        #         json.dump(self.new_meta, f)

        heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)

        return image, joint_2d, heatmap


class val_set(CustomDataset):
    def __init__(self,  *args):
        super().__init__(*args)
        self.ratio_of_aug = 0
        self.args.ratio_of_dataset = 1
        with open(os.path.join(self.path, "revision_data.pkl"), "rb") as st_json:
            self.meta = pickle.load(st_json)["images"]
        self.path = "/".join(self.path.split('/')[:-2]) + "/images/evaluation"


class test_set(Dataset):
    def __init__(self, args, path):
        self.args = args
        path = "/".join(path.split("/")[:-1])
        self.image_path = os.path.join(f'{path}', "images/test")
        anno_path = os.path.join(
            f'{path}', "annotations", "test", "test_data_update.json")
        with open(anno_path, "r") as st_json:
            self.meta = json.load(st_json)

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        # Color order of cv2 is BGR
        image = cv2.imread(os.path.join(self.image_path, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = 224

        image = Image.fromarray(image)
        joint_2d = torch.tensor(self.meta['images'][idx]['joint_2d'])

        trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        joint_2d[:, 0] = joint_2d[:, 0]
        joint_2d[:, 1] = joint_2d[:, 1]

        return image, joint_2d


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

        trans = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image = Image.open(f"../../datasets/{file_name}")
        trans_image = trans(image)
        joint_2d_v[:, 0] = joint_2d_v[:, 0] * 256
        joint_2d_v[:, 1] = joint_2d_v[:, 1] * 256
        joint_2d[:, 0] = joint_2d[:, 0] * 256
        joint_2d[:, 1] = joint_2d[:, 1] * 256

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


class Pkl_transform(Dataset):
    def __init__(self, phase):
        self.phase = phase
        self.root = "../../datasets/Armo"
        self.folder_num = os.listdir(
            self.root + "/annotations/train/wrist_angles")
        self.dict = dict()

    def set_path(self, num):
        if self.phase != "train":
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/{self.phase}/CISLAB_{self.phase}_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
        else:
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_train_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_train_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(self.root, f"annotations/train/wrist_angles/{num}/CISLAB_train_data.json"), "r") as st_json:
                self.meta = json.load(st_json)

        self.img_root = os.path.join(self.root, f"images/{self.phase}")
        self.store_path = os.path.join(
            self.root, f"annotations/{self.phase}/revision_data.pkl")

    def processing(self, num):
        if num != None:
            self.set_path(num)

        pbar = tqdm(total=len(self.meta['images']))
        self.dict["%s" % num] = list()

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
                                                      :2] * focal_length + 112

            if any(joint[idx] < 20 or joint[idx] > 200 for joint in calibrationed_joint for idx in range(2)):
                continue

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            # If wrist was rotated, it happend black area under rotated wrist
            lowest_wrist_left, lowest_wrist_right = [
                79-112, -112], [174-112, -112]
            rot_lowest_wrist_left = math.cos(
                rad) * lowest_wrist_left[1] - math.sin(rad) * lowest_wrist_left[0] + 112
            rot_lowest_wrist_right = math.cos(
                rad) * lowest_wrist_right[1] - math.sin(rad) * lowest_wrist_right[0] + 112

            if rot_lowest_wrist_left > 0:
                lift_y = rot_lowest_wrist_left

            elif rot_lowest_wrist_right > 0:
                lift_y = rot_lowest_wrist_right

            else:
                lift_y = 0

            translation_y = random.uniform(0, 40)

            calibrationed_joint[:, 0] = math.cos(
                rad) * (calibrationed_joint[:, 0] - 112) + math.sin(rad) * (calibrationed_joint[:, 1] - 112) + 112
            calibrationed_joint[:, 1] = math.cos(rad) * (calibrationed_joint[:, 1] - 112) - math.sin(
                rad) * (calibrationed_joint[:, 0] - 112) + 112 + lift_y + translation_y

            if any(joint[idx] < 20 or joint[idx] > 200 for joint in calibrationed_joint for idx in range(2)):
                continue

            self.dict["%s" % num].append({'file_name': self.meta['images'][idx]['file_name'], 'joint_2d': calibrationed_joint.tolist(
            ), 'joint_3d': joint_3d.tolist(), 'move': lift_y + translation_y, 'degree': degrees})

    def get_json(self):
        if self.phase == "train":
            for num in self.folder_num:
                self.processing(num)
        else:
            self.processing(None)

        with open(self.store_path, 'wb') as f:
            pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Done ===> {self.store_path}")


def main():
    Pkl_transform(phase="train").get_json()
    # Json_e(phase="test").get_json_g()
    print("ENDDDDDD")


if __name__ == '__main__':
    main()
