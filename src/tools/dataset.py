import sys
from tqdm import tqdm
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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


def build_dataset(args):
    path = "../../../../../../data1/ArmoHand/training"
    if not os.path.isdir(path):
        path = "../../datasets/ArmoHand/training"
        
    if args.eval:
        test_dataset = eval_set(args)
        return test_dataset, test_dataset
    
    if args.test:
        test_dataset = test_set(args, path)
        return test_dataset, test_dataset
    
    assert args.name.split("/")[0] in ["simplebaseline", "hourglass", "hrnet", "ours"], "Please write down the model name in [simplebaseline, hourglass, hrnet, ours], not %s" % args.name.split("/")[0]
    assert args.name.split("/")[1] in ["rhd", "stb", "frei", "interhand","gan", "ours"], "Please write down the dataset name in [rhd, stb, frei, interhand, gan, ours], not %s" % args.name.split("/")[1]
    args.model = args.name.split("/")[0]
    args.dataset = args.name.split("/")[1]
    
    folder = os.listdir(path)
    folder_num = [i for i in folder if i not in ["README.txt", "data.zip"]]
        
    if args.dataset == "interhand":
        train_dataset = Dataset_interhand("train", args)     
        test_dataset = Dataset_interhand("val", args)     

    elif args.dataset == "frei":            ## Frei don't provide 2d anno in valid-set
        dataset = make_hand_data_loader(                
            args, args.train_yaml,  is_train=True)                                
        eval_path = "/".join(path.split('/')[:-1]) + "/annotations/evaluation"
        train_dataset1 = CustomDataset(args,  path, "train")
        test_dataset1 = val_set(args , eval_path, "val")
        ## This function's name is random_split but i change it to split the dataset by sqeuntial order
        train_dataset2, test_dataset2 = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))]) 
        
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        test_dataset = ConcatDataset([test_dataset1, test_dataset2])
 
    elif args.dataset == "rhd":
        train_dataset = RHD(args, "train")
        test_dataset = RHD(args, "test")      
    
    elif args.dataset == "stb":
        train_dataset = STB(args)
        test_dataset = STB(args)
        

    elif args.dataset == "gan":
        dataset = GAN(args)     # same reason as above
        train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - (int(len(dataset) * 0.9))])
    
    elif args.dataset == "ours":
        eval_path = "/".join(path.split('/')[:-1]) + "/annotations/evaluation"
        train_dataset = CustomDataset(args,  path, "train")
        test_dataset = val_set(args , eval_path, "val")
    return train_dataset, test_dataset

    

class CustomDataset(Dataset):
    def __init__(self, args, path, phase):
        self.args = args
        self.path = path
        self.phase = phase
        self.ratio_of_aug = args.ratio_of_aug
        if phase == "train":
            with open(f"{path}/revision_data.json", "r") as st_json:
                self.meta = json.load(st_json)
                self.meta = self.meta[: int(len(self.meta) * args.ratio_of_our)]
    
    def __len__(self):
        if self.phase == 'train':
            return int((self.args.ratio_of_other) * 117000)     
        else:   
            return int((self.args.ratio_of_other) * 11700)    

    def __getitem__(self, idx):
        name = self.meta[idx]['file_name']
        move = self.meta[idx]['move']
        degrees = self.meta[idx]['degree']
        
        if self.phase == "train":
            num = self.meta[idx]["num"]
            image = cv2.imread(os.path.join(self.path, num, "images/train" , name))   ## Color order of cv2 is BGR    
        else:
            image = cv2.imread(os.path.join(self.path, name))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.args.model == "ours":
            image_size = 256
            joint_multiply = 256/224
        else:
            image_size = 224
            joint_multiply = 1

        if self.phase == "train": 
            image = i_rotate(image, degrees, 0, move)
            joint_2d = torch.tensor(self.meta[idx]['rot_joint_2d']) * joint_multiply  
        else:
            joint_2d = torch.tensor(self.meta[idx]['joint_2d']) * joint_multiply
            
        image = Image.fromarray(image) 
        if idx < len(self.meta) * self.ratio_of_aug:
            trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    
        else:
            trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        image = trans(image)
        
        heatmap = GenerateHeatmap(64, 21)(joint_2d / 4)

        return image, joint_2d, heatmap

    
class val_set(CustomDataset):
    def __init__(self,  *args):
        super().__init__(*args)
        self.ratio_of_aug = 0
        self.args.ratio_of_dataset = 1
        with open(os.path.join(self.path, "evaluation_data_update.json"), "r") as st_json:
            self.meta = json.load(st_json)["images"]
        self.path = "/".join(self.path.split('/')[:-2]) +"/images/evaluation"


class test_set(Dataset):
    def __init__(self, args, path):
        self.args = args
        path = "/".join(path.split("/")[:-1])
        self.image_path = os.path.join(f'{path}', "images/test")
        anno_path = os.path.join(f'{path}', "annotations", "test", "test_data_update.json")
        with open(anno_path, "r") as st_json:
            self.meta = json.load(st_json)

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):
        name = self.meta['images'][idx]['file_name']
        image = cv2.imread(os.path.join(self.image_path, name))   ## Color order of cv2 is BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.args.model == "ours":
            image_size = 256
            joint_multiply = 256/224
        else:
            image_size = 224
            joint_multiply = 1

        image = Image.fromarray(image)
        joint_2d = torch.tensor(self.meta['images'][idx]['joint_2d'])

        trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = trans(image)
        joint_2d[:, 0] = joint_2d[:, 0] * joint_multiply
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
        if not self.args.model == "ours":
            size = 256
        else:
            size = 224
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
        joint_2d_v = torch.concat([joint_2d, visible[:, None]], axis = 1)
        assert len(joint) == 21, f"{file_name} have joint error"
        assert len(visible) == 21, f"{file_name} have visible error"
            
        trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        image = Image.open(f"../../datasets/{file_name}")
        trans_image = trans(image)
        joint_2d_v[:, 0] = joint_2d_v[:, 0] * image.width
        joint_2d_v[:, 1] = joint_2d_v[:, 1] * image.height
        joint_2d[:, 0] = joint_2d[:, 0] * image.width
        joint_2d[:, 1] = joint_2d[:, 1] * image.height
            
        return trans_image, joint_2d_v, pose_type


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
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)
    translation = np.float32([[1, 0, move_x], [0, 1, move_y]])
    result = cv2.warpAffine(result, translation, (new_w, new_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.INTER_LINEAR)

    return result


class Json_transform(Dataset):
    def __init__(self, degree, path):
        self.degree = degree
        self.path = path
        self.degree = degree
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_camera.json", "r") as st_json:
            self.camera = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_joint_3d.json", "r") as st_json:
            self.joint = json.load(st_json)
        with open(f"{path}/{degree}/annotations/train/CISLAB_train_data.json", "r") as st_json:
            self.meta = json.load(st_json)
        self.root = f'{self.path}/{self.degree}/images/train'
        self.store_path = f'{self.path}/{self.degree}/annotations/train/CISLAB_train_data_update.json'

    def get_json_g(self):
        meta_list = self.meta['images'].copy()
        index = []
        pbar = tqdm(total = len(meta_list))
        for idx, j in enumerate(meta_list):
            pbar.update(1)
            if j['camera'] == '0':
                index.append(idx)
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(os.path.join(self.root, name))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            
            for i in range(21):
                a = np.dot(np.array(rot, dtype='float32'),
                           np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
                a[:2] = a[:2] / a[2]
                b = a[:2] * focal_length + 112
                b = torch.tensor(b)

                for u in b:
                    if u > 223 or u < 0:
                        index.append(idx)
                        flag = True
                        break
                if flag:
                    break

                if i == 0:  # 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue

            d = joint_2d.clone()

            flag = False
            for o in joint_2d:
                if o[0] > 220 or o[1] > 220:
                    flag = True
                    index.append(idx)
                    break
            if flag:
                continue
            
            center_j = np.array(d.mean(0))
            move_x = 112 - center_j[0]
            move_y = 112 - center_j[1]
            # tran_image = i_rotate(ori_image, 0, move_x, move_y)
            image = Image.fromarray(ori_image)


            j['joint_2d'] = d.tolist()
            j['joint_3d'] = joint.tolist()
            j['rot_joint_2d'] = joint_2d.tolist()
            j['move_x'] = move_x
            j['move_y'] = move_y
            # j['tran_image'] = tran_image

        count = 0
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        with open(self.store_path, 'w') as f:
            json.dump(self.meta, f)

        print(
            f"Done ===> {self.store_path}")
        
    def get_json(self):
        meta_list = self.meta['images'].copy()
        index = []
        pbar = tqdm(total = len(meta_list))
        for idx, j in enumerate(meta_list):
            pbar.update(1)
            if j['camera'] == '0':
                index.append(idx)
                continue

            camera = self.meta['images'][idx]['camera']
            id = self.meta['images'][idx]['frame_idx']

            joint = torch.tensor(self.joint['0'][f'{id}']['world_coord'][:21])
            focal_length = self.camera['0']['focal'][f'{camera}'][0]
            translation = self.camera['0']['campos'][f'{camera}']
            rot = self.camera['0']['camrot'][f'{camera}']
            flag = False
            name = j['file_name']
            ori_image = cv2.imread(os.path.join(self.root, name))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

            degrees = random.uniform(-20, 20)
            rad = math.radians(degrees)
            left_pixel, right_pixel = [79-112, -112], [174-112, -112]
            left_rot = math.cos(
                rad) * left_pixel[1] - math.sin(rad) * left_pixel[0] + 112
            right_rot = math.cos(
                rad) * right_pixel[1] - math.sin(rad) * right_pixel[0] + 112

            if left_rot > 0:
                move_y = left_rot

            elif right_rot > 0:
                move_y = right_rot

            else:
                move_y = 0
            move_y2 = random.uniform(0, 40)

            for i in range(21):
                a = np.dot(np.array(rot, dtype='float32'),
                           np.array(joint[i], dtype='float32') - np.array(translation, dtype='float32'))
                a[:2] = a[:2] / a[2]
                b = a[:2] * focal_length + 112
                b = torch.tensor(b)

                for u in b:
                    if u > 223 or u < 0:
                        index.append(idx)
                        flag = True
                        break
                if flag:
                    break

                if i == 0:  # 112 is image center
                    joint_2d = b
                elif i == 1:
                    joint_2d = torch.stack([joint_2d, b], dim=0)
                else:
                    joint_2d = torch.concat([joint_2d, b.reshape(1, 2)], dim=0)
            if flag:
                continue

            d = joint_2d.clone()
            x = joint_2d[:, 0] - 112
            y = joint_2d[:, 1] - 112
            joint_2d[:, 0] = math.cos(rad) * x + math.sin(rad) * y + 112
            joint_2d[:, 1] = math.cos(
                rad) * y - math.sin(rad) * x + 112 + move_y + move_y2

            flag = False
            for o in joint_2d:
                if o[0] > 223 or o[1] > 223:
                    flag = True
                    index.append(idx)
                    break
            if flag:
                continue
                    

            j['joint_2d'] = d.tolist()
            j['joint_3d'] = joint.tolist()
            j['rot_joint_2d'] = joint_2d.tolist()
            j['degree'] = degrees
            j['move'] = move_y + move_y2

        count = 0
        for w in index:
            del self.meta['images'][w-count]
            count += 1

        with open(self.store_path, 'w') as f:
            json.dump(self.meta, f)

        print(
            f"Done ===> {self.store_path}")
        
        
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





class Our_testset_media(Dataset):
    def __init__(self, path, folder_name):

        self.image_path = f'{path}/{folder_name}/rgb'
        self.anno_path = f'{path}/{folder_name}/annotations'
        self.list = os.listdir(self.image_path)

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_path, self.list[idx]))
        with open(os.path.join(self.anno_path, self.list[idx])[:-3]+"json", "r") as st_json:
            json_data = json.load(st_json)
            joint_total = json_data['annotations']
            joint = {}
            joint_2d = []

            for j in joint_total:
                if j['label'] != 'Pose':
                    if len(j['metadata']['system']['attributes']) > 0:
                        # Change 'z' to 'indicator function'
                        # Ex. 0 means visible joint, 1 means invisible joint
                        j['coordinates']['z'] = 0
                        joint[f"{int(j['label'])}"] = j['coordinates']
                    else:
                        j['coordinates']['z'] = 1
                        joint[f"{int(j['label'])}"] = j['coordinates']

            if len(joint) < 21:
                assert f"This {idx}.json is not correct"

            for h in range(0, 21):
                joint_2d.append(
                    [joint[f'{h}']['x'], joint[f'{h}']['y'], joint[f'{h}']['z']])
        joint_2d = torch.tensor(joint_2d)

        return image, joint_2d



    
class Json_e(Json_transform):
    def __init__(self, phase):
        root = "datasets/general_2M"
        if phase == 'eval':
            try:
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)       
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
                
            except:
                root = "../../datasets/ArmoHand"
                with open(os.path.join(root, "annotations/evaluation/evaluation_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)   
                with open(os.path.join(root, "annotations/evaluation/evaluation_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/evaluation/evaluation_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/evaluation")
                self.store_path = os.path.join(root, "annotations/evaluation/evaluation_data_update.json")
            
        elif phase == 'test':
                root = "../../../../../../data1/ArmoHand"
                with open(os.path.join(root, "annotations/test/CISLAB_test_camera.json"), "r") as st_json:
                    self.camera = json.load(st_json)   
                with open(os.path.join(root, "annotations/test/CISLAB_test_joint_3d.json"), "r") as st_json:
                    self.joint = json.load(st_json)
                with open(os.path.join(root, "annotations/test/CISLAB_test_data.json"), "r") as st_json:
                    self.meta = json.load(st_json)
                    
                self.root = os.path.join(root, "images/test")
                self.store_path = os.path.join(root, "annotations/test/test_data_update.json")
            
        else:
            root = "../../datasets/general_2M"
            with open(os.path.join(root, "annotations/val/CISLAB_val_camera.json"), "r") as st_json:
                self.camera = json.load(st_json)
            with open(os.path.join(root, "annotations/val/CISLAB_val_joint_3d.json"), "r") as st_json:
                self.joint = json.load(st_json)
            with open(os.path.join(root, "annotations/val/CISLAB_val_data.json"), "r") as st_json:
                self.meta = json.load(st_json)
                
            self.root = os.path.join(root, "images/val")
            self.store_path = os.path.join(root, "annotations/val/CISLAB_val_data_update.json")
    
def main():
    Json_e(phase = "test").get_json_g()
    print("ENDDDDDD")
    
if __name__ == '__main__':
    main()
    
        
