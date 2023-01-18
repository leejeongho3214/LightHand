import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torch
from torch.utils import data
from src.utils.argparser import parse_args
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  #
sys.path.append("/home/jeongho/tmp/Wearable_Pose_Model")
from dataset import *
from src.utils.loss import *
from src.datasets.build import make_hand_data_loader
from src.utils.geometric_layers import *
from src.utils.metric_logger import AverageMeter
from src.utils.visualize import *
import mediapipe as mp
from src.utils.drewing_utils import *

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def media_test(args, test_dataloader, T, dataset_name):
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1) as hands:
        pck_losses = AverageMeter()
        epe_losses = AverageMeter()
        count = 0
        for iteration, (images, gt_2d) in enumerate(test_dataloader):
            images = images[0]
            images = np.array(images)
            results = hands.process(cv2.flip(cv2.cvtColor(images, cv2.COLOR_BGR2RGB), 1))
            if not results.multi_hand_landmarks:
                continue
            # Draw hand landmarks of each hand.
            # print(f'Hand landmarks of {file}:')
            # image_hight, image_width = image.shape

            if len(results.multi_hand_landmarks) > 1:
                assert "This has two-hands"
            joint_2d = []
            iimage = images.copy()
            for hand_landmarks in results.multi_hand_landmarks:

                joint = draw_landmarks(
                    iimage,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            if len(joint) < 20: continue
            for i in range(21):
                joint_2d.append(joint[i])

            joint_2d = torch.tensor(joint_2d)
            joint_2d[:,0] = 224 - joint_2d[:,0]
            correct, visible_point, thresh = PCK_2d_loss_No_batch(joint_2d, gt_2d[0] ,images, T, threshold = 'proportion')
            epe_loss, epe_per = EPE(torch.unsqueeze(joint_2d, 0), gt_2d)
            pck_losses.update_p(correct, visible_point)
            epe_losses.update_p(epe_loss[0], epe_loss[1])

            count += 1
            if T == 0.05:
                fig = plt.figure()
                visualize_gt_media(images, gt_2d, fig, iteration)
                visualize_prediction_media(images, joint_2d.unsqueeze(0), fig, 'mediapipe', iteration,args, dataset_name)
                plt.close()
            if iteration == len(test_dataloader)-1:
                print(count)
                
        return pck_losses.avg, epe_losses.avg, thresh


def main(args, T):
    count = 0
    args.name = "final_models/MediaPipe/checkpoint-good/state_dict.bin"

    path = "../../datasets/our_testset"
    folder_path = os.listdir(path)

    categories = ['general', 'p', 't', 't+p']
    for name in categories:
        count = 0
        for num in folder_path:
            if num[0] == '.' or not os.path.isdir(f"{path}/{num}/{name}"):
                continue
            if count == 0:
                dataset = Our_testset_media(path, os.path.join(num,name))
                count += 1
                continue
            if count > 0:
                previous_dataset = Our_testset_media(path, os.path.join(num,name))
                globals()[f'dataset_{name}'] = ConcatDataset([previous_dataset, dataset])

    loss = []
    for set_name in categories:
        
        data_loader = data.DataLoader(dataset=globals()[f'dataset_{set_name}'], batch_size=args.batch_size, num_workers=0, shuffle=False)

        # pck, mpjpe = test(args, data_loader, _model, 0, 0, best_loss, T, set_name)
        pck, mpjpe, thresh = media_test(args, data_loader, T, set_name)
        if thresh == 'pixel':
            print("Model_Name = {}  // {} // Threshold = {} mm // pck===> {:.2f}% // epe===> {:.2f}mm // {} // {}".format(args.name[13:-31], thresh, T * 20, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
        else:
            print("Model_Name = {}  // {} // Threshold = {} // pck===> {:.2f}% // epe===> {:.2f}mm // {} // {}".format(args.name[13:-31], thresh, T, pck * 100, mpjpe * 0.26, len(globals()[f'dataset_{set_name}']), set_name))
        loss.append([set_name, pck * 100, mpjpe * 0.26])
    print("==" * 80)
    return loss


if __name__ == "__main__":
    args = parse_args()
    T_list = [0.025, 0.05, 0.075, 0.1]
    loss = main(args, T=T_list[0])
    loss1 = main(args, T=T_list[1])
    loss2 = main(args, T=T_list[2])
    loss3 = main(args, T=T_list[3])
    for idx,i in enumerate(loss):
        print("dataset = {} ,{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(i[0],loss[idx][1], loss1[idx][1], loss2[idx][1], loss3[idx][1], loss[idx][2]))
