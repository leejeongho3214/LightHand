
import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils.miscellaneous import mkdir
from src.utils.dir import reset_folder

def visualize_pred(images, pred_2d_joint, fig, method = None, epoch = 0, iteration = 0, args =None, dataset_name = None):

    num = iteration % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

    
    for i in range(21):
        cv2.circle(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])), 2, [0, 1, 0],
                thickness=-1)
        if i != 0:
            cv2.line(image, (int(pred_2d_joint[num][i][0]), int(pred_2d_joint[num][i][1])),
                    (int(pred_2d_joint[num][parents[i]][0]), int(pred_2d_joint[num][parents[i]][1])),
                    [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image)
    ax1.set_title('pred_image')
    ax1.axis("off")
    
    if method == 'evaluation':
        a = '/'.join(args.output_dir.split('/')[:-2])
        if not os.path.isdir(f"eval_image/{a})"): mkdir(f"eval_image/{a}")
        plt.savefig(os.path.join(f"eval_image/{a}", f'{iteration}.jpg'))

    else:
        epoch_path = f"{args.output_dir}/{method}_image/{epoch}_epoch"
        if iteration == 0 and epoch == 0:
            reset_folder(os.path.dirname(epoch_path))
        if not os.path.isdir(epoch_path): mkdir(epoch_path)
        plt.savefig(os.path.join(epoch_path, f'iter_{iteration}.jpg'))


def visualize_gt(images, gt_2d_joint, fig, iteration):

    num = iteration % images.size(0)
    image = np.moveaxis(images[num].detach().cpu().numpy(), 0, -1)
    image = ((image + abs(image.min())) / (image + abs(image.min())).max()).copy()
    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    
    for i in range(21):

        cv2.circle(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])), 2, [0, 1, 0],
                    thickness=-1)
        if i != 0:
            cv2.line(image, (int(gt_2d_joint[num][i][0]), int(gt_2d_joint[num][i][1])),
                        (int(gt_2d_joint[num][parents[i]][0]), int(gt_2d_joint[num][parents[i]][1])),
                        [0, 0, 1], 1)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('gt_image')
    ax1.axis("off")

