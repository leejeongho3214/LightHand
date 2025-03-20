import json
import os
import sys
import argparse

import yaml

from src.utils.bar import colored
from src.utils.pre_argparser import pre_arg
from src.modeling.simplebaseline.config import config as config_simple
from src.modeling.simplebaseline.pose_resnet import get_pose_net
from src.modeling.hrnet.config import cfg, update_config
from src.modeling.hrnet.pose_hrnet import get_hrnet
import torch
import time
from src.utils.dir import reset_folder
from torch.utils.tensorboard import SummaryWriter
from src.utils.method import Runner_t, Runner_v
from src.utils.dir import resume_checkpoint, dump
import numpy as np
from matplotlib import pyplot as plt
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.visualize import *


def parse_args(phase="train"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="hrnet/ours",
        help="You write down to store the directory path",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="84k",
        help="You write down to store the directory path",
        type=str,
    )
    parser.add_argument(
        "--root_path",
        default=f"output",
        type=str,
        required=False,
        help="The root directory to save location which you want",
    )
    parser.add_argument("--model", default="ours", type=str, required=False)
    parser.add_argument("--dataset", default="ours", type=str, required=False)
    parser.add_argument("--view", default="wrist", type=str, required=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--count", default=30, type=int)
    parser.add_argument(
        "--num_our",
        default=84000,
        type=int,
        help="Our dataset have 420k imaegs so you can use train data as many as you want, according to this ratio",
    )
    parser.add_argument("--ratio_of_other", default=0, type=float)
    parser.add_argument(
        "--ratio_of_aug",
        default=0.6,
        type=float,
        help="You can use color jitter to train data as many as you want, according to this ratio",
    )
    parser.add_argument("--epoch", default=50, type=int)

    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--plt", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--rot", action="store_true")
    parser.add_argument(
        "--color",
        action="store_true",
        help="If you write down, This dataset would be applied color jitter to train data, according to ratio of aug",
    )
    parser.add_argument(
        "--D3",
        action="store_true",
        help="If you write down, The output of model would be 3d joint coordinate",
    )

    args = parser.parse_args()
    args.phase = phase
    args.name = os.path.join(args.root, args.name)
    args, logger = pre_arg(args)
    args.logger = logger

    return args


def load_model(args):
    epoch = 0
    best_pck = 0
    count = 0
    optimizer_state = 0
    args.device = torch.device(args.device)

    if args.model == "hrnet":
        with open("src/modeling/hrnet/config/cfg.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        _model = get_hrnet(cfg, is_train=True)

    else:
        _model = get_pose_net(config_simple, is_train=True)

    log_dir = f"tensorboard/{args.name}"
    writer = SummaryWriter(log_dir)

    if args.reset:
        if os.path.isfile(
            os.path.join(
                args.root_path, args.name, "checkpoint-good/state_dict.bin"
            )
        ):
            if (
                True
                if input("There is resume_point but do you want to delete?")
                in ["o", "y", "yes"]
                else False
            ):
                reset_folder(log_dir)
                reset_folder(os.path.join(args.root_path, args.name))

                print(colored("Ignore the check-point model", "green"))
                msg = "resume but init"
        else:
            msg = "init"
    else:
        if os.path.isfile(
            os.path.join(
                args.root_path, args.name, "checkpoint-good/state_dict.bin"
            )
        ):
            best_pck, epoch, _model, count, optimizer_state = resume_checkpoint(
                _model,
                os.path.join(
                    args.root_path, args.name, "checkpoint-good/state_dict.bin"
                ),
            )
            args.logger.debug(
                "Loading ===> %s" % os.path.join(args.root_path, args.name)
            )
            print(
                colored(
                    "Loading ===> %s" % os.path.join(args.root_path, args.name),
                    "green",
                )
            )
            msg = "resume"
        else:
            reset_folder(log_dir)
            msg = "init"

    _model.to(args.device)

    return _model, best_pck, epoch, count, writer, msg, optimizer_state


def train(
    args,
    train_dataloader,
    _model,
    epoch,
    data_len,
    logger,
    count,
    writer,
    len_total,
    batch_time,
    optimizer
):
    end = time.time()
    runner = Runner_t(
        args,
        _model,
        epoch,
        train_dataloader,
        "TRAIN",
        batch_time,
        logger,
        data_len,
        len_total,
        count,
        writer,
        optimizer
    )
    trained_model, optimizer, batch_time = runner.run(end)

    return trained_model, optimizer, batch_time, runner


def valid(
    test_dataloader,
    trained_model,
    best_pck,
    train_runner
):
    end = time.time()
    runner = Runner_v(
        train_runner,
        trained_model,
        test_dataloader,
        "VALID",
        best_pck,
    )
    loss, count, pck, batch_time = runner.run(end)

    return loss, count, pck, batch_time


def pred_store(args, dataloader, model, pbar):
    meta = {
        "Standard": {"bb": [], "pred": [], "gt": []},
        "Occlusion_by_Pinky": {"bb": [], "pred": [], "gt": []},
        "Occlusion_by_Thumb": {"bb": [], "pred": [], "gt": []},
        "Occlusion_by_Both": {"bb": [], "pred": [], "gt": []},
    }

    with torch.no_grad():
        for images, gt_2d_joints, annos in dataloader:
            anno, idx = annos
            bbox_size = list()
            pred_2d_joints = model(images.cuda())
            pred_2d_joints = np.array(pred_2d_joints.detach().cpu())

            # get the joint location from heatmap
            pred_joint, _ = get_max_preds(pred_2d_joints)

            # heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
            pred_joint = pred_joint * 4
            pred_joint = torch.tensor(pred_joint)

            for j in gt_2d_joints:
                width = max(j[:, 0]) - min(j[:, 0])
                height = max(j[:, 1]) - min(j[:, 1])
                length = np.sqrt(width**2 + height**2)
                bbox_size.append(length.item())

            for idx, name in enumerate(anno):
                meta[name]["bb"].append(bbox_size[idx])
                meta[name]["pred"].append(pred_joint[idx].tolist())
                meta[name]["gt"].append(gt_2d_joints[idx].tolist())

            pbar.update(1)

        dump(os.path.join("output", args.name, "evaluation.json"), meta)


def pred_store_test(args, dataloader, model, pbar):

    # if os.path.isfile(os.path.join("final_model", args.name, "test.json")):
    #     pbar.update(len(dataloader))
    #     return

    meta = {"pred": [], "gt": [], "bb": []}

    with torch.no_grad():
        for images, gt_2d_joints in dataloader:
            bbox_size = list()
            images = images.cuda()
            pred_2d_joints = model(images)
            pred_2d_joints = np.array(pred_2d_joints.detach().cpu())
            # get the joint location from heatmap
            pred_joint, _ = get_max_preds(pred_2d_joints)
            # heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
            pred_joint = pred_joint * 4
            pred_joint = torch.tensor(pred_joint)

            if args.plt:
                for i in range(images.size(0)):
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joints, fig, 0)
                    visualize_pred(images, pred_joint, fig, "evaluation", 0, i, args)
                    plt.close()

            for j in gt_2d_joints:
                width = max(j[:, 0]) - min(j[:, 0])
                height = max(j[:, 1]) - min(j[:, 1])
                length = np.sqrt(width**2 + height**2)
                bbox_size.append(length.item())

            meta["pred"].append(pred_joint.tolist())
            meta["gt"].append(gt_2d_joints.tolist())
            meta["bb"].append(bbox_size)

            pbar.update(1)

        dump(os.path.join("final_model", args.name, "test.json"), meta)


def pred_eval(args, T_list, p_bar, method):

    pck_list = dict()

    with open(os.path.join("output", args.name, "evaluation.json"), "r") as fi:
        meta = json.load(fi)

    meta = meta[0]
    if method == "mm":
        # change pixel coordinate to mm coordinate
        thresholds_list = np.linspace(T_list[0], T_list[-1], 101)[1:] * 3.7795275591
    elif method == "pckb":
        thresholds_list = np.linspace(T_list[0], T_list[-1], 100)
    else:
        assert 0, "this method is the wrong"

    thresholds = np.array(thresholds_list)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
    total_pck = torch.empty(0)
    total_epe = np.zeros([971, 21])

    for p_type in meta:
        # each bounding box's dianogal length
        bbox = np.array(meta[p_type]["bb"])
        pred = np.array(meta[p_type]["pred"])
        gt = np.array(meta[p_type]["gt"])  # batch_siez x 21 x 2

        diff = np.sqrt(
            np.sum(np.square(gt[:, :, :2] - pred[:, :, :2]), axis=-1)
        )  # 900 x 21
        if method == "pckb":
            norm_diff = diff / bbox[:, None].repeat(21, axis=1)
            norm_diff = norm_diff[:, :, None]
        else:
            norm_diff = diff[:, :, None]

        norm_diff = torch.concat(
            [torch.tensor(norm_diff), torch.tensor(gt[:, :, -1][:, :, None])], dim=-1
        )
        norm_diff = norm_diff[norm_diff[:, :, 1] == 1][:, 0]

        total_epe = np.concatenate([total_epe, diff], axis=0)
        total_pck = torch.concat([norm_diff, total_pck])
        total = len(norm_diff)
        # Compute a PCK for each threshold with 100 values
        pck_t = [(len(norm_diff[norm_diff < T]) / total) * 100 for T in thresholds_list]
        pck_t = np.array(pck_t)

        auc = np.trapz(pck_t, thresholds)
        auc /= norm_factor + sys.float_info.epsilon

        pck_list["%s" % p_type] = [auc, diff.mean().item() / 3.7795275591, pck_t]
        p_bar.update(1)

    total = len(total_pck)
    # Compute a PCK for each threshold with 100 values
    pck_t = [(len(total_pck[total_pck < T]) / total) * 100 for T in thresholds_list]
    pck_t = np.array(pck_t)
    auc = np.trapz(pck_t, thresholds)
    auc /= norm_factor + sys.float_info.epsilon
    pck_list["mean_auc"] = [auc, total_epe.mean() / 3.7795275591, pck_t]

    return pck_list, p_bar


def pred_test(args, T_list, pbar, method):

    with open(os.path.join("final_model", args.name, "test.json"), "r") as fi:
        meta = json.load(fi)

    meta = meta[0]
    if method == "mm":
        # change pixel coordinate to mm coordinate
        thresholds_list = np.linspace(T_list[0], T_list[-1], 101)[1:] * 3.7795275591
    elif method == "pckb":
        thresholds_list = np.linspace(T_list[0], T_list[-1], 100)
    else:
        assert 0, "this method is the wrong"
    thresholds = np.array(thresholds_list)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

    bbox = np.array(meta["bb"])  # each bounding box's dianogal length
    pred = np.array(meta["pred"])
    gt = np.array(meta["gt"])  # batch_siez x 21 x 2

    bbox = np.array(
        [np.array(bbox[i][j]) for i in range(len(bbox)) for j in range(len(bbox[i]))]
    )
    gt = np.array(
        [np.array(gt[i][j]) for i in range(len(gt)) for j in range(len(gt[i]))]
    )
    pred = np.array(
        [np.array(pred[i][j]) for i in range(len(pred)) for j in range(len(pred[i]))]
    )

    diff = np.sqrt(np.sum(np.square(gt - pred), axis=-1))  # batch x 21
    if method == "pckb":
        norm_diff = diff / bbox[:, None].repeat(21, axis=1)
    else:
        norm_diff = diff
    norm_diff = norm_diff.flatten()

    total = len(norm_diff)

    # calculate a pck according to each threshold that it has 100 values
    pck_t = [(len(norm_diff[norm_diff < T]) / total) * 100 for T in thresholds_list]
    pck_t = np.array(pck_t)

    auc = np.trapz(pck_t, thresholds)
    auc /= norm_factor + sys.float_info.epsilon
    pbar.update(1)

    return auc, diff.mean(), pbar
