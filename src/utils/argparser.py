import json
import os
import sys
import argparse

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
from src.utils.method import Runner
from src.utils.dir import  resume_checkpoint, dump
import numpy as np
from matplotlib import pyplot as plt
from src.utils.loss import *
from src.utils.geometric_layers import *
from src.utils.visualize import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("name", default='None',
                        help = 'You write down to store the directory path',type=str)
    parser.add_argument("--root_path", default=f'output', type=str, required=False,
                        help="The root directory to save location which you want")
    parser.add_argument("--model", default='ours', type=str, required=False)
    parser.add_argument("--dataset", default='ours', type=str, required=False)
    parser.add_argument("--view", default='wrist', type=str, required=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--ratio_of_our", default=0.3, type=float,
                        help="Our dataset have 420k imaegs so you can use train data as many as you want, according to this ratio")
    parser.add_argument("--ratio_of_other", default=0.3, type=float)
    parser.add_argument("--ratio_of_aug", default=0.2, type=float,
                        help="You can use color jitter to train data as many as you want, according to this ratio")
    parser.add_argument("--epoch", default=50, type=int)
    
    parser.add_argument("--loss_2d", default=0, type=float)
    parser.add_argument("--loss_3d", default=1, type=float)
    parser.add_argument("--loss_3d_mid", default=0, type=float)
    parser.add_argument("--scale", action='store_true')
    parser.add_argument("--plt", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--logger", action='store_true')
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--rot", action='store_true')
    parser.add_argument("--color", action='store_true',
                        help="If you write down, This dataset would be applied color jitter to train data, according to ratio of aug")
    parser.add_argument("--D3", action='store_true',
                        help="If you write down, The output of model would be 3d joint coordinate")
    
    args = parser.parse_args()
    args, logger = pre_arg(args)
    args.logger = logger
    
    return args


def load_model(args):
    epoch = 0
    best_loss = np.inf
    count = 0
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.device = torch.device(args.device)

    if args.model == "hrnet":   
        update_config(cfg, args)
        _model = get_hrnet(cfg, is_train=True)
        
    elif args.model == 'simplebaseline': 
        _model = get_pose_net(config_simple, is_train=True)
        
        
    log_dir = f'tensorboard/{args.name}'
    if args.name.split("/")[0] != "final_model":
        if args.reset: reset_folder(log_dir); reset_folder(os.path.join(args.root_path, args.name)); args.reset = "Init"
        else: args.reset = "No Init"
        
    writer = SummaryWriter(log_dir)
    if os.path.isfile(os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin')):
        best_loss, epoch, _model, count = resume_checkpoint(_model, os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin'))
        args.logger.debug("Loading ===> %s" % os.path.join(args.root_path, args.name))
        print(colored("Loading ===> %s" % os.path.join(args.root_path, args.name), "green"))
        
    
    _model.to(args.device)
    
    return _model, best_loss, epoch, count, writer



def train(args, train_dataloader, test_dataloader, Graphormer_model, epoch, best_loss, data_len ,logger, count, writer, pck, len_total, batch_time):
    end = time.time()
    phase = 'TRAIN'
    runner = Runner(args, Graphormer_model, epoch, train_dataloader, test_dataloader, phase, batch_time, logger, data_len, len_total, count, pck, best_loss, writer)
    Graphormer_model, optimizer, batch_time= runner.other(end)
        
    return Graphormer_model, optimizer, batch_time, best_loss

def valid(args, train_dataloader, test_dataloader, Graphormer_model, epoch, count, best_loss,  data_len ,logger, writer, batch_time, len_total, pck):
    end = time.time()
    phase = 'VALID'
    runner = Runner(args, Graphormer_model, epoch, train_dataloader, test_dataloader, phase, batch_time, logger, data_len, len_total, count, pck, best_loss, writer)
    loss, count, pck, batch_time = runner.other(end)
       
    return loss, count, pck, batch_time

def pred_store(args, dataloader, model, pbar):
    
    if os.path.isfile(os.path.join(args.output_dir, "pred.json")):
        return
    
    xy_list, p_list, gt_list = [], [], []
    with torch.no_grad():
        for (images, gt_2d_joints, _, anno) in dataloader:
            images = images.cuda()
            gt_2d_joint = gt_2d_joints.cuda()
            pred_2d_joints = model(images)
            pred_2d_joints[:, :, 1] = pred_2d_joints[:, :, 1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:, :, 0] = pred_2d_joints[:, :, 0] * images.size(3)
            if args.plt:
                for i in range(images.size(0)):
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig, 0)
                    visualize_pred(images, pred_2d_joints, fig, 'evaluation', 0, i, args, anno)
                    plt.close()
            gt_list.append(gt_2d_joints.tolist())
            xy_list.append(pred_2d_joints.tolist())
            p_list.append(anno)
            pbar.update(1) 
            
    dump(os.path.join("output", "gt_test.json"), gt_list)
    dump(os.path.join(args.output_dir, "pred.json"), xy_list)
    # dump(os.path.join(args.output_dir, "pred_p.json"), p_list)

    
def pred_eval(args, T_list, Threshold_type):

    gt_path = os.path.join("output/gt.json")
    pred_path = os.path.join(args.output_dir, "pred.json")
    pose_path = os.path.join("output/pred_p.json")
    
    with open(gt_path, 'r') as fi:
        gt_json = json.load(fi)
    with open(pred_path, 'r') as fi:
        pred_json = json.load(fi)
    with open(pose_path, 'r') as fi:
        pose_json = json.load(fi)
        
    pred = [x for i in range(len(pred_json[0])) for x in pred_json[0][i]]
    pose = [x for i in range(len(pose_json[0])) for x in pose_json[0][i]]
    gt = [x for i in range(len(gt_json[0])) for x in gt_json[0][i]]
    thresholds_list = np.linspace(T_list[0], T_list[-1], 100)
    thresholds = np.array(thresholds_list)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
    
    pck_list = {'Standard':{}, 'Occlusion_by_Pinky': {}, 'Occlusion_by_Thumb': {}, 'Occlusion_by_Both': {}}
    epe_list = {'Standard':[], 'Occlusion_by_Pinky': [], 'Occlusion_by_Thumb': [], 'Occlusion_by_Both': []}

    for p_type in pck_list: 
        pck_list[f'{p_type}']['total'] = []
        for T in T_list: 
            pck_list[f'{p_type}'][f'{T:.2f}'] = []
            
    for (pred_joint, p_type, gt_joint) in zip(pred, pose, gt):
        
        gt_joint = torch.tensor(gt_joint)[None, :]
        pred_joint = torch.tensor(pred_joint)[None, :]
        pck_t = list()
        for T in T_list:     
            pck = PCK_2d_loss_visible(pred_joint, gt_joint, T, Threshold_type)
            if T == T_list[0]:
                epe, _ = EPE(pred_joint, gt_joint)
                epe_list[f'{p_type}'].append((epe[0]/epe[1]) * 0.264583) ## pixel -> mm
            pck_list[f'{p_type}'][f'{T:.2f}'].append(pck)
        for th in thresholds_list:
            pck = PCK_2d_loss_visible(pred_joint, gt_joint, th, Threshold_type)
            pck_t.append(pck * 100)
        
        pck_t = np.array(pck_t)
        auc = np.trapz(pck_t, thresholds)
        auc /= (norm_factor + sys.float_info.epsilon)
        pck_list[f'{p_type}']['total'].append(auc)
    
    for j in pck_list:
        for i in pck_list[j]:
            pck_list[j][i] = np.array(pck_list[j][i]).mean()
    for i in epe_list:
        epe_list[i] = np.array(epe_list[i]).mean()
        
        
    return pck_list, epe_list


def pred_test(args, T_list, Threshold_type, pbar, dataloader, model):

    thresholds_list = np.linspace(T_list[0], T_list[-1], 100)
    thresholds = np.array(thresholds_list)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

    pck_list = {'0.05':[], "0.1":[], "0.15": [], "0.2": [], "epe": [], "total": []}
    with torch.no_grad():
        for (images, gt_2d_joints, _, anno) in dataloader:
            images = images.cuda()
            gt_2d_joint = gt_2d_joints.cuda()
            pred_2d_joints = model(images)
            pred_2d_joints[:, :, 1] = pred_2d_joints[:, :, 1] * images.size(2) ## You Have to check whether weight and height is correct dimenstion
            pred_2d_joints[:, :, 0] = pred_2d_joints[:, :, 0] * images.size(3)
            
            if args.plt:
                for i in range(images.size(0)):
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joint, fig,i)
                    visualize_pred(images, pred_2d_joints, fig, 'evaluation', 0, i, args, anno)     ##
                    plt.close()
            
            pck_l, auc_l= PCK_2d_loss_list(pred_2d_joints, gt_2d_joint, T_list, Threshold_type, thresholds)       ##
            epe, _ = EPE_train(pred_2d_joints, gt_2d_joint)
            for pck_p in pck_l:
                pck_list[f"{pck_p[0]}"].append(pck_p[1])
            pck_list["epe"].append((epe[0]/epe[1]) * 0.264583)
    
            auc_l = np.array(auc_l)
            auc = np.trapz(auc_l, thresholds)
            auc /= (norm_factor + sys.float_info.epsilon)
            pck_list["total"].append(auc)
            pbar.update(1)
        
        for j in pck_list:
            pck_list[j] = np.nanmean(np.array(pck_list[j]))
        
    return pck_list, pbar