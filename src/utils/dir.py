# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import shutil
import torch
import os
import sys

def dump(pred_out_path, xyz_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    # xyz_pred_list = [x if type(x) == tuple else x.tolist() for x in xyz_pred_list]
    # if not os.path.isdir(pred_out_path): os.makedirs(pred_out_path)
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [ xyz_pred_list
            ], fo)
        
def reset_file(txt_name):
    if os.path.isfile(txt_name):
        os.remove(txt_name)

def reset_folder(name):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name)
        

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
def resume_checkpoint(_model, path):
    
    state_dict = torch.load(path)
    best_loss = state_dict['best_loss']
    epoch = state_dict['epoch'] + 1
    count = state_dict['count']
    _model.load_state_dict(state_dict['model_state_dict'], strict=False)
    del state_dict
    
    return best_loss, epoch, _model, count