
import os
import sys
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.bar import colored
from torch.utils import data
from src.utils.argparser import load_model, pred_store, pred_eval, parse_args
from src.tools.dataset import *

def main(args):
    args.eval = True
    _, eval_dataset = build_dataset(args)
    testset_loader = data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print(len(eval_dataset))
    model_path = "final_model/"
    model_list, pck_list = list(), list()
    for (root, _, files) in os.walk(model_path):
        for file in files:
            if '.bin' in file:
                model_list.append(os.path.join(root, file))
                
    pbar = tqdm(total = len(model_list) * (len(testset_loader) + 4))           ## 4 means a number of category in pred_eval
    for path_name in model_list:
        args.model = path_name.split('/')[1]
        args.name = ('/').join(path_name.split('/')[1:-2])
        args.output_dir = path_name
        _model, _, _, _, _ = load_model(args)
        state_dict = torch.load(path_name)
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)
        pred_store(args, testset_loader, _model, pbar)        
        T_list = [0, 30] ## this mean mm as a threshold
        pck, pbar = pred_eval(args, T_list, pbar, "mm")
        pck_list.append([pck, args.name])

    pbar.close()
    
    f = open(f"pck_eval.txt", "w")
    for total_pck, name in pck_list:
        for p_type in total_pck:
            f.write("{};{};{:.2f};{:.2f}\n".format(p_type, name, total_pck[p_type][0], total_pck[p_type][1]/3.7795275591))     ## category, model_name, auc
    f.close()
    print(colored("Writting ===> %s" % os.path.join(os.getcwd(), f"pck_eval.txt")))
    

if __name__ == "__main__":
    args= parse_args()
    main(args)