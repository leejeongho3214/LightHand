
import os
import sys
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.bar import colored
from torch.utils import data
from src.utils.argparser import load_model,pred_test, parse_args
from dataset import *

def main(args):
    args.test = True
    _, test_dataset = build_dataset(args)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model_path = "final_model"
    model_list, eval_list = list(), list()
    for (root, _, files) in os.walk(model_path):
        for file in files:
            if '.bin' in file:
                model_list.append(os.path.join(root, file))
                
    pbar = tqdm(total = len(model_list) * (len(testset_loader) + 14))
    for path_name in model_list:
        args.model = path_name.split('/')[1]
        args.name = ('/').join(path_name.split('/')[1:-2])
        args.output_dir = os.path.join(model_path, args.name)
        _model, _, _, _, _ = load_model(args)
        state_dict = torch.load(path_name)
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)
        T_list = [0.05, 0.1, 0.15, 0.2]
        threshold_type = 'proportion'
        pck_list, pbar = pred_test(args, T_list, threshold_type, pbar, testset_loader, _model)
        eval_list.append([args.name, pck_list])
    pbar.close()
    
    f = open(f"pck_test.txt", "w")
    for each_list in eval_list:
        pck_list, list_name = each_list[1], each_list[0]
        f.write("{};".format(list_name))
        f.write("{:.2f},".format(pck_list["0.05"])); f.write("{:.2f},".format(pck_list["0.1"])); f.write("{:.2f},".format(pck_list["0.15"])); f.write("{:.2f},".format(pck_list["0.2"]))
        f.write("{:.2f}, {:.2f}\n".format(pck_list["epe"], pck_list["total"]))
    f.close()
    print(colored("Writting ===> %s" % os.path.join(os.getcwd(), f"pck_test.txt")))
    

if __name__ == "__main__":
    args= parse_args()
    main(args)