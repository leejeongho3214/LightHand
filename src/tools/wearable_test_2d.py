
import os
import sys
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.bar import colored
from torch.utils import data
from src.utils.argparser import load_model,pred_test, parse_args, pred_store_test
from dataset import *

def main(args):
    args.test = True
    _, test_dataset = build_dataset(args)
    testset_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model_path = "final_model"
    model_list = list()
    for (root, _, files) in os.walk(model_path):
        for file in files:
            if '.bin' in file:
                model_list.append(os.path.join(root, file))
    pck_list = list()            
    pbar = tqdm(total = len(model_list) * (len(testset_loader) + 14))
    for path_name in model_list:
        args.model = path_name.split('/')[1]
        args.name = ('/').join(path_name.split('/')[1:-2])
        args.output_dir = path_name
        _model, _, _, _, _ = load_model(args)
        state_dict = torch.load(path_name)
        _model.load_state_dict(state_dict['model_state_dict'], strict=False)

        pred_store_test(args, testset_loader, _model, pbar)  
        T_list = [0, 30] ## this mean mm as a threshold
        pck, epe, pbar = pred_test(args, T_list, pbar, "mm")
        pck_list.append([pck, epe, args.name])

    pbar.close()
    
    f = open(f"pck_test.txt", "w")
    for auc, epe, name in pck_list:
        f.write("{};{:.2f};{:.2f}\n".format(name, auc, epe / 3.7795275591))     ## category, model_name, auc
    f.close()
    print(colored("Writting ===> %s" % os.path.join(os.getcwd(), f"pck_test.txt")))
    

if __name__ == "__main__":
    args= parse_args()
    main(args)