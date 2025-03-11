import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.bar import colored
from torch.utils import data
from src.utils.argparser import load_model, pred_store, pred_eval, parse_args
from src.tools.dataset import *

def main():
    args.eval = True
    _, eval_dataset = build_dataset(args)
    testset_loader = data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print(len(eval_dataset))
    model_path = "output/simplebaseline/ours"
    model_list= list()
    for (root, _, files) in os.walk(model_path):
        for file in files:
            if '.bin' in file:
                model_list.append(os.path.join(root, file))
                
    
    aa = [['pckb', [0.1, 0.3]], ['mm', [0, 30]], ['mm', [0, 50]]]
    for t_type, T_list in aa:            
        pbar = tqdm(total = len(model_list) * (len(testset_loader) + 4))           ## 4 means a number of category in pred_eval
        pck_list = list()
        for path_name in model_list:
            args.model = path_name.split('/')[1]
            args.name = ('/').join(path_name.split('/')[1:-2])
            args.output_dir = path_name
            _model, _, _, _, _, msg = load_model(args)
            state_dict = torch.load(path_name)
            _model.load_state_dict(state_dict['model_state_dict'], strict=False)
                
            pred_store(args, testset_loader, _model, pbar)        
            # T_list = [0, 30] ## this mean mm as a threshold
            # t_type = "mm"
            pck, pbar = pred_eval(args, T_list, pbar, t_type)
            pck_list.append([pck, args.name])
            if args.model == 'mediapipe':
                break
        pbar.close()

        file_name = f"{args.model}_pck_eval_{t_type}_{T_list[1]}.txt"
        f = open(file_name, "w")
        for total_pck, name in pck_list:
            for p_type in total_pck:
                f.write("{};{};{:.2f};{:.2f};".format(p_type, name, total_pck[p_type][0], total_pck[p_type][1]))   ## category, model_name, auc, epe
                for idx, pck in enumerate(total_pck[p_type][2]):
                    f.write("{:.2f};".format(pck))
                    if idx == len(total_pck[p_type][2])-1: f.write("\n".format(pck))
        f.close()
        print(colored("Writting ===> %s" % os.path.join(os.getcwd(), file_name)))
    

if __name__ == "__main__":
    args= parse_args()
    main()