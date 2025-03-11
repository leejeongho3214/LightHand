import gc
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torch.utils import data
from src.utils.argparser import parse_args, load_model, train, valid
from dataset import *
from src.utils.bar import colored

def main(args):
    
    random_seed = 9001
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_dataset, val_dataset = build_dataset(args)
    
    trainset_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valset_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    _model, best_loss, epo, count, writer, _ = load_model(args)
    pck_l = 0; batch_time = AverageMeter()
    d_type = "3D" if args.D3 else "2D"
    
    for epoch in range(epo, args.epoch):
        if epoch == epo: 
            args.logger.debug( f"Path: {args.output_dir} | Dataset_len: {len(train_dataset)} | Type: {d_type} | Dataset: {args.dataset} | Model: {args.model} | Status: {args.reset} | Max_count : {args.count} | Max_epoch : {args.epoch}")
            print(colored(f"Path: {args.output_dir} | Dataset_len: {len(train_dataset)} | Type: {d_type} | Dataset: {args.dataset} | Model: {args.model} | Status: {args.reset} | Max_count : {args.count} | Max_epoch : {args.epoch}", "yellow"))
        Graphormer_model, optimizer, batch_time, best_loss = train(args, trainset_loader, valset_loader, _model, epoch, best_loss, len(train_dataset),args.logger, count, writer, pck_l, len(trainset_loader)+len(valset_loader), batch_time)
        loss, count, pck, batch_time = valid(args, trainset_loader, valset_loader, Graphormer_model, epoch, count, best_loss, len(train_dataset), args.logger, writer, batch_time, len(trainset_loader)+len(valset_loader), pck_l)
        
        pck_l = max(pck, pck_l)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        if is_best:
            count = 0
            _model = Graphormer_model
            save_checkpoint(Graphormer_model, args, epoch, optimizer, best_loss, count,  'good',logger= args.logger)
            del Graphormer_model

        else:
            count += 1
            if count == args.count:
                break
        gc.collect()
        torch.cuda.empty_cache()
  
 
if __name__ == "__main__":
    args= parse_args()
    main(args)
    
    
    

    