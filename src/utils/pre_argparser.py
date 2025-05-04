
import os
from src.utils.comm import get_rank
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir
from src.utils.dir import reset_file

def pre_arg(args):
    args.output_dir = os.path.join(args.root_path, args.name)
    if args.reset or not os.path.isfile(os.path.join(args.root_path, args.name,'checkpoint-good/state_dict.bin')): reset_file(os.path.join(args.output_dir, "log.txt"))
    if (not args.output_dir.split('/')[1] == "output" and not os.path.isfile((args.output_dir))) or args.phase == "test":  mkdir(args.output_dir); logger = setup_logger(args.name, args.output_dir, get_rank())
    else: logger = None
    logger.debug(args)
    args.logging_steps = int(100)

    args.num_workers = int(8)
    args.train_yaml = str('../../dataset/freihand/train.yaml')
    args.val_yaml = str('../../dataset/freihand/test.yaml')
    args.device = str('cuda')
    
    return args, logger