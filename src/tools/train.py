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
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   ## If speed up, above 2 items turn into opposite
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_dataset, val_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )


    _model, best_pck, epo, count, writer, _, optimizer_state = load_model(args)
    batch_time = AverageMeter()
    d_type = "3D" if args.D3 else "2D"
    
    optimizer = torch.optim.Adam(
            params=list(_model.parameters()),
            lr=args.lr
        )
    
    if optimizer_state: optimizer.load_state_dict(optimizer_state)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [20, 40, 50], 0.1,
        last_epoch = -1
        )

    for epoch in range(epo, args.epoch):
        if epoch == epo:
            args.logger.debug(
                f"Path: {args.output_dir} | Dataset_len: {len(train_dataset)} | Type: {d_type} | Dataset: {args.dataset} | Model: {args.model} | Status: {args.reset} | Max_count : {args.count} | Max_epoch : {args.epoch}"
            )
            print(
                colored(
                    f"Path: {args.output_dir} | Dataset_len: {len(train_dataset)} | Type: {d_type} | Dataset: {args.dataset} | Model: {args.model} | Status: {args.reset} | Max_count : {args.count} | Max_epoch : {args.epoch}",
                    "yellow",
                )
            )
            
        lr_scheduler.step()
            
        trained_model, optimizer, batch_time, runner = train(
            args,
            trainset_loader,
            _model,
            epoch,
            len(train_dataset),
            args.logger,
            count,
            writer,
            len(trainset_loader) + len(valset_loader),
            batch_time,
            optimizer
        )
        _, count, pck, batch_time = valid(
            valset_loader,
            trained_model,
            best_pck,
            runner
        )

        is_best = best_pck < pck
        best_pck = max(pck, best_pck)

        if is_best:
            count = 0
            _model = trained_model
            save_checkpoint(
                trained_model,
                args,
                epoch,
                optimizer,
                best_pck,
                count,
                "good",
                logger=args.logger,
            )
            del trained_model

        else:
            count += 1
            if count == args.count:
                break
            
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
