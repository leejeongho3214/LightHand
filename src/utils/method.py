import torch
from src.utils.loss import adjust_learning_rate, JointsMSELoss
import numpy as np
import time
from time import ctime
from src.utils.visualize import *
from src.utils.metric_logger import AverageMeter
from src.utils.loss import *
from src.utils.bar import *


class Runner_t(object):
    def __init__(
        self,
        args,
        model,
        epoch,
        train_loader,
        phase,
        batch_time,
        logger,
        data_len,
        len_total,
        count,
        writer,
        optimizer
    ):
        super(Runner_t, self).__init__()
        self.args = args
        self.logger = logger
        self.len_data = data_len
        self.len_total = len_total
        self.count = count
        self.phase = phase
        self.writer = writer
        self.train_loader = train_loader
        self.batch_time = batch_time
        self.now_loader = train_loader
        self.bar = Bar(
            colored(str(epoch) + "_" + phase, color="blue"), max=len(self.now_loader)
        )
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.criterion_keypoints = torch.nn.MSELoss(reduction="none").cuda(args.device)
        self.log_losses = AverageMeter()
        self.pck_losses = AverageMeter()
        self.epe_losses = AverageMeter()
        self.criterion = JointsMSELoss(use_target_weight=False).cuda()

    def train_log(self, iteration, eta_seconds, end):
        tt = " ".join(ctime(eta_seconds + end).split(" ")[1:-1])
        if iteration % (self.args.logging_steps * 5) == 0:
            self.logger.debug(
                " ".join(
                    [
                        "dataset_length: {len}",
                        "epoch: {ep}",
                        "iter: {iter}",
                        "/{maxi}, count: {count}/{max_count}",
                        "lr: {lr:.6f}"
                    ]
                ).format(
                    len=self.len_data,
                    ep=self.epoch,
                    iter=iteration,
                    maxi=len(self.now_loader),
                    count=self.count,
                    max_count=self.args.count,
                    lr = self.optimizer.param_groups[0]['lr']
                )
                + " loss: {:.8f}".format(
                    self.log_losses.avg,
                )
            )

        if iteration == len(self.now_loader) - 1:
            self.bar.suffix = (
                "({iteration}/{data_loader}) "
                "name: {name} | "
                "count: {count} | "
                "loss: {total:.6f} \r"
            ).format(
                name=self.args.name.split("/")[-1],
                count=self.count,
                iteration=iteration,
                exp=tt,
                data_loader=len(self.now_loader),
                total=self.log_losses.avg,
            )
        else:
            self.bar.suffix = (
                "({iteration}/{data_loader}) "
                "name: {name} | "
                "count: {count} | "
                "loss: {total:.6f} |"
                "lr: {lr:.6f}"
            ).format(
                name=self.args.name.split("/")[-1],
                count=self.count,
                iteration=iteration,
                exp=tt,
                data_loader=len(self.now_loader),
                total=self.log_losses.avg,
                lr = self.optimizer.param_groups[0]['lr']
            )
        self.bar.next()

    def test_log(self, iteration, eta_seconds, end):
        tt = " ".join(ctime(eta_seconds + end).split(" ")[1:-1])

        if iteration == len(self.now_loader) - 1:
            self.bar.suffix = (
                "({iteration}/{data_loader}) "
                "count: {count} | "
                "loss: {loss:.6f} | "
                "best_loss: {best_loss:.6f}\n"
            ).format(
                name=self.args.name.split("/")[-1],
                count=self.count,
                iteration=iteration,
                best_loss=self.best_loss,
                data_loader=len(self.now_loader),
                loss=self.log_losses.avg,
            )
            self.logger.debug(
                " ".join(["Test =>> epoch: {ep}", "iter: {iter}", "/{maxi}"]).format(
                    ep=self.epoch, iter=iteration, maxi=len(self.now_loader)
                )
                + " epe: {:.2f}mm, count: {} / {}, total_pck: {:.2f} %, best_loss: {:.7f} , expected_date: {}".format(
                    self.epe_losses.avg * 0.26,
                    int(self.count),
                    self.args.count,
                    self.pck_losses.avg * 100,
                    self.best_loss,
                    tt,
                )
            )

        else:
            self.bar.suffix = (
                "({iteration}/{data_loader}) "
                "count: {count} | "
                "loss: {loss:.6f} | "
                "best_loss: {best_loss:.6f}"
            ).format(
                name=self.args.name.split("/")[-1],
                count=self.count,
                iteration=iteration,
                best_loss=self.best_loss,
                data_loader=len(self.now_loader),
                loss=self.log_losses.avg,
            )
        self.bar.next()

    def run(self, end):
        multiply = 4
        if self.phase == "TRAIN":
            self.model.train()
            for iteration, (images, gt_2d_joints, gt_heatmaps) in enumerate(
                self.train_loader
            ):
                batch_size = images.size(0)
                # adjust_learning_rate(self.optimizer, self.epoch, self.args)
                images = images.cuda()
                gt_heatmaps = gt_heatmaps.cuda()
                pred = self.model(images)

                loss = self.criterion(pred, gt_heatmaps, None)

                self.log_losses.update(loss.item(), batch_size)
                pred = np.array(pred.detach().cpu())
                pred_joint, _ = get_max_preds(
                    pred
                )  ## get the joint location from heatmap
                pred_joint = (
                    pred_joint * multiply
                )  ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256

                # back prop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (
                    iteration == 0
                    or iteration == int(len(self.train_loader) / 2)
                    or iteration == len(self.train_loader) - 1
                ):
                    fig = plt.figure()
                    visualize_gt(images, gt_2d_joints, fig, iteration)
                    visualize_pred(
                        images,
                        pred_joint,
                        fig,
                        "train",
                        self.epoch,
                        iteration,
                        self.args,
                        None,
                    )
                    plt.close()
                

                self.batch_time.update(time.time() - end)
                end = time.time()
                eta_seconds = self.batch_time.avg * (
                    (self.len_total - iteration)
                    + (self.args.epoch - self.epoch - 1) * self.len_total
                )

                self.train_log(iteration, eta_seconds, end)
                
            self.writer.add_scalar("Loss/train", self.log_losses.avg, self.epoch)
            
            return self.model, self.optimizer, self.batch_time

        else:
            self.model.eval()
            with torch.no_grad():
                for iteration, (images, gt_2d_joints, gt_heatmaps) in enumerate(
                    self.now_loader
                ):
                    batch_size = images.size(0)
                    images = images.cuda()
                    gt_2d_joint = gt_2d_joints.cuda()
                    gt_heatmaps = gt_heatmaps.cuda()

                    pred = self.model(images)
                    loss = self.criterion(pred, gt_heatmaps, None)
                    
                    pred = np.array(pred.detach().cpu())
                    pred_joint, _ = get_max_preds(
                        pred
                    )  ## get the joint location from heatmap
                    pred_joint = (
                        pred_joint * multiply
                    )  ## heatmap resolution was 64 x 64 so multiply 4 to make it 256 x 256
                    pred_joint = torch.tensor(pred_joint).cuda()

                    self.log_losses.update(loss.item(), batch_size)

                    pck = PCK_2d_loss(
                        pred_joint, gt_2d_joint, T=0.2, threshold="proportion"
                    )
                    epe_loss, _ = EPE_train(
                        pred_joint, gt_2d_joint
                    )  ## consider invisible joint
                    self.pck_losses.update(pck, batch_size)
                    self.epe_losses.update_p(epe_loss[0], epe_loss[1])

                    if (
                        iteration == 0
                        or iteration == int(len(self.now_loader) / 2)
                        or iteration == len(self.now_loader) - 1
                    ):
                        fig = plt.figure()
                        visualize_gt(images, gt_2d_joints, fig, iteration)
                        visualize_pred(
                            images,
                            pred_joint,
                            fig,
                            "val",
                            self.epoch,
                            iteration,
                            self.args,
                            None,
                        )
                        plt.close()

                    self.batch_time.update(time.time() - end)
                    end = time.time()
                    eta_seconds = self.batch_time.avg * (
                        (len(self.now_loader) - iteration)
                        + (self.args.epoch - self.epoch - 1) * self.len_total
                    )

                    self.test_log(iteration, eta_seconds, end)
                    
                self.writer.add_scalar("Loss/valid", self.log_losses.avg, self.epoch)
                
                return (
                    self.log_losses.avg,
                    self.count,
                    self.pck_losses.avg * 100,
                    self.batch_time,
                )


class Runner_v(Runner_t):
    def __init__(
        self,
        train_runner,
        model,
        test_dataloader,
        phase,
        best_loss,
    ):
        self.__dict__ = train_runner.__dict__.copy()
        self.now_loader = test_dataloader
        self.best_loss = best_loss
        self.model = model
        self.phase = phase
        self.bar = Bar(
            colored(str(self.epoch) + "_" + phase, color="blue"), max=len(self.now_loader)
        )
        self.log_losses.reset()
        self.pck_losses.reset()
        self.epe_losses.reset()