#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for training the model
"""

import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise import *
from models.utils import chamfer_distance_unit_sphere
from utils.logger import Logger

#DDP
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


__author__ = "Shitong Luo"
__license__ = "MIT"
__version__ = "1.1.0"


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)

    # Logging and other stuff
    if rank==0:
        logger = Logger("score-denoise")
        log_dir = get_new_log_dir(args.log_root, prefix='D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')
        task = None
        if args.clearml:
            from clearml import Task
            task = Task.init(project_name=args.project_name, task_name=args.task_name, output_uri=args.output_uri)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
        log_hyperparams(writer, log_dir, args)

    # Datasets and loaders
    if rank==0:
        logger.info('Loading datasets')
    train_dset = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=args.dataset_root,
                dataset=args.dataset,
                split='train',
                resolution=resl,
                transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
            ) for resl in args.resolutions
        ],
        patch_size=args.patch_size,
        patch_ratio=1.2,
        on_the_fly=True  
    )
    val_dset = PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='test',
            resolution=args.resolutions[0],
            transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False, scale_d=0),
        )
    train_iter = get_data_iterator(
        DataLoader(
            train_dset,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(train_dset)
        )
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(val_dset)
    )

    # Model
    if rank == 0:
        logger.info('Building model...')
    model = DenoiseNet(args).to(rank)
    if args.ckpt is not None:
        if rank == 0:
            logger.info(f'Loading pre-trained weights from {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location = f'cuda:{rank}')
        model.load_state_dict(ckpt['state_dict'])
    ddp_model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Train, validate and test   
    def train(it):
        # Load data
        batch = next(train_iter)
        pcl_noisy = batch['pcl_noisy'].to(rank)
        pcl_clean = batch['pcl_clean'].to(rank)

        # Reset grad and model state
        optimizer.zero_grad()
        ddp_model.train()

        # Forward
        if args.supervised:
            loss = ddp_model.module.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean)
        else:
            loss = ddp_model.module.get_selfsupervised_loss(pcl_noisy=pcl_noisy)

        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(ddp_model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        dist.barrier()
        loss = torch.Tensor([ loss ]).cuda()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        # Logging
        if rank==0:
            avg_loss = loss / world_size
            logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
                it, avg_loss.item(), orig_grad_norm,
            ))
            writer.add_scalar('train/loss', avg_loss, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()
        dist.barrier()


    def validate(it):

        ddp_model.eval()
        all_clean = []
        all_denoised = []

        for batch in tqdm(val_loader, desc="Validating..."):
            pcl_noisy_b = batch['pcl_noisy'].to(rank)
            pcl_clean_b = batch['pcl_clean'].to(rank)
            val_losses = []
            if args.supervised:
                val_loss = ddp_model.module.get_supervised_loss(pcl_noisy=pcl_noisy_b, pcl_clean=pcl_clean_b)
            else:
                val_loss = ddp_model.module.get_selfsupervised_loss(pcl_noisy=pcl_noisy_b)
            val_losses.append(val_loss)

            for i in range(len(pcl_noisy_b)):
                pcl_noisy = pcl_noisy_b[i]
                pcl_clean = pcl_clean_b[i]
                pcl_denoised = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)
                all_clean.append(pcl_clean.unsqueeze(0))
                all_denoised.append(pcl_denoised.unsqueeze(0))
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)
        dist.barrier()

        # Validation loss
        val_losses = torch.Tensor(val_losses).cuda()
        dist.all_reduce(val_losses, op=dist.ReduceOp.SUM)

        # Metric
        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        avg_chamfer = torch.Tensor([ avg_chamfer ]).cuda()
        dist.all_reduce(avg_chamfer, op=dist.ReduceOp.SUM)

        if rank==0:
            avg_chamfer = avg_chamfer / world_size
            avg_val_loss = val_losses / world_size
            logger.warning('[Val] Iter %04d | CD %.6f | Val loss %.6f ' % (it, avg_chamfer, avg_val_loss))
            writer.add_scalar('val/loss', avg_val_loss, it)
            writer.add_scalar('val/chamfer', avg_chamfer, it)
            writer.add_mesh('val/pcl', all_denoised[:args.val_num_visualize], global_step=it)
            writer.flush()
        dist.barrier()

        # scheduler.step(avg_chamfer)
        return avg_chamfer

    # Main loop
    if rank==0:
        logger.info('Start training...')
    try:
        for it in range(1, args.max_iters+1):
            train(it)
            if it % args.val_freq == 0 or it == args.max_iters:
                cd_loss = validate(it)
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }
                dist.barrier()
                if rank==0:
                    ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
                    logger.warning(f'Model saved at {os.path.join(ckpt_mgr.save_dir, ckpt_mgr.ckpts[-1])}')
                # ckpt_mgr.save(model, args, 0, opt_states, step=it)

    except KeyboardInterrupt:
        if rank==0:
            logger.error('Ctrl+C pressed. Terminating! Please wait a bit!')
    finally:
        destroy_process_group()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.020)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_devices', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
    ## Model architecture
    parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
    parser.add_argument('--frame_knn', type=int, default=32)
    parser.add_argument('--num_train_points', type=int, default=512)
    parser.add_argument('--num_clean_nbs', type=int, default=4, help='For supervised training.')
    parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
    parser.add_argument('--dsm_sigma', type=float, default=0.01)
    parser.add_argument('--score_net_hidden_dim', type=int, default=128)
    parser.add_argument('--score_net_num_blocks', type=int, default=4)
    ## Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    ## Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--clearml', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=1*MILLION)
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('--val_upsample_rate', type=int, default=4)
    parser.add_argument('--val_num_visualize', type=int, default=4)
    parser.add_argument('--val_noise', type=float, default=0.015)
    parser.add_argument('--ld_step_size', type=float, default=0.2)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    ## Project info
    parser.add_argument('--project_name', type=str, default=None, help='Project name')
    parser.add_argument('--task_name', type=str, default=None, help='Task in execution')
    parser.add_argument('--output_uri', type=str, default=None, help='ClearML URI')
    args = parser.parse_args()

    # Adjust devices and workers
    if args.device == 'cuda' and args.num_devices == -1:
        args.num_devices = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = int(0.666 * os.cpu_count())

    print("Args: ", args)
    seed_all(args.seed)

    # Start distributed processing
    world_size = args.num_devices
    mp.spawn(
        main, 
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )