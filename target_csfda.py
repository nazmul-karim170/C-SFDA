from copy import deepcopy
import logging
import os
import time

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import math
import torch.nn as nn 

from classifier import Classifier
from image_list import ImageList
from Model import CSFDA_Model, NCropsTransform
import numpy as np
from Contrastive_loss import *
import matplotlib.pyplot as plt
import torchvision.utils as utils

from utils_csfda import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
)


@torch.no_grad()
def eval_and_label_dataset(dataloader, model, args):
    wandb_dict = dict()

    # Make sure to switch to eval mode
    model.eval()

    # Run inference
    logits, gt_labels, indices = [], [], []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for _ , imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        _, logits_cls = model(imgs, cls_only=True)

        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    logits    = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices   = torch.cat(indices).to("cuda")

    if args.distributed:

        ## Gather results from all ranks
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        ## Remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)

    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict["Test Acc"] = accuracy
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Avg"] = acc_per_class.mean()
        wandb_dict["Test Per-class"] = acc_per_class

    if use_wandb(args):
        wandb.log(wandb_dict)

    return acc_per_class



def get_augmentation_versions(args):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in args.learn.aug_versions:    ## Change the value of augmented versions 
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "t":
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    
    transform = NCropsTransform(transform_list)

    return transform


def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": 30*args.optim.lr,                  ## For Fully test-time domain adaptation, don't use a high learning rate for VISDA-C.
                                                             ## And, For DomainNet-126, best online result comes when Learning Rate is 2e-4, and use LRM of 20
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer



def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )

    # If not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.tgt_domain}_list.txt"
        )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length   = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    checkpoint_path = os.path.join(
        args.model_tta.src_log_dir,
        f"best_{args.data.src_domain}_{args.seed}.pth.tar",
    )

    # filename = f"checkpoint_0001_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
    # checkpoint_path = os.path.join(args.log_dir, filename)    

    ## Model Initization
    src_model = Classifier(args.model_src, checkpoint_path)
    ema_model = Classifier(args.model_src, checkpoint_path)       ## For contrastive loss
    
    model = CSFDA_Model(
        src_model,
        ema_model,
        m=args.model_tta.m,
    ).cuda()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
    logging.info(f"1 - Created target model")

    ## Validation Data
    val_transform   = get_augmentation("test")
    train_transform = get_augmentation_versions(args)
    label_file  = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform =val_transform,
    )
    
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )

    acc = eval_and_label_dataset(
        val_loader, model, args=args
    )
    logging.info("2 - Computed initial pseudo labels")

    # banks= None
    # pseudo_item_list = []

    ### Training data  (Is it domain by domain, look for online settings!)
    train_transform = get_augmentation_versions(args)
    train_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,                # Uses pseudo labels
        transform=train_transform,
        pseudo_item_list=None,
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.data.workers,
        pin_memory=False,            ## make it False, otherwise 
        sampler=train_sampler,
        drop_last=False,
    )

    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("3 - Created train/val loader")

    ### Define loss function (criterion) and optimizer   
    optimizer = get_target_optimizer(model, args)
    logging.info("4 -- Created Optimizer")
    logging.info("Start Training ...")

    ### Main Training Part    
    if args.distributed:
        train_sampler.set_epoch(1)

    ### Our Proposed Training Algorithm
    train_csfda(train_loader, val_loader, model, optimizer,  args)

    filename = f"checkpoint_{1:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
    save_path = os.path.join('./checkpoints/', filename)
    save_checkpoint(model, optimizer, 1, save_path=save_path)
    logging.info(f"Saved checkpoint {save_path}")


            ### Our Proposed Training Algorithm ######
            ##########################################
def train_csfda(train_loader, val_loader, model, optimizer, args):
    
    epoch = 1
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins   = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd   = AverageMeter("CLS-Acc@1", ":6.2f")
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    ## Make sure to switch to train mode
    model.train()

    ## Number of Augmentations 
    if args.data.dataset == "VISDA-C":
        num_class = 12
    else:
        num_class = 126

    N = 8
    accuracy_tot = 0
    accuracy_r = 0
    total_acc = 1
    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    loss_coef   = 1
    con_coeff   = 0.5
    contrastive_criterion = SupConLoss()
    L2loss                = torch.nn.MSELoss()
    
    mem_size       = 2
    class_features = torch.zeros((mem_size, num_class, 256))
    probs_class    = torch.zeros((mem_size, num_class, num_class))

    con_coeffs   = np.zeros(20000) 
    loss_classes = torch.zeros(20000) 
    loss_coefs   = torch.zeros(20000) 
    con_losses   = torch.zeros(20000) 
    unsupervised_losses    = torch.zeros(20000) 
    uncertainty_thresholds = torch.zeros(20000) 
    conf_thress = torch.zeros(20000) 
    acc_classes = []
    accuracies  = []
    sel_Samples = []
    unsel_samples = []
    missed_images =  {'img_path': [], 'labels': []}
    ind = 0

    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        for i, data in enumerate(train_loader):
            
            ## Unpack and move data
            img_path, images, labels_check , idxs = data
            labels_check = labels_check.to("cuda")
            idxs = idxs.to("cuda")

            ## Images for updating the model
            images_w, images_q, images_k = (
                images[0].to("cuda"),
                images[1].to("cuda"),
                images[2].to("cuda"),
            )

            ## (N-3) number of Images for Calculating uncertainty 
            images_un    = torch.stack(images[3:N]).to("cuda")
            outputs_emas = []
            feats = []

            with torch.no_grad():
                for jj in range(N-3):
                    outputs_emas.append(model.ema_model(images_un[jj], return_feats=True)[1])                      

                ## Average the predictions for pseudo-labels
                outputs_ema = torch.stack(outputs_emas).mean(0)
                probs_w, pseudo_labels_w = torch.nn.functional.softmax(outputs_ema, dim=1).max(1)

            ## Per-step scheduler (Learning Rate Decay)
            step = i + epoch * len(train_loader)
            adjust_learning_rate(optimizer, step, args)

            ## Get the logits 
            feats_con, logits_q = model(images_q , images_k)

                                ############################################
                                ########## PL Selection Module ##############
                                #############################################
            pred_start     = torch.nn.functional.softmax(torch.squeeze(torch.stack(outputs_emas)), dim=2).max(2)[0] 

            ## Confidence Based Selection
            pred_con       = pred_start                                                                                     
            conf_thres     = pred_con.mean()
            confidence_sel = pred_con.mean(0) > conf_thres                                      
            conf_th = pred_con.mean()

            ## Uncertainty Based Selection
            pred_std              = pred_start.std(0)                                                                               
            uncertainty_threshold = pred_std.mean(0)    
            uncertainty_sel       = pred_std<uncertainty_threshold
            uncer_th = pred_std.mean(0)

            ## Confidence and Uncertainty Based Selection
            truth_array = torch.logical_and(uncertainty_sel, confidence_sel)
            ind_keep   = truth_array.nonzero()
            ind_remove = (~truth_array).nonzero()
            
            try:
                ind_total = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove)), dim=0)
            except:
                ind_total = ind_remove

            ## Confidence Score Difference (DoC) Based Selection
            if ind_remove.numel():
                threshold = torch.zeros(len(ind_remove))
                num = 0
                for kk in ind_remove:
                    out     = torch.squeeze(outputs_ema[kk])
                    out , _ = out.sort(descending=True)
                    threshold[num] = out[0] - out[1]
                    num    += 1

                pre_threshold = threshold.mean(0) 
                truth_array1  = threshold>pre_threshold
                truth_array2  = pred_std[ind_remove] < pred_std[ind_remove].mean(0)                   ## Add Underconfident Clean Samples 
                truth_array   = torch.logical_and(truth_array1.cuda(), truth_array2.cuda())
                ind_add       = truth_array.nonzero()
                
                try:
                    ind_keep   = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove[ind_add])), dim=0)
                    ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
                except:
                    pass 
                            
            try:
                                ### Apply Class-Balancing (Only the selected Samples) ###
                unique_labels, counts =  pseudo_labels_w[ind_keep].unique(return_counts = True)
                min_count             =  torch.min(counts)

                ## For Missing Classes
                if len(counts) < num_class:
                    counts_new = torch.ones(num_class)
                    missing_classes = [ii for ii in range(num_class) if ii not in unique_labels]
                    
                    for kk in missing_classes:
                        indices = (pseudo_labels_w == kk).nonzero(as_tuple=True)[0]

                        if indices.numel()>0 and ind_keep.numel()>0:
                            probs  = probs_w[indices]
                            _ , index_miss = probs.sort(descending=True)                                    
                
                            try:
                                ind_keep  = torch.cat((ind_keep, indices[index_miss[0:min_count]]))         ## Taking all missing classes samples deteriorates the performance
                            except:
                                pass
                            counts_new[kk] = 1 
                        else:
                            counts_new[kk] = 1
                    
                    ## Other Classes
                    num = 0
                    for nn in unique_labels:
                        counts_new[nn] = counts[num]
                        num += 1
                else:
                    counts_new = counts 

                loss_cls , accuracy_psd = classification_loss(
                    torch.squeeze(outputs_ema[ind_keep]), torch.squeeze(logits_q[ind_keep]), torch.squeeze(pseudo_labels_w[ind_keep]), torch.squeeze(outputs_ema[ind_keep]),  args, 1/counts_new.cuda()
                )

            except:
                print("Oh No!!!, There is no confident samples::", ind_keep.numel(), ind_remove.numel())
                loss_cls , accuracy_psd = propagation_loss(
                    torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w), torch.squeeze(outputs_ema), args
                )       


                ### Calculate Pseudo-Label Accuracy Accuracy ###
            accuracy = (pseudo_labels_w[ind_keep] == labels_check[ind_keep]).float().mean() * 100
            if not math.isnan(accuracy):
                accuracy_tot += accuracy
                total_acc += 1
            accuracies.append(accuracy_tot/total_acc)


                ### Contrastive Learning ### 
            feats_k  = model(images_k, cls_only=True)[0]                                  
            f1       = F.normalize(torch.squeeze(feats_con[ind_remove]), dim=1)
            f2       = F.normalize(torch.squeeze(feats_k[ind_remove]), dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_contrast = contrastive_criterion(features)              


                ### Propagation Loss ###
            ## If the clean selected set is empty, calculate loss for all samples  
            try:
                loss_cls_rem , accuracy_psd_meter = propagation_loss(
                    torch.squeeze(outputs_ema[ind_remove]), torch.squeeze(logits_q[ind_remove]), torch.squeeze(pseudo_labels_w[ind_remove]), torch.squeeze(outputs_ema[ind_remove]), args
                )
            except:
                loss_cls_rem = 0

            _ , accuracy_psd_meter = propagation_loss(
                torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w), torch.squeeze(outputs_ema), args
            )        
            top1_psd.update(accuracy_psd_meter.item(), len(outputs_ema))

                   ## Loss Coefficients ###
            difficulty_score = uncer_th/conf_th
            loss_coef *= (1- 0.001* torch.exp(-1/difficulty_score))
            con_coeff *=  np.exp(-0.0001)

            ## At the beginning, we want to learn from more confident samples
            loss = loss_coef * loss_cls + (1-loss_coef)* loss_cls_rem + con_coeff*loss_contrast  

            ## Update the Parameters
            loss_meter.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                    ### Training Statistics ####
            con_coeffs[ind] = con_coeff 
            loss_classes[ind] = loss_cls.item()
            loss_coefs[ind] = loss_coef
            con_losses[ind]  = loss_contrast.item()
            unsupervised_losses[ind] = loss_cls_rem.item()
            uncertainty_thresholds[ind] = uncer_th
            conf_thress[ind] = conf_th
            sel_Samples.append(len(ind_keep))
            unsel_samples.append(len(ind_remove))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.learn.print_freq == 0:
                progress.display(i)

            missed_images['labels'].append(labels_check[ind_remove])


            np.savez("training_stats.npz", pseudo_label_acc = accuracies, acc_class= acc_classes, conf = conf_thress.cpu().numpy(), unc = uncertainty_thresholds.cpu().numpy(), labeled_loss_coeff = loss_coefs.cpu().numpy(), con_coeff = con_coeffs, ce_loss = loss_classes.cpu().numpy(), con_loss = con_losses.cpu().numpy(), prop_loss = unsupervised_losses.cpu().numpy(), sel_Samples = sel_Samples , unsel_samples = unsel_samples)
            ind += 1 

            ## Evaluate the model ##
        acc_per_class = eval_and_label_dataset(val_loader, model, args)
        model.train()
        acc_classes.append(acc_per_class.mean())        




@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy


def classification_loss(outputs_ema, logits_s, target_labels, targets_preds, args, class_weights= None):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(outputs_ema, target_labels, args, class_weights)
        accuracy = calculate_acc(outputs_ema, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        # loss_cls = torch.mean((logits_s - targets_preds)**2)
        loss_cls = cross_entropy_loss(logits_s, target_labels, args, class_weights)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy

def propagation_loss(outputs_ema, logits_s, target_labels, targets_preds, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(outputs_ema, target_labels, args)
        accuracy = calculate_acc(outputs_ema, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = torch.mean((targets_preds-logits_s)**2)
        # loss_cls = cross_entropy_loss(logits_s, target_labels, args)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy



def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args, class_weights=None):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels, weight = class_weights)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()

    return loss
