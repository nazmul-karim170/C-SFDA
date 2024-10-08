import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import PIL

import my_transforms as my_transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import wandb
import math
import torch.nn.functional as F
from moco.loader import NCropsTransform
from Contrastive_loss import *
from PIL import ImageFilter

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_augmentation(aug_type, normalize=None):
    if not normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    if aug_type == "moco-v2":
        soft = True

        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.85, 0.85, 0.65, 0.25)],
                    p=0.85,  
                ),
                transforms.RandomGrayscale(p=0.6),
                transforms.RandomRotation(degrees = [-2,2]),
                transforms.RandomPosterize(8, p=0.6),
                transforms.RandomEqualize(p=0.6),
                transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.6),
                # transforms.AugMix(5,5),           ## While Applying Augmix, comment out the ColorJitter
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )


    elif aug_type == "moco-v1":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.85, 0.85, 0.65, 0.25)],
                    p=0.8,  
                ),
                transforms.RandomGrayscale(p=0.65),
                transforms.RandomRotation(degrees = [-2,2]),
                transforms.RandomPosterize(8, p=0.65),
                transforms.RandomEqualize(p=0.65),
                transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.65),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.05),           ## prob 0.1 works
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  ## all 0.1 works
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return None

def get_augmentation_versions(aug_versions= 'tcctwswswswswsswstwswswswstwswswswswstwswswswswswstwswswswswswstwswswswswswswwswswswstwswswswswswswswswswstwswswswswswswswstwswswswsws'):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in aug_versions:    ## Change the value of augmented versions 
        if version == "s":
            transform_list.append(get_augmentation("moco-v2"))
        elif version == "c":
            transform_list.append(get_augmentation("moco-v1"))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "t":
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    
    transform = NCropsTransform(transform_list)

    return transform

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    transform_train = get_augmentation_versions()
    dsets["target"] = ImageList_idx(txt_tar, transform=transform_train)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

## For Ema Update
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model
    

## Contrastive loss (Modify It)
def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):

    # Labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # In class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)
    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy

class MLP(nn.Module):
    def __init__(self, input_channels=512, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    model_anchor.eval()
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    
    for param in ema_model.parameters():
        param.detach_()

    return model_state, optimizer_state, ema_model, model_anchor



def train_target_con(args):
    dset_loaders = data_load(args)
    
    ## Set Base Network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF_ema = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim= netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netR = network.feat_classifier(type='linear', class_num=128, bottleneck_dim=args.bottleneck).cuda()

    modelpath = osp.join(args.output_dir, "net_S_SSL.pt")
    netR.load_state_dict(torch.load(modelpath))

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))

    netF.eval()
    netB.eval()
    netR.train()
 
    contrastive_criterion = SupConLoss()


    for k, v in netR.named_parameters():
        param_group = [{'params': v, 'lr': args.lr*1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = 0.1*args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    contrast_acc = 0
    while iter_num < max_iter:

        try:
            inputs_test, labels_check, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, labels_check, tar_idx = iter_test.next()

        inputs_test = torch.stack(inputs_test).cuda()
        labels_check= labels_check.cuda()
        iter_num   += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

                ## get the Features and logits
        features_test1  = netR(netB(netF(inputs_test[1])))
        
            ###  Contrastive Learning ### (Gradually move into Momentum Contrastive Learning)
        f1       = F.normalize(torch.squeeze(features_test1), dim=1)
        f2       = F.normalize(torch.squeeze(netR(netB(netF(inputs_test[2]))), dim=1))
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_contrast = contrastive_criterion(features)   

        loss = loss_contrast     


        ## Update the Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_netR = netR.state_dict()
        print("Iteration Number:", iter_num)

        if args.issave:   
            torch.save(netR.state_dict(), osp.join(args.output_dir, "net_S_SSL.pt"))
            
    return best_netR


def train_target(args):
    dset_loaders = data_load(args)
    
    ## Set Base Network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF_ema = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim= netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

                    # Load Contrastive head ##
    netR = network.feat_classifier(type='linear', class_num=128, bottleneck_dim=args.bottleneck).cuda()
    # netR_dict = train_target_con(args)
    modelpath = osp.join(args.output_dir, "net_S_SSL.pt")
    netR.load_state_dict(torch.load(modelpath))


    netB_ema = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC_ema = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    # netF = torch.nn.DataParallel(netF)
    # netB = torch.nn.DataParallel(netB)
    # netC = torch.nn.DataParallel(netC)

    ## You can do this
    netF.train()
    netB.train()
    netC.eval()

    # ## Or this (Both Case should work)
    # netF.eval()
    # netB.eval()
    # netC.train()

    loss_coef = 1
    con_coeff = 0.5
    contrastive_criterion = SupConLoss()

    param_group = []

                ### For Backbone Training ###
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
    netR.train()

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    while iter_num < max_iter:

        try:
            inputs_test, labels_check, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, labels_check, tar_idx = iter_test.next()

        inputs_test = torch.stack(inputs_test).cuda()
        labels_check= labels_check.cuda()
        iter_num   += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

                ## get the Features and logits
        features_test1  = netB(netF(inputs_test[1]))
        logits_q        = netC(features_test1)
        
        #### Number of Augmentations (We need more augmentations for this)
        K    =   36

        outputs_emas = []
        with torch.no_grad():
            for jj in range(3,K):
                outputs_emas.append(netC(netB(netF(inputs_test[jj]))))

                        ## Average the predictions for pseudo-labels
            outputs_ema = torch.stack(outputs_emas).mean(0)
            probs_w, pseudo_labels_w = torch.nn.functional.softmax(outputs_ema, dim=1).max(1)
        

                            ########## PL Selection Module ##############
        mean_expand =  0.6        
        pred_start     = torch.nn.functional.softmax(torch.squeeze(torch.stack(outputs_emas)), dim=2).max(2)[0]  
        conf_th        = pred_start.mean() - mean_expand*pred_start.std() 
        confidence_sel = pred_start.mean(0) > conf_th                                     ## Confidence score based selection (Some classes "maximum confidence score"

        ## Uncertainty Based Selection
        pred_std              = pred_start.std(0)                                                                              ## Standard Deviation 
        uncer_th              = pred_std.mean(0) + 0.6*pred_std.std()    
        uncertainty_sel       = pred_std<uncer_th

        ## Confidence and Uncertainty Based Selection
        truth_array = torch.logical_and(uncertainty_sel, confidence_sel)
        ind_keep   = truth_array.nonzero()
        ind_remove = (~truth_array).nonzero()
        
        try:
            ind_total = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove)), dim=0)
        except:
            ind_total = ind_remove
                        
        try:
                            ### Apply Class-Balancing (Only the selected Samples) ###
            unique_labels, counts =  pseudo_labels_w[ind_keep].unique(return_counts = True)
            min_count             =  torch.min(counts)
            # print("Sample in each class:", unique_labels.size(), counts)

            # print(unique_labels.size(),unique_labels)
            ## For Missing Classes
            num_class = args.class_num
            if len(counts) < args.class_num:
                counts_new = torch.ones(args.class_num)
                missing_classes = [ii for ii in range(args.class_num) if ii not in unique_labels]
                
                for kk in missing_classes:
                    indices = (pseudo_labels_w == kk).nonzero(as_tuple=True)[0]

                    if indices.numel()>0 and ind_keep.numel()>0:
                        probs  = probs_w[indices]
                        _ , index_miss = probs.sort(descending=True)                                                          ## print("The index ::::::::::::::::::::::::::::: ", index_miss)

                        try:
                            ind_keep  = torch.cat((torch.squeeze(ind_keep), torch.unsqueeze(indices[index_miss[0]], 0)))         ## Taking all missing classes samples deteriorates the performance
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

            loss_cls  = classification_loss(
                torch.squeeze(outputs_ema[ind_keep]), torch.squeeze(logits_q[ind_keep]), torch.squeeze(pseudo_labels_w[ind_keep]), torch.squeeze(outputs_ema[ind_keep]),  args, 1/counts_new.cuda()
            )

        except:
            loss_cls  = classification_loss_unsel(
                torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w), torch.squeeze(outputs_ema), args
            )       

        try:
            ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
        except:
            pass

        try:
            loss_cls_rem , accuracy_psd_meter = classification_loss_unsel(
                torch.squeeze(outputs_ema[ind_remove]), torch.squeeze(logits_q[ind_remove]), torch.squeeze(pseudo_labels_w[ind_remove]), torch.squeeze(outputs_ema[ind_remove]), args
            )

                ###  Contrastive Learning ### (Gradually move into Momentum Contrastive Learning)
            f1       = F.normalize(torch.squeeze(netR(features_test1[ind_remove])), dim=1)
            f2       = F.normalize(torch.squeeze(netR(netB(netF(inputs_test[2][ind_remove])))), dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_contrast = contrastive_criterion(features)   

        except:
            loss_cls_rem = 0
            loss_contrast = 0


                ## Loss Coefficients ###
        difficulty_score = uncer_th/conf_th
        loss_coef *= (1- 0.00001* torch.exp(-1/difficulty_score))
        con_coeff *=  np.exp(-0.00001)

        ## At the beginning, we want to learn from more confident samples
        loss = loss_coef * loss_cls + (1-loss_coef)* loss_cls_rem + con_coeff*loss_contrast  


        ## Update the Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter or iter_num==1:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = '............Task: {}, Iter:{}/{}; Accuracy = {:.2f}%..........'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = '.................Task: {}, Iter:{}/{}; Accuracy = {:.2f}%..............'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()


        if args.issave:   
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
            
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def classification_loss(outputs_ema, logits_s, target_labels, targets_preds, args, class_weights= None):

    # loss_cls = torch.mean((logits_s - targets_preds)**2)
    loss_cls = cross_entropy_loss(logits_s, target_labels, args, class_weights)
        
    return loss_cls

def classification_loss_unsel(outputs_ema, logits_s, target_labels, targets_preds, args):
    loss_cls = torch.mean((targets_preds-logits_s)**2) 
    # loss_cls = cross_entropy_loss(logits_s, target_labels, args)

    return loss_cls

def div(logits, epsilon=1e-8):

    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def diversification_loss(outputs_ema, logits_s, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_div = div(outputs_ema)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(outputs_ema) + div(logits_s)

    return loss_div


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args, class_weights=None):
    return F.cross_entropy(logits, labels, weight = class_weights)


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()

    return loss


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=160, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)

                ## Work on This ## 
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1)
    parser.add_argument('--lr_decayC', type=float, default=1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        # names = ['Art', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)