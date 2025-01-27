# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from torch.autograd import Variable
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary, visual_segmentation_sets, \
    visual_segmentation_sets_with_pt, visual_save_seg_gt_img
from einops import rearrange
from utils.generate_prompts import get_click_prompt, get_cls_prompt
import time
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import scoreatpercentile
from scipy import ndimage
from thop import profile

def hd95_manhattan(A, B):
    dist_A_to_B = cdist(A, B, 'cityblock')
    dist_B_to_A = cdist(B, A, 'cityblock')
    min_dist_A_to_B = np.min(dist_A_to_B, axis=1)
    min_dist_B_to_A = np.min(dist_B_to_A, axis=1)
    all_min_distances = np.concatenate((min_dist_A_to_B, min_dist_B_to_A))
    hd95_value = scoreatpercentile(all_min_distances, 95)
    return hd95_value

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def obtain_patien_id(filename):
    if "-" in filename:  # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid


def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype=torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype=torch.float32, device=opt.device))

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time = sum_time + (time.time() - start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j + 1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j + 1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            dices[1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[1] += iou
            accs[1] += acc
            ses[1] += se
            sps[1] += sp
            hds[1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices / eval_number
    hds = hds / eval_number
    ious, accs, ses, sps = ious / eval_number, accs / eval_number, ses / eval_number, sps / eval_number
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    print("test speed", eval_number / sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp


def eval_mask_slice2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype=torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype=torch.float32, device=opt.device))
        class_id = datapack['class_id']
        image_filename = datapack['image_name']
        print(image_filename)

        pt = get_click_prompt(datapack, opt)

        cls_sim_avg_label_input = torch.as_tensor(datapack['cls_sim_avg_label_input'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()

            pred = model(imgs, pt, None, cls_sim_avg_label_input)

            sum_time = sum_time + (time.time() - start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)

        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j + 1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j + 1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            dices[eval_number + j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)

            ious[eval_number + j, 1] += iou
            accs[eval_number + j, 1] += acc
            ses[eval_number + j, 1] += se
            sps[eval_number + j, 1] += sp
            hds[eval_number + j, 1] += hd95_manhattan(pred_i[0, :, :], gt_i[0, :, :])
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j + 1, :, :], image_filename[j], opt, pt[0][j, :, :])
                visual_save_seg_gt_img(seg[j:j + 1, :, :], gt[j:j + 1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number / sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(dices * 100, axis=0)
        dices_std = np.std(dices * 100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious * 100, axis=0)
        iou_std = np.std(ious * 100, axis=0)
        acc_mean = np.mean(accs * 100, axis=0)
        acc_std = np.std(accs * 100, axis=0)
        se_mean = np.mean(ses * 100, axis=0)
        se_std = np.std(ses * 100, axis=0)
        sp_mean = np.mean(sps * 100, axis=0)
        sp_std = np.std(sps * 100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
        label = datapack['label'].to(dtype=torch.float32, device=opt.device)
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time = sum_time + (time.time() - start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j + 1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j + 1, :, :] == 1] = 255
            dices[eval_number + j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number + j, 1] += iou
            accs[eval_number + j, 1] += acc
            ses[eval_number + j, 1] += se
            sps[eval_number + j, 1] += sp
            hds[eval_number + j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j + 1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :]
    hds = hds[:eval_number, :]
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number / sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(dices * 100, axis=0)
        dices_std = np.std(dices * 100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious * 100, axis=0)
        iou_std = np.std(ious * 100, axis=0)
        acc_mean = np.mean(accs * 100, axis=0)
        acc_std = np.std(accs * 100, axis=0)
        se_mean = np.mean(ses * 100, axis=0)
        se_std = np.std(ses * 100, axis=0)
        sp_mean = np.mean(sps * 100, axis=0)
        sp_std = np.std(sps * 100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def get_eval(valloader, model, criterion, opt, args):
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)
