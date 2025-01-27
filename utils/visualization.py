import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0)

    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_segmentation_sets(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 

    fulldir = opt.result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt):
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))

    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :].astype(np.uint8)
    if seg0.shape[0] != 256:
        seg0 = cv2.resize(seg0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 

    pt = np.array(pt.cpu())
    N = pt.shape[0]

    load_path = opt.load_path
    epoch = (load_path.split('.')[0]).split('_')[-2]
    section_classid = opt.test_split.split('test')[-1]

    fulldir = opt.result_path + "/epoch" + epoch + section_classid + "/img_add_seg/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_save_seg_gt_img(seg, gt, image_filename, opt, pt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori_save = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))

    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay_seg = img_ori * 0
    overlay_gt = img_ori * 0
    img_r_seg = img_ori[:, :, 0]
    img_g_seg = img_ori[:, :, 1]
    img_b_seg = img_ori[:, :, 2]
    img_r_gt = img_ori[:, :, 0]
    img_g_gt = img_ori[:, :, 1]
    img_b_gt = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                      [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :].astype(np.uint8)
    gt0 = gt[0, :, :]
    if seg0.shape[0] != 256:
        seg0 = cv2.resize(seg0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    if gt0.shape[0] != 256:
        gt0 = cv2.resize(gt0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    for i in range(1, opt.classes):
        img_r_seg[seg0 == i] = table[i - 1, 0]
        img_g_seg[seg0 == i] = table[i - 1, 1]
        img_b_seg[seg0 == i] = table[i - 1, 2]
        img_r_gt[gt0 == i] = table[9 - i, 0]
        img_g_gt[gt0 == i] = table[9 - i, 1]
        img_b_gt[gt0 == i] = table[9 - i, 2]

    overlay_seg[:, :, 0] = img_r_seg
    overlay_seg[:, :, 1] = img_g_seg
    overlay_seg[:, :, 2] = img_b_seg
    overlay_seg = np.uint8(overlay_seg)
    overlay_gt[:, :, 0] = img_r_gt
    overlay_gt[:, :, 1] = img_g_gt
    overlay_gt[:, :, 2] = img_b_gt
    overlay_gt = np.uint8(overlay_gt)

    img_add_gt = cv2.addWeighted(img_ori0, 0.4, overlay_gt, 0.6, 0)   # img_ori0 0.4 overlay 0.6
    img_add_seg_gt = cv2.addWeighted(img_add_gt, 0.4, overlay_seg, 0.6, 0)   #
    # img = img_ori0

    pt = np.array(pt.cpu())
    N = pt.shape[0]
    load_path = opt.load_path
    epoch = (load_path.split('.')[0]).split('_')[-2]
    section_classid = opt.test_split.split('test')[-1]

    fulldir = opt.result_path + "/epoch" + epoch + section_classid + "/img_add_seg_gt/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img_add_seg_gt)
    # save seg gt and img in a file
    save_seg_gt_img_path = opt.result_path + "/epoch" + epoch + section_classid + "/img_gt_seg"
    os.makedirs(save_seg_gt_img_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_seg_gt_img_path, image_filename.split('.')[0] + '_img.png'), img_ori_save)
    cv2.imwrite(os.path.join(save_seg_gt_img_path, image_filename.split('.')[0] + '_seg.png'), seg0 * 80)
    cv2.imwrite(os.path.join(save_seg_gt_img_path, image_filename.split('.')[0] + '_label.png'), gt0 * 80)

def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)