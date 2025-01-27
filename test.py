import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse

from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch

from utils.config import get_config
from utils.evaluation import get_eval

from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile
import matplotlib.pyplot as plt
import cv2
import pandas as pd


def main():
    #  =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='EchoONE', type=str, help='type of model, e.g., SAM, EchoONE...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in EchoONE')
    parser.add_argument('--task', default='Cardiac_multi_plane_test', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "val"
    opt.visual = True
    # opt.eval_mode = "patient"
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    #  ============================================= model and data preparation ===================================================
    # get similarity
    class_similarity_test_array = pd.read_csv('..cos_sim/test.csv').values

    # get average label
    avg_label_path = '../avg_label'
    dict_avg_label = {
        'avg_label_cluster1': np.load(os.path.join(avg_label_path, 'avg_label_cluster1.npy')),
        'avg_label_cluster2': np.load(os.path.join(avg_label_path, 'avg_label_cluster2.npy')),
        'avg_label_cluster3': np.load(os.path.join(avg_label_path, 'avg_label_cluster3.npy')),
        'avg_label_cluster4': np.load(os.path.join(avg_label_path, 'avg_label_cluster4.npy'))
    }

    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu

    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    # ------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    #  ======================================= begin to evaluate the model =======================================================
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops / 1000000000, 'params:', params)

    tf_test = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)

    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_test, img_size=args.encoder_input_size,
                                 class_id=args.test_classid, cls_sim_arr=class_similarity_test_array, dict_avg_label=dict_avg_label)
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()

    mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(
        testloader, model, criterion=criterion, opt=opt, args=args)
    print("dataset:" + args.task + " -----------model name: " + args.modelname)
    print("task", args.task, "checkpoints:", opt.load_path)
    print("mean_dice, mean_hd95, mean_iou, mean_acc, mean_se, mean_sp:")
    print(mean_dice[1:], mean_hdis[1:], mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
    print("std_dice, std_hd95, std_iou, std_acc, std_se, std_sp:")
    print(std_dice[1:], std_hdis[1:], std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])


if __name__ == '__main__':
    main()
