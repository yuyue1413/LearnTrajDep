#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""overall code framework is adapped from https://github.com/weigq/3d_pose_baseline_pytorch"""
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
import utils.model as nnmodel
import utils.data_utils as data_utils


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # 定义日志 CSV 文件
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + "_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.dct_n)

    # 创建模型
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    # 48 个角度预测节点
    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=48)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # Continue from checkpoint （从检查点继续）
    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_gcn_muti_att_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # 数据加载
    print(">>> loading data")
    train_dataset = H36motion(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                              split=0, sample_rate=sample_rate, dct_n=dct_n)
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean

    val_dataset = H36motion(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                            split=2, sample_rate=sample_rate, data_mean=data_mean, data_std=data_std, dct_n=dct_n)

    # 加载 dataset 进行训练
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    # 加载测试数据集 H36motion，并使用 DataLoader 对各动作（acts）分别创建数据加载器，存入字典 test_data 中
    acts = data_utils.define_actions('all') # 动作列表
    test_data = dict()
    for act in acts:
        test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                 sample_rate=sample_rate, data_mean=data_mean, data_std=data_std, dct_n=dct_n)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # 每个 epoch
        lr_now, t_l, t_e, t_3d = train(train_loader, model, optimizer, input_n=input_n,
                                       lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda,
                                       dim_used=train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l, t_e, t_3d])
        head = np.append(head, ['lr', 't_l', 't_e', 't_3d'])

        v_e, v_3d = val(val_loader, model, input_n=input_n, is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                        dct_n=dct_n)

        ret_log = np.append(ret_log, [v_e, v_3d])
        head = np.append(head, ['v_e', 'v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_e, test_3d = test(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=dct_n)
            ret_log = np.append(ret_log, test_e)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,
                                     [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            head = np.append(head, [act + '80', act + '160', act + '320', act + '400'])
            if output_n > 10:
                head = np.append(head, [act + '560', act + '1000'])
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # 更新日志文件并保存检查点
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_e):
            is_best = v_e < err_best
            err_best = min(v_e, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, input_n=20, dct_n=20, lr_now=None, max_norm=True, is_cuda=True, dim_used=[]):
    t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        # 如果 batch_norm 图层只有一个样本，则跳过最后一个批次
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = inputs.cuda().float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = all_seq.cuda(non_blocking=True).float()

        # 前向传播，将预测值维度展平为（batch_size, -1）
        outputs = model(inputs)
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)


        # 计算损失和反向传播
        loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n)
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        n, _, _ = all_seq.data.shape


        # mpjpe_error 和 euler_error 在内部保存了梯度计算的相关变量（如未调用 .detach() 或 .item()）导致显存占用增长
        # 所以outputs传进去的时候要调用detach()
        # 3D 误差
        m_err = loss_funcs.mpjpe_error(outputs.detach(), all_seq.detach(), input_n, dim_used, dct_n)
        # 角度空间误差
        e_err = loss_funcs.euler_error(outputs.detach(), all_seq.detach(), input_n, dim_used, dct_n).item()

        # 更新训练损失
        # loss.item()获取标量值
        t_l.update(loss.item() * n, n)
        t_e.update(e_err * n, n)
        t_3d.update(m_err * n, n)

        # 显存释放
        del outputs
        # torch.cuda.empty_cache()

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg, t_e.avg, t_3d.avg


def test(train_loader, model, input_n=20, output_n=50, dct_n=20, is_cuda=False, dim_used=[]):
    N = 0
    # t_l = 0
    if output_n >= 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]

    t_euler_error = np.zeros(len(eval_frame))
    t_3d_error = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    with torch.no_grad():
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()

            if is_cuda:
                inputs = inputs.cuda().float()
                # targets = Variable(targets.cuda(async=True)).float()
                all_seq = all_seq.cuda(non_blocking=True).float()

            outputs = model(inputs)
            n = outputs.shape[0]
            # outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)

            # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)


            # inverse dct transformation
            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = torch.from_numpy(idct_m).float().cuda()
            outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
            outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t.detach()).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                       seq_len).transpose(1,2)

            pred_expmap = all_seq.clone()
            dim_used = np.array(dim_used)
            pred_expmap[:, :, dim_used] = outputs_exp
            pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
            targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

            pred_expmap[:, 0:6] = 0
            targ_expmap[:, 0:6] = 0
            pred_expmap = pred_expmap.view(-1, 3)
            targ_expmap = targ_expmap.view(-1, 3)

            # get euler angles from expmap
            pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
            pred_eul = pred_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)
            targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
            targ_eul = targ_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)

            # get 3d coordinates
            targ_p3d = data_utils.expmap2xyz_torch(targ_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)
            pred_p3d = data_utils.expmap2xyz_torch(pred_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)

            # update loss and testing errors
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                # t_e[k] += torch.mean(torch.norm(pred_eul[:, j, :] - targ_eul[:, j, :], 2, 1)).item() * n
                # t_3d[k] += torch.mean(torch.norm(
                #     targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                #     1)).item() * n
                # 直接使用 PyTorch 张量
                t_euler_error[k] += (torch.norm(pred_eul[:, j, :] - targ_eul[:, j, :], dim=1).mean() * n).item()
                t_3d_error[k] += (torch.norm(
                    targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                    dim=1).mean() * n).item()
            # t_l += loss.cpu().data.numpy()[0] * n
            N += n

            # 显存释放
            del outputs, outputs_exp, pred_expmap, targ_expmap, pred_eul, targ_eul, targ_p3d, pred_p3d
            # torch.cuda.empty_cache()

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                             time.time() - st)
            bar.next()
    bar.finish()
    return t_euler_error / N, t_3d_error / N


def val(train_loader, model, input_n=20, dct_n=20, is_cuda=False, dim_used=[]):
    # t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))

    with torch.no_grad():
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()

            if is_cuda:
                inputs = inputs.cuda().float()
                all_seq = all_seq.cuda(non_blocking=True).float()

            outputs = model(inputs)
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)

            # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

            n, _, _ = all_seq.data.shape
            m_err = loss_funcs.mpjpe_error(outputs.detach(), all_seq.detach(), input_n, dim_used, dct_n).item()
            e_err = loss_funcs.euler_error(outputs.detach(), all_seq.detach(), input_n, dim_used, dct_n).item()

            # t_l.update(loss.cpu().data.numpy()[0] * n, n)
            t_e.update(e_err * n, n)
            t_3d.update(m_err * n, n)

            # 显存释放
            del outputs
            # torch.cuda.empty_cache()

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                             time.time() - st)
            bar.next()
    bar.finish()
    return t_e.avg, t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)