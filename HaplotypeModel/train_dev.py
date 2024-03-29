import sys
import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
import math
import numpy as np
from model_dev import CatModel, weights_init, LSTMNetwork
from dataset_dev import TrainingDataset,EvaluateDataset
from optim import Optimizer
from utils import AttrDict, init_logger, count_parameters, save_model
from tensorboardX import SummaryWriter
import torchnet.meter as meter
from datetime import datetime
from get_truth import load_reference_file


# from options import gt_decoded_labels, zy_decoded_labels, indel1_decoded_labels, indel2_decoded_labels


def train(epoch, config, model, training_data, references, num_batches_per_epoch, batch_size, optimizer, logger, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    start = time.process_time()
    total_loss = 0
    total_images = 0
    optimizer.epoch()
    batch_steps = num_batches_per_epoch

    for bin_file in os.listdir(training_data):
        train_dataset = TrainingDataset(bin_path = training_data+'/'+bin_file,references=references,pileup_length=config.model.pileup_length,haplotype_length=config.model.haplotype_length, pn_value=config.data.pn_value)
        if train_dataset.__len__()==0:
            continue
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.training.num_gpu * 3)

        for step, (pileup_feat, haplotype_feat, gt, zy) in enumerate(train_loader):
            """
            pileup_feat: [N, 104, 33]
            haplotype_feat: [N, 104, 11]
            gt: [N,]
            zy: [N,]
            """
            x_pileup = pileup_feat.type(torch.FloatTensor)
            x_haplotype = haplotype_feat.type(torch.FloatTensor)
            gt_label = gt.type(torch.LongTensor)
            zy_label = zy.type(torch.LongTensor)
            if x_pileup.shape[0] == 0 or x_pileup.shape[1] == 0 or x_pileup.ndim != 3:
                continue
            if config.training.num_gpu > 0:
                x_pileup = x_pileup.cuda()
                x_haplotype = x_haplotype.cuda()
                gt_label = gt_label.cuda()
                zy_label = zy_label.cuda()
                loss,_,_ = model(x_pileup, x_haplotype, gt_label, zy_label)
                loss.backward()
                total_loss += loss.item()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                # grad_norm = 0
                optimizer.step()
                total_images += x_pileup.shape[0]
            avg_loss = (total_loss / total_images) if total_images else 0
            if visualizer is not None:
                visualizer.add_scalar(
                    'train_loss', loss.item(), optimizer.global_step)
                visualizer.add_scalar(
                    'learn_rate', optimizer.lr, optimizer.global_step)
        end = time.process_time()
        process = (optimizer.global_step % batch_steps) / batch_steps * 100
        logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                    grad_norm, loss.item(), avg_loss, end - start))
        start = time.process_time()

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' % (epoch, total_loss / (step + 1), end_epoch - start_epoch))


def eval(epoch, config, model, validating_data, references, batch_size, logger, visualizer=None):
    model.eval()
    total_loss = 0
    total_images = 0
    gt_confusion_matrix = meter.ConfusionMeter(config.model.gt_num_class)
    batch_steps = len(validating_data)
    total_acc = 0
    total_cnt = 0
    for bin_file in os.listdir(validating_data):
        validate_dataset = EvaluateDataset(bin_path = validating_data+'/'+bin_file,references=references,pileup_length=config.model.pileup_length,haplotype_length=config.model.haplotype_length)
        if validate_dataset.__len__()==0:
            continue
        validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=config.training.num_gpu * 3)
        for step, (pileup_feat, haplotype_feat, gt, zy) in enumerate(validate_loader):
            """
            pileup_feat: [N, 104, 33]
            haplotype_feat: [N, 104, 11]
            gt: [N,]
            zy: [N,]
            """
            x_pileup = pileup_feat.type(torch.FloatTensor)
            x_haplotype = haplotype_feat.type(torch.FloatTensor)
            gt_label = gt.type(torch.LongTensor)
            zy_label = zy.type(torch.LongTensor)
            if x_pileup.shape[0] == 0 or x_pileup.shape[1] == 0 or x_pileup.ndim != 3:
                continue
            if config.training.num_gpu > 0:
                x_pileup = x_pileup.cuda()
                x_haplotype = x_haplotype.cuda()
                gt_label = gt_label.cuda()
                zy_label = zy_label.cuda()
                loss, gt_out, zy_out = model(x_pileup, x_haplotype, gt_label, zy_label)
                gt_logits = torch.softmax(gt_out, 1).detach().cpu().numpy()
                gt_output = np.argmax(gt_logits, axis=1)
                # print('pred:',gt_output[:50].tolist())
                # print('labl:',sub_gt_label.detach().cpu().numpy()[:50].tolist())
                total_acc += sum((gt_output == gt_label.detach().cpu().numpy()).astype(int)) / len(gt_output)
                total_cnt += 1

                gt_confusion_matrix.add(gt_out.data.contiguous().view(-1, config.model.gt_num_class), gt_label.data.contiguous().view(-1))

                total_images += x_pileup.shape[0]

                gt_cm_value = gt_confusion_matrix.value()

                gt_denom = gt_cm_value.sum() if gt_cm_value.sum() > 0 else 1.0

                gt_total_accurate = 0
                for j in range(0, config.model.gt_num_class):
                    gt_total_accurate = gt_total_accurate + gt_cm_value[j][j]
                gt_accuracy = (100.0 * gt_total_accurate) / gt_denom

                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)

            # if step % config.training.show_interval == 0:
            #     process = step / batch_steps * 100
            #     logger.info(
            #         '-Validation-Epoch:%d(%.5f%%), Loss: %.5f , AverageLoss: %.5f, GT_Acc: %.5f, ZY_Acc: %.5f ' % (
            #             epoch, process, loss, avg_loss, gt_accuracy, zy_accuracy))

    gtcm = gt_confusion_matrix.value()
    gt_tp, gt_fp, gt_fn = np.zeros(config.model.gt_num_class), np.zeros(config.model.gt_num_class), np.zeros(
        config.model.gt_num_class)
    for x in range(len(gtcm)):
        gt_tp[x] += gtcm[x][x]
        gt_fp[x] += sum(gtcm[:, x]) - gtcm[x][x]
        gt_fn[x] += sum(gtcm[x, :]) - gtcm[x][x]
    gt_recall = gt_tp / (gt_tp + gt_fn + 1e-6)
    gt_precision = gt_tp / (gt_tp + gt_fp + 1e-6)
    gt_f1 = 2 * (gt_precision * gt_recall) / (gt_precision + gt_recall + 1e-6)
    gt_recall = gt_recall.mean()
    gt_precision = gt_precision.mean()
    gt_f1 = gt_f1.mean()

    logger.info(
        '-Validation-Epoch:%d, AverageLoss: %.5f, GT:|Recall: %.4f, Precision: %.4f, F1: %.4f |, ACC: %.4f' % (
            epoch, avg_loss, gt_recall, gt_precision, gt_f1, total_acc / total_cnt))

    if visualizer is not None:
        visualizer.add_scalar('eval_loss', avg_loss, epoch)
        visualizer.add_scalar('gt_f1_score', gt_f1, epoch)
        visualizer.add_scalar('accuracy', total_acc / total_cnt, epoch)

    return {'loss': avg_loss,
            'gt_accuracy': gt_accuracy,
            'gt_confusion_matrix': str(gt_confusion_matrix.conf.tolist()),
            'gt_f1_score': gt_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='path to config file', required=True)
    parser.add_argument('-log', type=str, default='train.log', help='name of log file')
    parser.add_argument('-mode', type=str, default='retrain', help="training mode: retain or finetune")
    parser.add_argument('-model_path', type=str, help='path to pre-trained model')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('edges', config.configname, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    # num_workers = config.training.num_gpu * 2
    # train_dataset = TrainingDataset(bin_path = config.data.train,pn_value=1.0)
    # training_data = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=num_workers)
    # logger.info('Load Train Set!')

    # dev_dataset = TrainingDataset(bin_path = config.data.eval,pn_value=1.0)
    # validate_data = torch.utils.data.DataLoader(dev_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=num_workers)
    # logger.info('Load Dev Set!')

    # test_dataset = PolishTrainDataset(config.data.test, config.model.max_rle)
    # test_data = torch.utils.data.DataLoader(test_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
    #                                         shuffle=False, num_workers=num_workers)
    # logger.info('Load Test Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    references = load_reference_file(config.data.reference)

    num_batches_per_epoch = 0
    for bin_file in os.listdir(config.data.train):
        train_dataset = TrainingDataset(bin_path = config.data.train+'/'+bin_file,references=references,pileup_length=config.model.pileup_length,haplotype_length=config.model.haplotype_length,pn_value=config.data.pn_value)
        num_batches_per_epoch += math.ceil(train_dataset.__len__()/config.training.batch_size)
    print("batch_size:",config.training.batch_size)
    print("num_batches_per_epoch:",num_batches_per_epoch)
    print("num_epochs:",config.training.epochs)

    if opt.mode == 'retrain':
        #model = CatModel(pileup_dim=config.model.pileup_dim, haplotype_dim=config.model.haplotype_dim, hidden_size=config.model.hidden_size, nclass=config.model.gt_num_class).cuda()
        model = LSTMNetwork(config).cuda()
        model.apply(weights_init)
        optimizer = Optimizer(model.parameters(), config, num_batches_per_epoch)
        logger.info('Created a %s optimizer.' % config.optim.type)
    elif opt.mode == 'finetune':
        #model = CatModel(pileup_dim=config.model.pileup_dim, haplotype_dim=config.model.haplotype_dim, hidden_size=config.model.hidden_size, nclass=config.model.gt_num_class).cuda()
        model = LSTMNetwork(config).cuda()
        model.load_state_dict(torch.load(opt.model_path))
        optimizer = Optimizer(model.parameters(), config, num_batches_per_epoch, finetune=True)
        logger.info('Created a %s optimizer.' % config.optim.type)
    else:
        print("ERROR: Unknown training mode.")
        sys.exit()



    start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visual_log = os.path.join(exp_name, 'log')
        visualizer = SummaryWriter(os.path.join(visual_log, 'train'))
        dev_visualizer = SummaryWriter(os.path.join(visual_log, 'dev'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    confusion_matrix_logger = open(os.path.join(exp_name, "confusion_matrix.txt"), 'w')
    stats = dict()
    stats['loss_epoch'] = []
    stats['gt_accuracy_epoch'] = []
    max_gt_f1_score = 0

    for epoch in range(start_epoch, config.training.epochs):

        train(epoch, config, model, config.data.train, references, num_batches_per_epoch, config.training.batch_size, optimizer, logger, visualizer)

        if config.training.eval_or_not:
            stats_dictioanry = eval(epoch, config, model, config.data.eval, references, config.training.batch_size, logger, dev_visualizer)

            stats['loss'] = stats_dictioanry['loss']
            stats['gt_accuracy'] = stats_dictioanry['gt_accuracy']
            stats['loss_epoch'].append((epoch, stats_dictioanry['loss']))
            stats['gt_accuracy_epoch'].append((epoch, stats_dictioanry['gt_accuracy']))
            confusion_matrix_logger.write(str(epoch + 1) + "\n" + str(stats_dictioanry['gt_confusion_matrix']) + "\n")
            confusion_matrix_logger.flush()

            if stats_dictioanry['gt_f1_score'] >= max_gt_f1_score:
                max_gt_f1_score = stats_dictioanry['gt_f1_score']
                save_name = os.path.join(exp_name, '%s.chkpt' % (config.training.save_model))
                # save_model(model, optimizer, config, save_name)
                torch.save(model.state_dict(), save_name)
                logger.info('Epoch %d model has been saved.' % epoch)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        torch.save(model.state_dict(), save_name)
        # save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)

        if config.optim.type!='Ranger':
            if epoch >= config.optim.begin_to_adjust_lr:
                optimizer.decay_lr()
                # early stop
                if optimizer.lr < 1e-7:
                    logger.info('The learning rate is too low to train.')
                    break
                logger.info('Epoch %d update learning rate: %.6f' % (epoch, optimizer.lr))

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
