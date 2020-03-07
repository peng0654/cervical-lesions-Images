"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from models.res2net import res2net50_26w_4s, res2net50_26w_8s, res2net50_14w_8s, res2net101_26w_4s
from models.res2net_v1b import res2net101_v1b_26w_4s, res2net50_v1b
from models.res2next import res2next50
from models.resnet_plus import resnet50_plus, resnet34_cbam_plus, resnet34_plus
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from apex import amp
import config
from models import WSDAN
from models import vgg19_bn
from models import inception_v3
from models import resnet50_cbam
from models import  *
from models import res2net
from models.google_net_gmp import net_add
from models.res2net import *
import torch.hub
from datasets import get_trainval_datasets, CervicalDataset
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
# GPU settings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
# weight = torch.tensor([0.5, 1.0])

cross_entropy_loss = nn.CrossEntropyLoss()
cross_entropy_loss.cuda()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, ))
crop_metric = TopKAccuracyMetric(topk=(1, ))
drop_metric = TopKAccuracyMetric(topk=(1, ))


def main():
    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################

    # train_data_size = int(config.num_image * 0.8)
    # val_data_size = config.num_image - train_data_size
    indices_train = np.random.RandomState(777).permutation(500)
    indices_test = np.random.RandomState(777).permutation(140)

    print(indices_test)
    print('tran_data_size', len(indices_train), 'val_data_size', len(indices_test))
    train_dataset = CervicalDataset(phase='train', resize=config.image_size, indices = indices_train)
    validate_dataset = CervicalDataset(phase='val', resize=config.image_size, indices= indices_test)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0

    #net = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True,)
    net =res2net50(pretrained=True,num_classes=2)
    print(net)
    net.aux_logits = False
    # i=0
    # for m in net.children():
    #     if isinstance(m, nn.Conv2d) and i < 3:
    #         for param in m.parameters():
    #             param.requires_grad=False
    #             i = i+1

    # print(net)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    # feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate =  config.learning_rate
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    #net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    ##################################
    # TRAINING
    ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('{}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              optimizer=optimizer,
              pbar=pbar)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net)
        torch.save(net.state_dict(), os.path.join(config.save_dir, config.model_name))
        pbar.close()



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)

        r = np.random.rand(1)
        if config.beta > 0 and r < config.cutmix_prob:

            # feature_center_batch = F.normalize(feature_center[y], dim=-1)
            # feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)
            # generate mixed sample
            lam = np.random.beta(config.beta, config.beta)
            rand_index = torch.randperm(X.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
            X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
            # compute output
            y_pred_raw= net(X)
            batch_loss = cross_entropy_loss(y_pred_raw, target_a) * lam + cross_entropy_loss(y_pred_raw, target_b) * (1. - lam)
        else:
            # compute output
            y_pred_raw = net(X)
            batch_loss = cross_entropy_loss(y_pred_raw, y)



        y_pred = (y_pred_raw )
        # backward
        batch_loss.backward()
        # optimizer.zero_grad()
        # with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred, y)


        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_raw_acc[0], epoch_raw_acc[0])

        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred= net(X)



            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[0])
    print(batch_info)
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


if __name__ == '__main__':
    main()
