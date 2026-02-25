# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import time
import os
import scipy.io
import math
from torch.optim import swa_utils
from tqdm import tqdm
from model import ft_net, ft_net_dense, PCB, PCB_test
from utils import fuse_all_conv_bn
version =  torch.__version__

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    # mask = np.in1d(index, junk_index, invert=True) # old numpy
    mask = np.isin(index, junk_index, invert=True) # new numpy
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    # mask = np.in1d(index, good_index) # old numpy
    mask = np.isin(index, good_index) # new numpy
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../datasets/PersonReIDDataset/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='PCB', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--nclasses', default=751, type=int, help='number of classes')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--usam', action='store_true', help='use usam.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')

def load_network(network):
    save_path = os.path.join('./weights', name, 'net_%s.pth'%opt.which_epoch)
    try:
        network.load_state_dict(torch.load(save_path))
    except: 
        #if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
            #print("Compiling model...")
            # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
            #torch.set_float32_matmul_precision('high')
            #network = torch.compile(network, mode="reduce-overhead", dynamic = True) # pytorch 2.0
        if 'average' in opt.which_epoch: # load averaged model.
            network = swa_utils.AveragedModel(network)
        network.load_state_dict(torch.load(save_path))
        if 'average' in opt.which_epoch:
            print("We average %d snapshots"%network.n_averaged)
            #swa_utils.update_bn(dataloaders['query'], network, device='cuda:0')
            network = network.module
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    # count = 0
    pbar = tqdm()
    if opt.linear_num <= 0:
        if opt.use_dense:
            opt.linear_num = 1024
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        if opt.use_PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.use_PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    pbar.close()
    return features

def evaluate_results(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_file, 
                    mquery_feature=None, mquery_label=None, mquery_cam=None):
    
    # Convert labels and cams to numpy arrays if they are lists
    if isinstance(gallery_label, list): gallery_label = np.array(gallery_label)
    if isinstance(gallery_cam, list): gallery_cam = np.array(gallery_cam)
    if isinstance(query_label, list): query_label = np.array(query_label)
    if isinstance(query_cam, list): query_cam = np.array(query_cam)
    
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    
    # Evaluate
    print('Evaluating...')
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    
    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
    print(str_result)
    
    # Write to file (tee behavior)
    with open(result_file, 'a') as f:
        f.write(str_result + '\n')

    # Multi-query evaluation
    if mquery_feature is not None:
        if isinstance(mquery_label, list): mquery_label = np.array(mquery_label)
        if isinstance(mquery_cam, list): mquery_cam = np.array(mquery_cam)

        mquery_feature = mquery_feature.cuda()
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        print('Evaluating Multi-query...')
        for i in range(len(query_label)):
            mquery_index1 = np.argwhere(mquery_label==query_label[i])
            mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
            mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
            mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
            ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        str_result_multi = 'multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
        print(str_result_multi)
        
        with open(result_file, 'a') as f:
            f.write(str_result_multi + '\n')

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

if __name__ == '__main__':
    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    #which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    h, w = 256, 128

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ############### Ten Crop        
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.ToTensor()(crop) 
            #      for crop in crops]
            # )),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
            #       for crop in crops]
            # ))
    ])

    if opt.use_PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        h, w = 384, 192


    data_dir = test_dir

    if opt.multi:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=opt.num_workers) for x in ['gallery','query','multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=opt.num_workers) for x in ['gallery','query']}
    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam,mquery_label = get_id(mquery_path)

    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
    elif opt.use_PCB:
        model_structure = PCB(opt.nclasses)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num, usam=opt.usam)

    # if opt.fp16:
    #    model_structure = network_to_half(model_structure)

    model = load_network(model_structure)

# Remove the final fc layer and classifier layer
    if opt.use_PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()


    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)

    print(model)
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders['gallery'])
        query_feature = extract_feature(model,dataloaders['query'])
        if opt.multi:
            mquery_feature = extract_feature(model,dataloaders['multi-query'])
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s')
    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}

    save_dir = f'./result/{opt.name}'
    os.makedirs(save_dir, exist_ok=True)

    scipy.io.savemat(f'./result/{opt.name}/pytorch_result.mat',result)

    print(opt.name)
    result_path = f'./result/{opt.name}/result.txt'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Run evaluation directly instead of using os.system and tee
    mquery_f, mquery_l, mquery_c = None, None, None
    if opt.multi:
        result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
        scipy.io.savemat(f'./result/{opt.name}/multi_query.mat',result)
        mquery_f = mquery_feature
        mquery_l = mquery_label
        mquery_c = mquery_cam

    evaluate_results(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_path, 
                     mquery_feature=mquery_f, mquery_label=mquery_l, mquery_cam=mquery_c)


