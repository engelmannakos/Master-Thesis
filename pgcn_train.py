from tensorboardX import SummaryWriter
import os
import time
import shutil
import torch
import random
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import numpy as np
from pgcn_dataset import PGCNDataSet
from pgcn_models import PGCN
from pgcn_opts import parser
from ops.pgcn_ops import CompletenessLoss, ClassWiseRegressionLoss
from ops.utils import get_and_save_args, get_logger
from tools.Recorder import Recorder
#from tensorboardX import SummaryWriter

import json
import pandas as pd

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_loss = 100
cudnn.benchmark = True
pin_memory = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def reconstruct_data(prop_path, ft_path, gt_annotations, subject, dataset, feat_x):
    print('Reconstruct data')
    n_sbj = len([os.path.join(gt_annotations, f) for f in os.listdir(gt_annotations)]) 
    multip = 1

    with open(gt_annotations+f'/loso_sbj_{subject}.json', 'r') as fid:
        json_data = json.load(fid)
    json_db = json_data['database']

    # create tensor features files
    if not os.path.isdir(ft_path+f'/sbj_{subject}'):
        os.makedirs(ft_path+f'/sbj_{subject}/train')
        os.makedirs(ft_path+f'/sbj_{subject}/test')

        for key, sbj in json_db.items():
            if feat_x == 'raw':
                #print(pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv').dtypes)
                sbj_X = pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv', low_memory=False).drop(['sbj_id','label'], axis=1)
                sbj_X = sbj_X.fillna(float(0))
                #print(key, sbj_X.isna().sum().sum())
                #print(sbj_X.values.shape)
                
                if dataset != 'wear':
                    #! Sometimes the whole feature is just 0s, which ends in NaNs after normalization
                    for column in sbj_X.columns: 
                        sbj_X[column] = (sbj_X[column] - sbj_X[column].min()) / (sbj_X[column].max() - sbj_X[column].min())	 
                    
                    sbj_X = sbj_X.fillna(float(0))
                print(key, sbj_X.isna().sum().sum())

                tensor = torch.tensor(sbj_X.values)


            else:
                sbj_X = np.load(f'data/{dataset}/processed/{feat_x}_features/sbj_{subject}/{key}.npy')
                tensor = torch.tensor(sbj_X)

            if sbj['subset'] == 'Validation':
                for m in range(multip):
                    torch.save(tensor, ft_path+f'/sbj_{subject}/test/{key}_{m}')
            else:
                for m in range(multip):
                    torch.save(tensor, ft_path+f'/sbj_{subject}/train/{key}_{m}')


    # create prop_files
    if not os.path.isdir(prop_path+f'/sbj_{subject}'):
        os.makedirs(prop_path+f'/sbj_{subject}')


    prop_file_list = {}
    for i in range(n_sbj):
        with open(f'../PRSA-Net/output/{dataset}/{feat_x}/sbj_{i}/generated_result_proposal.json', 'r') as fid:
            json_data = json.load(fid)
        prop_file_list.update(json_data['results']) #! in gen_res_prop.json file there's always only 1 subject data


    n_vals = 0
    n_train = 0
    if 'hangtime' in ft_path:
        n_used_props = 500
    elif 'wear' in ft_path:
        n_used_props = 200
    elif 'rwhar' in ft_path:
        n_used_props = 80
    elif 'opportunity' in ft_path:
        n_used_props = 500
    elif 'wetlab' in ft_path:
        n_used_props = 50
    elif 'sbhar' in ft_path:
        n_used_props = 50
    else:
        raise KeyError('n_used_props not specified.')



    for key, sbj in json_db.items():
        fps = sbj['fps']
        if sbj['subset'] == 'Validation':

            # test ground truths
            _n_frame = len(pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv', low_memory=False))
            _gt_list = []

            for _gt in sbj['annotations']:
                label = _gt['label_id'] +1 #?????????
                startf = round(_gt['segment (frames)'][0])
                endf = round(_gt['segment (frames)'][1])

                #if len(_gt_list) > 0 and _gt_list[-1][2]+1 != startf:
                #    _gt_list.append([0, _gt_list[-1][2]+1, startf-1]) #background
                _gt_list.append([label, startf, endf]) #actual gt

            #add first and last background gts
            #_gt_list.insert(0, [0, 1, _gt_list[0][1]-1]) #first BG from the 1st frame to the first actual gt
            #_gt_list.append([0, _gt_list[-1][2]+1, _n_frame]) #last BG from the last actual gt to the last frame

            _n_gt = len(_gt_list)


            # test proposals
            props = prop_file_list.pop(key) # [{'score': 0.3476788401603699, 'segment': [1259.8, 1260.4]}, ...]
            _prop_list = []
            for proposal in props:
                if len(_prop_list) == n_used_props: #!!!!!!!!!!!!!
                    break #!!!!!!!!!!!!!
                label = 0 #?-1
                best_iou = 0
                self_overlap = 0
                p_startf = round(proposal['segment'][0]*fps)
                p_endf = round(proposal['segment'][1]*fps)

                for _gt in _gt_list:
                    if p_startf > _gt[2] or p_endf < _gt[1]:        # <>
                        continue
                    else:
                        if p_endf < _gt[2]:
                            i_endf = p_endf
                            u_endf = _gt[2]
                        else:
                            i_endf = _gt[2]
                            u_endf = p_endf

                        if p_startf > _gt[1]:
                            i_startf = p_startf
                            u_startf = _gt[1]
                        else:
                            i_startf = _gt[1]
                            u_startf = p_startf
                        
                        iou = (i_endf-i_startf+1) / (u_endf-u_startf+1) #! Intersection / Union, +1 is needed to get the correct frame number
                        s_overlap = (i_endf-i_startf+1) / (p_endf-p_startf+1) #! Intersection / Proposal

                        if iou > best_iou:
                            label = _gt[0] #+1 #?????????
                            best_iou = iou
                            self_overlap = s_overlap
                
                if label == -1:
                    continue

                _prop_list.append([label, best_iou, self_overlap, p_startf, p_endf])

            _n_prop = len(_prop_list)

            # create final test txt
            with open(prop_path+f'/sbj_{subject}/test_proposal_list.txt', 'a') as f: #! 'w' would create a new file, 'a' creates or appends
                for m in range(multip):
                    f.write(f'#{n_vals}'+'\n')
                    n_vals += 1
                    f.write(f'{key}_{m}\n')
                    f.write(str(_n_frame)+'\n')
                    f.write('1'+'\n') #! It's a multiplier, but it's always 1
                    f.write(str(_n_gt)+'\n')
                    for gt in _gt_list:
                        for data in gt:
                            f.write(str(data)+' ')
                        f.write('\n')
                    f.write(str(_n_prop)+'\n')
                    for prop in _prop_list:
                        for data in prop:
                            f.write(str(data)+' ')
                        f.write('\n')

        else:
            # train ground truths
            #! Since the original paper uses leave-one-out strategy, there's always only 1 validation dataset
            _n_frame = len(pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv'))
            _gt_list = []

            for _gt in sbj['annotations']:
                label = _gt['label_id'] +1 #?????????
                startf = round(_gt['segment (frames)'][0])
                endf = round(_gt['segment (frames)'][1])

                #if len(_gt_list) > 0:
                #    _gt_list.append([0, _gt_list[-1][2]+1, startf-1]) #background #! Faszsag, mert a raw databol ezeken a helyeken nem null class van
                _gt_list.append([label, startf, endf]) #actual gt

            #add first and last background gts
            #_gt_list.insert(0, [0, 1, _gt_list[0][1]-1]) #first BG from the 1st frame to the first actual gt
            #_gt_list.append([0, _gt_list[-1][2]+1, _n_frame]) #last BG from the last actual gt to the last frame

            _n_gt = len(_gt_list)

            # train proposals
            props = prop_file_list.pop(key) #! props has only one subject data
            _prop_list = []
            for proposal in props:
                if len(_prop_list) == n_used_props: #!!!!!!!!!!!!!
                    break #!!!!!!!!!!!!!
                label = 0 #?-1
                best_iou = 0
                self_overlap = 0
                p_startf = round(proposal['segment'][0]*fps)
                p_endf = round(proposal['segment'][1]*fps)

                for _gt in _gt_list:
                    if p_startf > _gt[2] or p_endf < _gt[1]:        # <>
                        continue
                    else:
                        if p_endf < _gt[2]:
                            i_endf = p_endf
                            u_endf = _gt[2]
                        else:
                            i_endf = _gt[2]
                            u_endf = p_endf

                        if p_startf > _gt[1]:
                            i_startf = p_startf
                            u_startf = _gt[1]
                        else:
                            i_startf = _gt[1]
                            u_startf = p_startf
                        
                        iou = (i_endf-i_startf+1) / (u_endf-u_startf+1) #! Intersection / Union, +1 is needed to get the correct frame number
                        s_overlap = (i_endf-i_startf+1) / (p_endf-p_startf+1) #! Intersection / Proposal

                        if iou > best_iou:
                            label = _gt[0] #+1 #???????????
                            best_iou = iou
                            self_overlap = s_overlap

                if label == -1:
                    continue

                _prop_list.append([label, best_iou, self_overlap, p_startf, p_endf])

            #_prop_list = sorted(_prop_list, key=lambda x: x[1])
            _n_prop = len(_prop_list)

            # create final test txt
            with open(prop_path+f'/sbj_{subject}/train_proposal_list.txt', 'a') as f:
                for m in range(multip):
                    f.write(f'#{n_train}'+'\n')
                    n_train += 1
                    f.write(f'{key}_{m}\n')
                    f.write(str(_n_frame)+'\n')
                    f.write('1'+'\n') #! It's a multiplier, but it's always 1
                    f.write(str(_n_gt)+'\n')
                    for gt in _gt_list:
                        for data in gt:
                            f.write(str(data)+' ')
                        f.write('\n')
                    f.write(str(_n_prop)+'\n')
                    for prop in _prop_list:
                        for data in prop:
                            f.write(str(data)+' ')
                        f.write('\n')

        #if len(prop_file_list) == 0:
        #    break


    return


def reconstruct_mini_data(prop_path, ft_path, gt_annotations, subject, dataset, num_cls):
    print('Reconstruct data')
    n_sbj = len([os.path.join(gt_annotations, f) for f in os.listdir(gt_annotations)]) 
    multip = 1

    with open(gt_annotations+f'/loso_sbj_{subject}.json', 'r') as fid:
        json_data = json.load(fid)
    json_db = json_data['database']

    # create tensor features files
    if not os.path.isdir(ft_path+f'/sbj_{subject}'):
        os.makedirs(ft_path+f'/sbj_{subject}/train')
        os.makedirs(ft_path+f'/sbj_{subject}/test')
        for key, sbj in json_db.items():
            sbj_X = pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv').drop(['sbj_id','label'], axis=1).values
            tensor = torch.tensor(sbj_X)
            if sbj['subset'] == 'Validation':
                for cls in range(1, num_cls+1):
                    for m in range(multip):
                        torch.save(tensor, ft_path+f'/sbj_{subject}/test/{key}_{cls}_{m}')

            else:
                for cls in range(1, num_cls+1):
                    for m in range(multip):
                        torch.save(tensor, ft_path+f'/sbj_{subject}/train/{key}_{cls}_{m}')




    # create prop_files
    if not os.path.isdir(prop_path+f'/sbj_{subject}'):
        os.makedirs(prop_path+f'/sbj_{subject}')


    prop_file_list = {}
    for i in range(n_sbj):
        with open(f'../PRSA-Net/output/{dataset}/sbj_{i}/generated_result_proposal.json', 'r') as fid:
            json_data = json.load(fid)
        prop_file_list.update(json_data['results']) #! in gen_res_prop.json file there's always only 1 subject data

    n_vals = 0
    n_train = 0

    for key, sbj in json_db.items():
        fps = sbj['fps']
        if sbj['subset'] == 'Validation':

            # test ground truths
            #_id = f'#{n_vals}'
            #n_vals += 1
            #_name = key
            _n_frame = len(pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv'))
            #_n_gt = len(sbj['annotations'])
            _gt_list = []
            for _gt in sbj['annotations']:
                label = _gt['label_id'] +1 #?????????
                startf = round(_gt['segment (frames)'][0])
                endf = round(_gt['segment (frames)'][1])

                #if len(_gt_list) > 0:
                #    _gt_list.append([0, _gt_list[-1][2]+1, startf-1]) #background
                _gt_list.append([label, startf, endf]) #actual gt

            #add first and last background gts
            #_gt_list.insert(0, [0, 1, _gt_list[0][1]-1]) #first BG from the 1st frame to the first actual gt
            #_gt_list.append([0, _gt_list[-1][2]+1, _n_frame]) #last BG from the last actual gt to the last frame
        
            _n_gt = []
            for i in range(1, num_cls+1):
                _n_gt.append(len([gt for gt in _gt_list if gt[0] == i]))

            # test proposals
            props = prop_file_list.pop(key) # [{'score': 0.3476788401603699, 'segment': [1259.8, 1260.4]}, ...]
            _prop_list = []
            for proposal in props:
                if len(_prop_list) == 50: #!!!!!!!!!!!!!
                    break #!!!!!!!!!!!!!
                label = 0 #?-1
                best_iou = 0
                self_overlap = 0
                p_startf = round(proposal['segment'][0]*fps)
                p_endf = round(proposal['segment'][1]*fps)

                for _gt in _gt_list: #! Ez igy nem jo, mert igy majdnem mindig 1.0-s self_overlap lesz
                    if p_startf > _gt[2] or p_endf < _gt[1]:
                        continue
                    else:
                        if p_endf < _gt[2]:
                            i_endf = p_endf
                            u_endf = _gt[2]
                        else:
                            i_endf = _gt[2]
                            u_endf = p_endf

                        if p_startf > _gt[1]:
                            i_startf = p_startf
                            u_startf = _gt[1]
                        else:
                            i_startf = _gt[1]
                            u_startf = p_startf
                        
                        iou = (i_endf-i_startf+1) / (u_endf-u_startf+1) #! Intersection / Union, +1 is needed to get the correct frame number
                        s_overlap = (i_endf-i_startf+1) / (p_endf-p_startf+1) #! Intersection / Proposal

                        if iou > best_iou:
                            label = _gt[0] #+1 #?????????
                            best_iou = iou
                            self_overlap = s_overlap
                
                if label == -1:
                    continue

                _prop_list.append([label, best_iou, self_overlap, p_startf, p_endf])

            _n_prop = []
            for i in range(1, num_cls+1):
                _n_prop.append(len([prop for prop in _prop_list if prop[0] == i]))

            # create final test txt
            with open(prop_path+f'/sbj_{subject}/test_proposal_list.txt', 'a') as f: #! 'w' would create a new file, 'a' creates or appends
                for m in range(multip):
                    for cls in range(1, num_cls+1):
                        f.write(f'#{n_vals}\n')
                        n_vals += 1
                        f.write(f'{key}_{cls}_{m}\n')
                        f.write(str(_n_frame)+'\n') #? CHANGE nem lehet ugy feldarabolni classokra, h frameszamot is lehessen mondani. Maradjon mindegyiknek az egesz
                        f.write('1'+'\n') #! It's a multiplier, but it's always 1
                        f.write(str(_n_gt[cls-1])+'\n')
                        for gt in _gt_list:
                            if gt[0] == cls:
                                for data in gt:
                                    f.write(str(data)+' ')
                                f.write('\n')
                        f.write(str(_n_prop[cls-1])+'\n')
                        for prop in _prop_list:
                            if prop[0] == cls:
                                for data in prop:
                                    f.write(str(data)+' ')
                                f.write('\n')

        else:
            #continue
            # train ground truths
            #_id = f'#{n_train}' #! Since the original paper uses leave-one-out strategy, there's always only 1 validation dataset
            #n_train += 1
            #_name = key
            _n_frame = len(pd.read_csv(f'data/{dataset}/raw/inertial/{key}.csv'))
            #_n_gt = len(sbj['annotations'])
            _gt_list = []
            for _gt in sbj['annotations']:
                label = _gt['label_id'] +1 #?????????
                startf = round(_gt['segment (frames)'][0])
                endf = round(_gt['segment (frames)'][1])

                #if len(_gt_list) > 0:
                #    _gt_list.append([0, _gt_list[-1][2]+1, startf-1]) #background
                _gt_list.append([label, startf, endf]) #actual gt

            #add first and last background gts
            #_gt_list.insert(0, [0, 1, _gt_list[0][1]-1]) #first BG from the 1st frame to the first actual gt
            #_gt_list.append([0, _gt_list[-1][2]+1, _n_frame]) #last BG from the last actual gt to the last frame

            _n_gt = []
            for i in range(1, num_cls+1):
                _n_gt.append(len([gt for gt in _gt_list if gt[0] == i]))

            # train proposals
            props = prop_file_list.pop(key) #! props has only one subject data
            _prop_list = []
            for proposal in props:
                if len(_prop_list) == 50: #!!!!!!!!!!!!!
                    break #!!!!!!!!!!!!!
                label = 0 #?-1
                best_iou = 0
                self_overlap = 0
                p_startf = round(proposal['segment'][0]*fps)
                p_endf = round(proposal['segment'][1]*fps)

                for _gt in _gt_list:
                    if p_startf > _gt[2] or p_endf < _gt[1]:        # <>
                        continue
                    else:
                        if p_endf < _gt[2]:
                            i_endf = p_endf
                            u_endf = _gt[2]
                        else:
                            i_endf = _gt[2]
                            u_endf = p_endf

                        if p_startf > _gt[1]:
                            i_startf = p_startf
                            u_startf = _gt[1]
                        else:
                            i_startf = _gt[1]
                            u_startf = p_startf
                        
                        iou = (i_endf-i_startf+1) / (u_endf-u_startf+1) #! Intersection / Union, +1 is needed to get the correct frame number
                        s_overlap = (i_endf-i_startf+1) / (p_endf-p_startf+1) #! Intersection / Proposal

                        if iou > best_iou:
                            label = _gt[0] #+1 #???????????
                            best_iou = iou
                            self_overlap = s_overlap

                if label == -1:
                    continue

                _prop_list.append([label, best_iou, self_overlap, p_startf, p_endf])

            _n_prop = []
            for i in range(1, num_cls+1):
                _n_prop.append(len([prop for prop in _prop_list if prop[0] == i]))

            # create final test txt
            with open(prop_path+f'/sbj_{subject}/train_proposal_list.txt', 'a') as f:
                for m in range(multip):
                    for cls in range(1, num_cls+1):  
                        f.write(f'#{n_train}\n')
                        n_train += 1
                        f.write(f'{key}_{cls}_{m}\n')
                        f.write(str(_n_frame)+'\n')
                        f.write('1'+'\n') #! It's a multiplier, but it's always 1
                        f.write(str(_n_gt[cls-1])+'\n') #! It has to be cls-1, bc class 1's num will be in the 0th place in _n_gt
                        for gt in _gt_list:
                            if gt[0] == cls:
                                for data in gt:
                                    f.write(str(data)+' ')
                                f.write('\n')
                        f.write(str(_n_prop[cls-1])+'\n')
                        for prop in _prop_list:
                            if prop[0] == cls:
                                for data in prop:
                                    f.write(str(data)+' ')
                                f.write('\n')

        #if len(prop_file_list) == 0:
        #    break



    return





def main():
    global args, best_loss, writer, adj_num, logger

    configs = get_and_save_args(parser)
    parser.set_defaults(**configs)
    dataset_configs = configs["dataset_configs"]
    model_configs = configs["model_configs"]
    graph_configs = configs["graph_configs"]
    args = parser.parse_args()

    model_configs['act_feat_dim'] = model_configs[f'{args.feat_ext}_act_feat_dim'] # now we get the proper dimensions
    model_configs['comp_feat_dim'] = model_configs[f'{args.feat_ext}_comp_feat_dim']

    """copy codes and creat dir for saving models and logs"""
    if args.dataset != 'thumos14' and args.dataset != 'activitynet1.3':
        args.snapshot_pref += f'{args.dataset}/{args.feat_ext}/sbj_{args.sbj}/'
    if not os.path.isdir(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    logger = get_logger(args)
    logger.info('\ncreating folder: ' + args.snapshot_pref)

    #if not args.evaluate:
    #    writer = SummaryWriter(args.snapshot_pref)
    #    recorder = Recorder(args.snapshot_pref, ["models", "__pycache__"])
    #    recorder.writeopt(args)

    logger.info('\nruntime args\n\n{}\n\nconfig\n\n{}'.format(args, dataset_configs))


    """construct model"""
    model = PGCN(model_configs, graph_configs)
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.dataset != 'thumos14' and args.dataset != 'activitynet1.3':
        
        reconstruct_data(
            prop_path=dataset_configs['train_prop_file']+f'/{args.feat_ext}', 
            ft_path=dataset_configs['train_ft_path']+f'/{args.feat_ext}',
            gt_annotations=dataset_configs['gt_annotations'],
            subject=args.sbj,
            dataset=args.dataset,
            feat_x = args.feat_ext
            )
        
        """        
        reconstruct_mini_data(
            prop_path=dataset_configs['train_prop_file'], 
            ft_path=dataset_configs['train_ft_path'],
            gt_annotations=dataset_configs['gt_annotations'],
            subject=args.sbj,
            dataset=args.dataset,
            num_cls=model_configs['num_class']
            )
        """

        dataset_configs['train_prop_file'] += f'/{args.feat_ext}/sbj_{args.sbj}/train_proposal_list.txt'
        dataset_configs['test_prop_file'] += f'/{args.feat_ext}/sbj_{args.sbj}/test_proposal_list.txt'

        dataset_configs['train_ft_path'] += f'/{args.feat_ext}/sbj_{args.sbj}/train'
        dataset_configs['test_ft_path'] += f'/{args.feat_ext}/sbj_{args.sbj}/test'

        if not os.path.isdir(dataset_configs['train_dict_path']+f'/{args.feat_ext}/sbj_{args.sbj}'):
            os.makedirs(dataset_configs['train_dict_path']+f'/{args.feat_ext}/sbj_{args.sbj}')

        dataset_configs['train_dict_path'] += f'/{args.feat_ext}/sbj_{args.sbj}/train_prop_dict.pkl'
        dataset_configs['val_dict_path'] += f'/{args.feat_ext}/sbj_{args.sbj}/val_prop_dict.pkl'
        #dataset_configs['test_dict_path'] += f'/sbj_{args.sbj}/test_prop_dict.pkl'

    print('----------- DATASET ----------')
    """construct dataset"""
    train_loader = torch.utils.data.DataLoader(
        PGCNDataSet(dataset_configs, graph_configs,
                    prop_file=dataset_configs['train_prop_file'],
                    prop_dict_path=dataset_configs['train_dict_path'],
                    ft_path=dataset_configs['train_ft_path'],
                    epoch_multiplier=dataset_configs['training_epoch_multiplier'],
                    test_mode=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)  # in training we drop the last incomplete minibatch
    #print('///////')
    print('len(train_loader):', len(train_loader))
    #print('///////')

    val_loader = torch.utils.data.DataLoader(
        PGCNDataSet(dataset_configs, graph_configs,
                    prop_file=dataset_configs['test_prop_file'],
                    prop_dict_path=dataset_configs['val_dict_path'],
                    ft_path=dataset_configs['test_ft_path'],
                    epoch_multiplier=dataset_configs['testing_epoch_multiplier'],
                    reg_stats=train_loader.dataset.stats,
                    test_mode=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    """loss and optimizer"""
    activity_criterion = torch.nn.CrossEntropyLoss().cuda()
    completeness_criterion = CompletenessLoss().cuda()
    regression_criterion = ClassWiseRegressionLoss().cuda()

    for group in policies:
        logger.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, activity_criterion, completeness_criterion, regression_criterion, 0)
        return


    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, model, activity_criterion, completeness_criterion, regression_criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, activity_criterion, completeness_criterion, regression_criterion, (epoch + 1) * len(train_loader))
            # remember best validation loss and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': loss,
                'reg_stats': torch.from_numpy(train_loader.dataset.stats)
            }, is_best, epoch, filename='thumos_checkpoint.pth.tar')

    #writer.close()


def train(train_loader, model, act_criterion, comp_criterion, regression_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    ohem_num = train_loader.dataset.fg_per_video
    comp_group_size = train_loader.dataset.fg_per_video + train_loader.dataset.incomplete_per_video
    for i, (prop_fts, prop_type, prop_labels, prop_reg_targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = prop_fts[0].size(0)

        activity_out, activity_target, activity_prop_type, \
        completeness_out, completeness_target, \
        regression_out, regression_labels, regression_target = model((prop_fts[0], prop_fts[1]), prop_labels,
                                                                     prop_reg_targets, prop_type)

        #print('/////prop_type/////')
        #print(prop_type)
        #print(prop_type[0])
        #print(prop_type.size())

        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(completeness_out, completeness_target, ohem_num, comp_group_size)
        reg_loss = regression_criterion(regression_out, regression_labels, regression_target)

        loss = act_loss + comp_loss * args.comp_loss_weight + reg_loss * args.reg_loss_weight

        losses.update(loss.item(), batch_size)
        act_losses.update(act_loss.item(), batch_size)
        comp_losses.update(comp_loss.item(), batch_size)
        reg_losses.update(reg_loss.item(), batch_size)

        act_acc = accuracy(activity_out, activity_target)
        act_accuracies.update(act_acc[0].item(), activity_out.size(0))

        #! With these modifications, the model can handle if activity_prop_type only has 1 FG and 1 BG
        #! which is the case when we're trying to use BGs.
        if len((activity_prop_type == 0).nonzero().squeeze().size()) == 0:
            fg_indexer = (activity_prop_type == 0).nonzero().squeeze(0)
        else:
            fg_indexer = (activity_prop_type == 0).nonzero().squeeze()

        if len((activity_prop_type == 2).nonzero().squeeze().size()) == 0:
            bg_indexer = (activity_prop_type == 2).nonzero().squeeze(0)
        else:
            bg_indexer = (activity_prop_type == 2).nonzero().squeeze()

        fg_acc = accuracy(activity_out[fg_indexer, :], activity_target[fg_indexer])
        fg_accuracies.update(fg_acc[0].item(), len(fg_indexer))

        if len(bg_indexer) > 0:
            bg_acc = accuracy(activity_out[bg_indexer, :], activity_target[bg_indexer])
            bg_accuracies.update(bg_acc[0].item(), len(bg_indexer))

        loss.backward()

        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    logger.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0

            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        #writer.add_scalar('data/loss', losses.val, epoch*len(train_loader)+i+1)
        #writer.add_scalar('data/Reg_loss', reg_losses.val, epoch*len(train_loader)+i+1)
        #writer.add_scalar('data/Act_loss', act_losses.val, epoch*len(train_loader)+i+1)
        #writer.add_scalar('data/comp_loss', comp_losses.val, epoch*len(train_loader)+i+1)

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Act. Loss {act_losses.val:.3f} ({act_losses.avg: .3f}) \t'
                  'Comp. Loss {comp_losses.val:.3f} ({comp_losses.avg: .3f}) '
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, act_losses=act_losses,
                    comp_losses=comp_losses, lr=optimizer.param_groups[0]['lr'], ) +
                  '\tReg. Loss {reg_loss.val:.3f} ({reg_loss.avg:.3f})'.format(
                      reg_loss=reg_losses)
                  + '\n Act. FG {fg_acc.val:.02f} ({fg_acc.avg:.02f}) Act. BG {bg_acc.avg:.02f} ({bg_acc.avg:.02f})'
                  .format(act_acc=act_accuracies,
                    fg_acc=fg_accuracies, bg_acc=bg_accuracies)
                  )


def validate(val_loader, model, act_criterion, comp_criterion, regression_criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    ohem_num = val_loader.dataset.fg_per_video
    comp_group_size = val_loader.dataset.fg_per_video + val_loader.dataset.incomplete_per_video

    for i, (prop_fts, prop_type, prop_labels, prop_reg_targets) in enumerate(val_loader):
        # measure data loading time
        batch_size = prop_fts[0].size(0)
        print(batch_size)

        activity_out, activity_target, activity_prop_type, \
        completeness_out, completeness_target, \
        regression_out, regression_labels, regression_target = model((prop_fts[0], prop_fts[1]), prop_labels,
                                                                     prop_reg_targets, prop_type)
        #print('/////prop_type/////')
        #print(prop_type)
        #print(prop_type[0])
        #print(prop_type.size())

        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(completeness_out, completeness_target, ohem_num, comp_group_size)
        #print('/////regression_out/////')
        #print(regression_out)
        reg_loss = regression_criterion(regression_out, regression_labels, regression_target)
        #raise ValueError('Temporal STOP')
        loss = act_loss + comp_loss * args.comp_loss_weight + reg_loss * args.reg_loss_weight

        losses.update(loss.item(), batch_size)
        act_losses.update(act_loss.item(), batch_size)
        comp_losses.update(comp_loss.item(), batch_size)
        reg_losses.update(reg_loss.item(), batch_size)

        act_acc = accuracy(activity_out, activity_target)
        act_accuracies.update(act_acc[0].item(), activity_out.size(0))

        #! With these modifications, the model can handle if activity_prop_type only has 1 FG and 1 BG
        #! which is the case when we're trying to use BGs.
        if len((activity_prop_type == 0).nonzero().squeeze().size()) == 0:
            fg_indexer = (activity_prop_type == 0).nonzero().squeeze(0)
        else:
            fg_indexer = (activity_prop_type == 0).nonzero().squeeze()

        if len((activity_prop_type == 2).nonzero().squeeze().size()) == 0:
            bg_indexer = (activity_prop_type == 2).nonzero().squeeze(0)
        else:
            bg_indexer = (activity_prop_type == 2).nonzero().squeeze()

        fg_acc = accuracy(activity_out[fg_indexer, :], activity_target[fg_indexer])
        fg_accuracies.update(fg_acc[0].item(), len(fg_indexer))

        if len(bg_indexer) > 0:
            bg_acc = accuracy(activity_out[bg_indexer, :], activity_target[bg_indexer])
            bg_accuracies.update(bg_acc[0].item(), len(bg_indexer))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Act. Loss {act_loss.val:.3f} ({act_loss.avg:.3f})\t'
                  'Comp. Loss {comp_loss.val:.3f} ({comp_loss.avg:.3f})\t'
                  'Act. Accuracy {act_acc.val:.02f} ({act_acc.avg:.2f}) FG {fg_acc.val:.02f} BG {bg_acc.val:.02f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                    act_loss=act_losses, comp_loss=comp_losses, act_acc=act_accuracies,
                    fg_acc=fg_accuracies, bg_acc=bg_accuracies) +
                  '\tReg. Loss {reg_loss.val:.3f} ({reg_loss.avg:.3f})'.format(
                      reg_loss=reg_losses))

    logger.info('Testing Results: Loss {loss.avg:.5f} \t '
          'Activity Loss {act_loss.avg:.3f} \t '
          'Completeness Loss {comp_loss.avg:.3f}\n'
          'Act Accuracy {act_acc.avg:.02f} FG Acc. {fg_acc.avg:.02f} BG Acc. {bg_acc.avg:.02f}'
          .format(act_loss=act_losses, comp_loss=comp_losses, loss=losses, act_acc=act_accuracies,
                  fg_acc=fg_accuracies, bg_acc=bg_accuracies)
          + '\t Regression Loss {reg_loss.avg:.3f}'.format(reg_loss=reg_losses))

    return losses.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    filename = args.snapshot_pref + '_'.join(('PGCN', args.dataset, 'epoch', str(epoch), filename))
    torch.save(state, filename)
    if is_best:
        best_name = args.snapshot_pref + '_'.join(('PGCN', args.dataset, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
