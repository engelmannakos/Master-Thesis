import time
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.PRSA import PRSA_Net
from model.loss_function import Criterion
from dataset import VideoDataset, HangtimeDataset
from engine import train_epoch, test


def Train(args):
    model = PRSA_Net(
        batch_size=args.scheme['batch_size'],
        dataset_name=args.dataset['dataset_name'],
        temporal_scale=args.dataset['temporal_scale'],
        max_duration=args.dataset['max_duration'],
        min_duration=args.dataset['min_duration'],
        prop_boundary_ratio=args.model['prop_boundary_ratio'],
        num_sample=args.model['num_sample'],
        num_sample_perbin=args.model['num_sample_perbin'],
        feat_dim=args.model['feat_dim']
    )
    criterion = Criterion(tscale=args.dataset['temporal_scale'], duration=args.dataset['max_duration'])

    model = torch.nn.DataParallel(model, ).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.scheme['training_lr'], weight_decay=args.scheme['weight_decay'])

    if args.dataset['dataset_name'] != 'thumos14':

        anno_split = args.anno_json[args.sbj]
        """
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']

        if args.has_null:
            args.labels = ['null'] + list(file['label_dict'])
        else:
            args.labels = list(file['label_dict'])

        args.label_dict = dict(zip(args.labels, list(range(len(args.labels)))))
        #train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        #val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        """
        
        print('Split {} / {}, but now we\'ll run it for only for one subject.'.format(args.sbj+1, len(args.anno_json)))
        



        train_loader = DataLoader(
            HangtimeDataset(
                temporal_scale=args.dataset['temporal_scale'],
                mode=args.mode,
                subset="train",
                feat_dim=args.model['feat_dim'],
                sens_folder=args.dataset['sens_folder'],
                json_anno=anno_split,
                gap_videoframes=args.dataset['gap_videoframes'],
                max_duration=args.dataset['max_duration'],
                min_duration=args.dataset['max_duration'],
                feature_name=args.dataset['feature_name'],
                overwrite=args.dataset['overwrite'],
                sbj=args.sbj
            ),
            batch_size=args.scheme['batch_size'],
            shuffle=True,
            num_workers=args.dataset['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            HangtimeDataset(
                temporal_scale=args.dataset['temporal_scale'],
                mode=args.mode,
                subset="val",
                feat_dim=args.model['feat_dim'],
                sens_folder=args.dataset['sens_folder'],
                json_anno=anno_split,
                gap_videoframes=args.dataset['gap_videoframes'],
                max_duration=args.dataset['max_duration'],
                min_duration=args.dataset['max_duration'],
                feature_name=args.dataset['feature_name'],
                overwrite=args.dataset['overwrite'],
                sbj=args.sbj
            ),
            batch_size=args.scheme['batch_size'],
            shuffle=False,
            num_workers=args.dataset['num_workers'],
            pin_memory=True
        )
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheme['step_size'], gamma=args.scheme['step_gamma'])
        print('Loaders are ready. Let\'s start the training.')
        for epoch in range(args.scheme['train_epoch']):
            print("epoch start time:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            train_epoch(train_loader, model, criterion, optimizer, epoch)
            test(args, test_loader, model, criterion, epoch)
            scheduler.step()
        



        #! For now I only run it for 1 subject
        #break
            

        # results...


    else: #! The old way
        train_loader = DataLoader(
            VideoDataset(
                temporal_scale=args.dataset['temporal_scale'],
                mode=args.mode,
                subset="train",
                feature_path=args.dataset['feature_path'],
                video_info_path=args.dataset['video_info_path'],
                feat_dim=args.model['feat_dim'],
                gap_videoframes=args.dataset['gap_videoframes'],
                max_duration=args.dataset['max_duration'],
                min_duration=args.dataset['max_duration'],
                feature_name=args.dataset['feature_name'],
                overwrite=args.dataset['overwrite']
            ),
            batch_size=args.scheme['batch_size'],
            shuffle=True,
            num_workers=args.dataset['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            VideoDataset(
                temporal_scale=args.dataset['temporal_scale'],
                mode=args.mode,
                subset="val",
                feature_path=args.dataset['feature_path'],
                video_info_path=args.dataset['video_info_path'],
                feat_dim=args.model['feat_dim'],
                gap_videoframes=args.dataset['gap_videoframes'],
                max_duration=args.dataset['max_duration'],
                min_duration=args.dataset['max_duration'],
                feature_name=args.dataset['feature_name'],
                overwrite=args.dataset['overwrite']
            ),
            batch_size=args.scheme['batch_size'],
            shuffle=False,
            num_workers=args.dataset['num_workers'],
            pin_memory=True
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheme['step_size'], gamma=args.scheme['step_gamma'])

        # bm_mask = get_mask(args.dataset['temporal_scale'], args.dataset['max_duration'])

        for epoch in range(args.scheme['train_epoch']):
            print("epoch start time:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            train_epoch(train_loader, model, criterion, optimizer, epoch)
            test(args.output, test_loader, model, criterion, epoch)
            scheduler.step()