import argparse

ds = 'wetlab'
feat_ext = 'raw'


#argumentation setting
def make_parser():
    parser = argparse.ArgumentParser(description='hyperparameter options', add_help=False)

    if ds == 'thumos14':
        parser.add_argument('--video_anno', default='../data/thumos14_v2.json', type=str)
        parser.add_argument('--video_len_file', default='../output/video_len_{}.json', type=str)
        parser.add_argument('--num_of_class', default=21, type=int)
        parser.add_argument('--video_feature_all_train', default='../data/thumos_all_feature_val_V3.pickle', type=str)
        parser.add_argument('--video_feature_all_test', default='../data/thumos_all_feature_test_V3.pickle', type=str)
        parser.add_argument('--ontal_label_file', default="../output/label_{}_{}_{}_{}_{}_{}_{}.h5", type=str)
        parser.add_argument('--feat_dim', default=4096, type=int)
        parser.add_argument('--dataset', default='thumos14', type=str)

    else:
        parser.add_argument('--video_anno', default='../data/{}/annotations/loso_sbj_{}.json', type=str) #!
        parser.add_argument('--video_len_file', default='../output/{}/{}/sbj_{}/video_len_{}.json', type=str)
        parser.add_argument('--video_feature_all_train', default='../data/{}/pickles/{}/sbj_{}/train_features.pickle', type=str)#!
        parser.add_argument('--video_feature_all_test', default='../data/{}/pickles/{}/sbj_{}/test_features.pickle', type=str)#!
        parser.add_argument('--ontal_label_file', default="../output/{}/{}/sbj_{}/label_{}_{}_{}_{}_{}_{}_{}_{}.h5", type=str)
        parser.add_argument('--dataset', default=ds, type=str) #!
        parser.add_argument('--feat_ext', default=feat_ext, type=str) #!
        if feat_ext == 'raw':
            if ds in ['hangtime', 'sbhar', 'wetlab']:
                parser.add_argument('--feat_dim', default=3, type=int) #!
                if ds == 'hangtime':
                    parser.add_argument('--num_of_class', default=5, type=int) #!
                elif ds == 'sbhar':
                    parser.add_argument('--num_of_class', default=12, type=int) #!
                elif ds == 'wetlab':
                    parser.add_argument('--num_of_class', default=8, type=int) #!
            elif ds == 'opportunity':
                parser.add_argument('--feat_dim', default=113, type=int) #!
                parser.add_argument('--num_of_class', default=17, type=int) #!
                parser.add_argument('--hidden_dim', default=1024, type=int)
            elif ds == 'wear':
                parser.add_argument('--feat_dim', default=12, type=int) #!
                parser.add_argument('--num_of_class', default=18, type=int) #!
            elif ds == 'rwhar':
                parser.add_argument('--feat_dim', default=21, type=int) #!
                parser.add_argument('--num_of_class', default=8, type=int) #!
            else:
                raise KeyError('Invalid dataset!')
        elif feat_ext == 'lstm':
            ...
        elif feat_ext == 'attention':
            ...
        else:
            raise KeyError('Invalid feature extraction!')

    # dataset
    parser.add_argument('--num_frame', default=64, type=int) #! change
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--flow', action='store_true')

    parser.add_argument('--num_queries', default=10, type=int)
    
    parser.add_argument('--p_videos', default=1, type=int, help='number of parallel operation of videos')
    
    parser.add_argument('--detect_len', default=16, type=int)
    parser.add_argument('--anti_len', default=16, type=int)

    # model
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--training', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--ffn_dim', default=2048, type=int) #2048
    parser.add_argument('--e_nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--d_nheads', default=4, type=int)
    parser.add_argument('--dec_layers', default=5, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--activation', default="gelu", type=str)
    parser.add_argument('--max_memory_len', default=3, type=int) #! change

    parser.add_argument('--memory_sampler', default='gap2', type=str, help='[gapN or all]')
    
    parser.add_argument('--use_flag', action='store_true')
    parser.add_argument('--flag_threshold', default=0.5, type=float)

    # training

    if ds == 'thumos14':
        parser.add_argument('--epochs', default=100, type=int)
    else:
        parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=64, type=int) #! change
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--min_lr', default=1e-8, type=float)
    parser.add_argument('--max_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_gamma', default=0.9, type=float)
    parser.add_argument('--lr_Tup', default=3, type=int)
    parser.add_argument('--lr_Tcycle', default=10, type=int)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.3, type=float)    
    
    # criterion
    parser.add_argument('--use_empty_weight', action='store_true')
    parser.add_argument('--eos_coef', default=1e-8, type=float)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--reduce', default=1, type=int)
    parser.add_argument('--cls_threshold', default=0.1, type=float)
    
    parser.add_argument('--cls_coef', default=1, type=int)
    parser.add_argument('--flag_coef', default=1, type=int)
    parser.add_argument('--reg_l1_coef', default=1, type=int)
    parser.add_argument('--reg_diou_coef', default=1, type=int)
    parser.add_argument('--reg_stcls_coef', default=1, type=int) 
    
    # evaluation
    parser.add_argument('--nms_threshold', default=0.3, type=float)
    
    # output
    parser.add_argument('--make_output', action='store_true')
    parser.add_argument('--proposal_path', default='proposal_{}_{}_{}', type=str)
    
    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--code_testing', action='store_true')
    
    # default setting
    parser.add_argument('--subject', default=0, type=int) #! new
    parser.add_argument('--task', default="ontal", type=str)
    parser.add_argument('--random_seed', default=52, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--train_eval_step', default=5, type=int)
    parser.add_argument('--test_eval_step', default=1, type=int)
    
    # GPU
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--num_workers', default=4, type=int)
     
    return parser.parse_args()
