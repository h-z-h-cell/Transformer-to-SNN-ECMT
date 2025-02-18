import torch.distributed.launch
import argparse
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from timm.models import create_model
from datasets import build_dataset
from engine_for_finetuning import evaluate,evaluate_snn
import utils
import trans_utils
import model_vit
def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model', default='eva_g_patch14', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,help='images input size')
    parser.add_argument('--nb_classes', default=1000, type=int,help='number of the classification types')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--percent', default=0.99, type=float)
    parser.add_argument('--monitor', default=True, type=bool)

    # Dataset parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_data_path', default='../datasets/val', type=str,help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--data_set', default='image_folder', choices=['CIFAR10','CIFAR100', 'IMNET', 'image_folder'],type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='../models',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--savename', default='test', type=str)
    
    # Mode parameter
    parser.add_argument('--test_mode', default='ann',choices=['ann', 'for_v', 'snn'], help="test mode")
    # parser.add_argument('--gpu_for_use', default=[0,],type=list)
    parser.add_argument('--test_T', default=8,type=int)
    
    # Multi-threshold neuron parameter
    parser.add_argument('--linear_num', default=8,type=int)
    parser.add_argument('--qkv_num', default=8,type=int)
    parser.add_argument('--softmax_num', default=8,type=int)
    parser.add_argument('--softmax_p', default=0.0125,type=float)
    
    known_args, _ = parser.parse_known_args()

    return parser.parse_args()


def main(args):
    args.distributed = False
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    cudnn.benchmark = True
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,sampler=sampler_val,batch_size=int(args.batch_size),num_workers=args.num_workers,pin_memory=True,drop_last=False)
    model = create_model(args.model,pretrained=False,img_size=args.input_size,num_classes=args.nb_classes)
    trans_utils.get_modules('',model)
    if args.test_mode == 'ann':
        pass
    elif args.test_mode == 'for_v':
        pass
    elif args.test_mode == 'snn':
        trans_utils.replace_test_by_testneuron(model)
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        print("Load ckpt from %s" % args.model_path)
        checkpoint_model = None
        for model_key in ['model','module']:
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        utils.load_state_dict(model, checkpoint_model, prefix='')
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print("Batch size = %d" % args.batch_size)
    # args.gpu_for_use = [int(i) for i in args.gpu_for_use]
    # model = torch.nn.DataParallel(model,device_ids=args.gpu_for_use,output_device=0)
    if args.test_mode=='ann':
        test_stats = evaluate(data_loader_val, model, device,args=args,model_without_ddp=model_without_ddp)
    elif args.test_mode=='for_v':
        trans_utils.replace_test_by_testneuron(model,args.percent)
        test_stats = evaluate(data_loader_val, model, device,args=args,model_without_ddp=model_without_ddp)
    elif args.test_mode=='snn':
        trans_utils.replace_nonlinear_by_neuron(model)
        trans_utils.replace_at_by_neuron(model)
        trans_utils.replace_testneuron_by_twosideneuron(model,args)
        test_stats = evaluate_snn(data_loader_val, model, device,args.test_T,args)
if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
    