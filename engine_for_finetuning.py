import torch
from timm.utils import accuracy
from trans_utils import SOPMonitor
import utils
import numpy as np
from trans_utils import reset_net
from timm.loss import LabelSmoothingCrossEntropy
@torch.no_grad()
def evaluate(data_loader, model, device,args=None, model_without_ddp=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    header = 'Test ANN :'
    model.eval()
    nownum=0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        nownum += 1
        with torch.cuda.amp.autocast():
            output = model(images)[0]
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        #acc of this batch
        print(nownum,":",float(acc1),float(acc5))
        #acc of all batches
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        with open('log_ann.txt','a') as f:
            f.write(str(nownum)+': '+'* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss)+'\n')
        if args.test_mode=='for_v' and nownum%1==0:
            utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_snn(data_loader, model, device,T,args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test SNN :'
    #set monitor
    mon = SOPMonitor(model,type=1)
    if args.monitor==True:
        mon.enable()
    # switch to evaluation mode
    model.eval()
    tot = np.array([0. for i in range(T)])
    length = 0
    nownum = 0
    all_sops = [0 for i in range(T)]
    all_nums = [0 for i in range(T)]
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        with torch.cuda.amp.autocast():
            output = model(images,T=T)
        #output is a list of results from time 1 to T
        acc1_list = []
        for i in range(T):
            acc1,acc5 = accuracy(output[i], target, topk=(1, 5))
            print(float(acc1),end=' ')
            acc1_list.append(float(acc1))
        print('')
        output=output[-1]
        output/=T
        # reset potential of neuron
        reset_net(model)
        # loss = criterion(output, target)
        length += batch_size
        nownum += 1
        # tot records the correct num
        tot += np.array(acc1_list) * batch_size
        # metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
        # static fire rate
        if args.monitor==True:
            now_sop = [0 for i in range(T)]
            now_num = [0 for i in range(T)]
            for index in range(T):
                for name in mon.monitored_layers:
                    sublist = mon[name]
                    if len(sublist)>0:
                        now_sop[index]+=sublist[index][0]
                        now_num[index]+=sublist[index][1]
            for i in range(T):
                all_sops[i]+=now_sop[i]
                all_nums[i]+=now_num[i]
            fire_list = []
            energy_list = []
            for i in range(T):
                tmp = float(sum(all_sops[:i+1])/sum(all_nums[:i+1]))
                fire_list.append(tmp)
                energy_list.append((i+1)*0.9/4.6*tmp)
            print('平均发射率: '+','.join([str(round(i,4)) for i in fire_list]))
            print('总能耗: '+','.join([str(round(i,4)) for i in energy_list]))
            mon.clear_recorded_data()
        with open('log_snn.txt','a') as f:
            f.write(str(nownum)+': '+','.join([str(round(i/length,3)) for i in tot])+'\n')
            if args.monitor==True:
                f.write('平均发射率: '+','.join([str(round(i,4)) for i in fire_list])+'\n')
                f.write('总能耗: '+','.join([str(round(i,4)) for i in energy_list])+'\n')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return tot/length


