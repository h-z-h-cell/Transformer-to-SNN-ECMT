from torch import Tensor
import torch
import torch.nn as nn

def heaviside0(x: torch.Tensor):
    return ((x>=0)).to(x)

class TwoSideNeuron(nn.Module):
    def __init__(self, scale_p=1., scale_n=1., place=None, times = 3, gap=2):
        super(TwoSideNeuron, self).__init__()
        self.scale_p = scale_p
        self.scale_n = scale_n
        self.place = place
        self.times = times
        self.gap = gap
        self.t = 0
        self.neuron = None
    def forward(self, x):
        if self.t == 0:
            self.neuron = torch.zeros_like(x)
        self.neuron += x
        v_threshold1 = self.scale_p
        v_threshold2 = self.scale_n
        for i in range(self.times - 1):
            v_threshold1 *= self.gap
            v_threshold2 *= self.gap
        fire_list = []
        not_fire = torch.ones_like(x)
        for i in range(self.times):
            tmp1 = heaviside0(self.neuron-v_threshold1+self.scale_p/2)
            tmp2 = heaviside0(-self.neuron-v_threshold2+self.scale_n/2)
            tmp1*=v_threshold1
            tmp2*=-v_threshold2
            tmp1 *= not_fire
            tmp2 *= not_fire
            not_fire[tmp1!=0] =0
            not_fire[tmp2!=0] =0
            self.neuron -= tmp1
            self.neuron -= tmp2
            fire_list.append(tmp1)
            fire_list.append(tmp2)
            v_threshold1 /= self.gap
            v_threshold2 /= self.gap
        self.t += 1
        return fire_list
    def reset(self):
        self.t = 0
        self.neuron=None

#Modify the multi-threshold neuron parameters within this function
def get_data_from_place(place,p,n,args):
    sp = p
    sn = n
    gap = 2
    if 'vit' in args.model and place == 'fc2':
        sp = 0.25
        sn = 0.08
    times = args.linear_num
    if place == 'q' or place == 'k' or place == 'q':
        times = args.qkv_num
    if place == 's':
        times = args.softmax_num
        sp = args.softmax_p
        sn = args.softmax_p
    return sp,sn,times,gap
        
def replace_testneuron_by_twosideneuron(model,args):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_testneuron_by_twosideneuron(module,args)
        if 'testneuron' == module.__class__.__name__.lower():
            place = model._modules[name].place
            scale_p,scale_n,times,gap = get_data_from_place(place,float(model._modules[name].scale_p[0]),float(model._modules[name].scale_n[0]),args)
            print(name,scale_p,scale_n,place)
            model._modules[name] = TwoSideNeuron(scale_p=scale_p,scale_n=scale_n,place=place,times=times,gap=gap)
    return model

class MyTestPlace(nn.Module):
    def __init__(self,place=None):
        super(MyTestPlace, self).__init__()
        self.place = place
    def forward(self, x, *args):
        return [x,]

class TestNeuron(nn.Module):
    def __init__(self,place=None,percent=None):
        super(TestNeuron, self).__init__()
        self.place = place
        self.percent = percent
        self.num = 0
        self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.]))
        self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.]))
    #choose threshold
    def forward(self, x, times=2,gap=3,show=0, tmptime=0,scaletimes=1):
        x2 = x.reshape(-1)
        threshold = torch.kthvalue(x2, int(self.percent * x2.numel()), dim=0).values.item()
        self.scale_p = torch.nn.Parameter((self.scale_p*self.num+threshold)/(self.num+1))
        threshold = -torch.kthvalue(x2, int((1-self.percent) * x2.numel()), dim=0).values.item()
        self.scale_n = torch.nn.Parameter((self.scale_n*self.num+threshold)/(self.num+1))
        self.num+=1
        print(self.scale_p,self.scale_n)
        return [x,]
    def reset(self):
        pass

#replace testplace to testneuron to caculate thresholds
def replace_test_by_testneuron(model,percent=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_test_by_testneuron(module,percent)
        if module.__class__.__name__.lower() == 'mytestplace':
            model._modules[name] = TestNeuron(place=model._modules[name].place,percent=percent)
    return model

class exp_comp_neuron(nn.Module):
    def __init__(self,func,*args,**keywords):
        super(exp_comp_neuron,self).__init__(*args,**keywords)
        self.tot = None
        self.func = func
        self.t=0
    def forward(self, x):
        if self.tot==None:
            last=torch.zeros_like(x)
            self.tot=x.clone()
        else:
            last = self.func(self.tot/self.t)*self.t
            self.tot+=x
        self.t+=1
        now = self.func(self.tot/self.t)*self.t
        return now-last
    def reset(self):
        self.tot = None
        self.t=0

def replace_nonlinear_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_nonlinear_by_neuron(module)
        if 'softmax' in module.__class__.__name__.lower() or 'gelu' in module.__class__.__name__.lower() or 'layernorm' in module.__class__.__name__.lower():
            model._modules[name] = exp_comp_neuron(func=model._modules[name])
    return model


class MyAt(nn.Module):
    def __init__(self):
        super(MyAt, self).__init__()
    def forward(self, x, y):
        return x @ y

class AtNeuron(nn.Module):
    def __init__(self):
        super(AtNeuron, self).__init__()
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t=0
    def forward(self, x,y):
        if self.t == 0:
            self.tot_a=x
            self.tot_b=y
            self.tot_t = x@y
            self.t = 1
            return x@y
        else:
            self.tot_t+= x@y+x@self.tot_b+self.tot_a@y
            self.tot_a+=x
            self.tot_b+=y
            self.t += 1
            return (x@self.tot_b+self.tot_a@y-x@y)/(self.t-1)-self.tot_t/(self.t*(self.t-1))

    def reset(self):# Reset the accumulator
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t=0

def replace_at_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_at_by_neuron(module)
        if module.__class__.__name__.lower()=="myat":
            model._modules[name] = AtNeuron()
    return model

def get_modules(nowname,model):
    flag = 0
    for name, module in model._modules.items():
        if flag==0:
            print(nowname,end=' ')
            flag=1
        print(module.__class__.__name__.lower(),end=' ')
    if flag==1:
        print('')
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = get_modules(name,module)
    return model

def reset_net(model):#initialize all neurons
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'neuron' in module.__class__.__name__.lower():
            module.reset()
    return model

class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)
    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()
    def enable(self):
        self._enable = True
    def disable(self):
        self._enable = False
    def is_enable(self):
        return self._enable
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    def __del__(self):
        self.remove_hooks()

class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module,type=1):
        super().__init__()
        for name, m in net.named_modules():
            # calculated energy consumption
            if type ==1 and m.__class__.__name__.lower() == 'linear':
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1(name)))
            if type ==1 and m.__class__.__name__.lower() == 'atneuron':
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1_2(name)))
            # calculate fire rate of neurons
            if type ==2 and m.__class__.__name__.lower() == 'twosideneuron':
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook2(name)))

    
    def cal_sop1(self, x: Tensor, m: nn.Linear):
        y = torch.zeros_like(x)
        y[x!=0]=1
        with torch.no_grad():
            out0 = (torch.nn.functional.linear(y, torch.ones_like(m.weight), torch.zeros_like(m.bias))).sum()
            sum0 = (torch.nn.functional.linear(torch.ones_like(y), torch.ones_like(m.weight), torch.zeros_like(m.bias))).sum()
            return out0,sum0

    def create_hook1(self, name):
        def hook1(m: nn.Linear, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop1(x[0], m))
        return hook1
                             
    def cal_sop1_2(self, A: Tensor,B: Tensor, m: AtNeuron):
        # print(A.shape,B.shape)
        tmp_A = torch.ones_like(A)
        tmp_B = torch.ones_like(B)
        sum0 = (tmp_A@tmp_B).sum()
        tmp_A = torch.zeros_like(A)
        tmp_B = torch.zeros_like(B)
        tmp_A[A!=0]=1
        tmp_B[B!=0]=1
        tmp_As = torch.zeros_like(m.tot_a)
        tmp_As[m.tot_a!=0]=1
        tmp_Bs = torch.zeros_like(m.tot_b)
        tmp_Bs[m.tot_b!=0]=1
        out01 = (tmp_A@tmp_B).sum()
        out02 = (tmp_A@tmp_Bs).sum()
        out03 = (tmp_As@tmp_B).sum()
        out0 = out01+out02+out03
        # print(round(float(out01/sum0),5),round(float(out02/sum0),5),round(float(out03/sum0),5))
        return out0,sum0

    def create_hook1_2(self, name):
        def hook1_2(m: AtNeuron, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop1_2(x[0],x[1], m))
        return hook1_2                   
    
    def cal_sop2(self, x):
        num_elements = x[0].numel()
        tmp = []
        for index,i in enumerate(x):
            tmp.append((i!= 0).sum())
        return sum(tmp),num_elements,tmp

    def create_hook2(self, name):
        def hook2(m: TwoSideNeuron, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop2(y))
        return hook2