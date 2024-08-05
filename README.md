# README

## 1. Structure Overview

An 11.8+ cuda environment is recommended

First, install the required libraries in the code directory by running the following command

```
pip install -r requirements.txt
```

The file directory structure should be

```
--datasets
  --cifar-10-batches-py (CIFAR10dataset)
  --cifar-100-python (CIFAR100dataset)
  --val (Imagenet1k dataset)
    --n01440764
    --n01443537
    ....
--models
  --eva_21k_1k_336px_psz14_ema_89p6.pt
  --vit_large_patch16_224.pth
  --vit_base_patch16_224.pth
  --vit_large_patch16_224.pth
  --vit_base_patch16_224.pth
  ...
--code
  --run_class_finetuning.py
  --engine_for_finetuning.py
  --datasets.py
  --model_vit.py
  --model_eva.py
  --trans_utils.py
  --utils.py
  --fintune_model.py
  --requirements.txt
```



## 2. Meaning of parameter

```
#model: The name of the model used
#model_path: The directory of the model used
#data_set: The name of the dataset
#eval_data_path: The directory of the evaluation data
#num_workers: The number of processes when loading the dataset
#input_size: The input dimensions of the image
#nb_class: The number of classes in the dataset
#batch_size: The batch size
#ouput_dir: The directory for the output model
#test_mode: Test type, either ann, for_v, or snn
#test_T: The number of time steps to run
#linear_num: The N value of multi-threshold neurons before the linear layer
#qkv_num: The N value of multi-threshold neurons before the qkv in the matrix multiplication
#softmax_num: The N value of multi-threshold neurons after the softmax layers in the matrix multiplication
#softmax_p: The positive threshold of multi-threshold neurons after the softmax layers in the matrix multiplication
```



## 3. How to start

### models needed to be downloaded

The model we use can be downloaded from the code of the corresponding paper on the web

| model name | dataset name | how to get model(use timm module or download from network)   |
| ---------- | ------------ | ------------------------------------------------------------ |
| ViTS/16    | imagenet1k   | timm.create_model("vit_small_patch16_224", pretrained=True)  |
| ViTB/16    | cifar10      | timm.create_model("hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar10", pretrained=True) |
| ViTB/16    | cifar100     | timm.create_model("hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar100", pretrained=True) |
| ViTB/16    | imagenet1k   | timm.create_model("vit_base_patch16_224", pretrained=True)   |
| ViTL/16    | imagenet1k   | timm.create_model("vit_large_patch16_224", pretrained=True)  |
| EVA        | imagenet1k   | https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt |

### quick start

```
parameter:
    your_data_path:  ../datasets/val
    class_of_dataset:  1000(for ImageNet),10(for CIFAR10)，100(for CIFAR100)
    dataset_name: image_folder(for ImageNet),CIFAR10,CIFAR100
    your_model_name:  vit_small_patch16_224 | vit_base_patch16_224 | 
                      vit_large_patch16_224 | eva_g_patch14
    your_model_path:  ../models/filename.pth
    your_input_size:  224(for vit_small_patch16_224)
    				 ,224(for vit_base_patch16_224)
    				 ,224(for vit_large_patch16_224)
    				 ,336(for eva_g_patch14)
    test mode: ann(for testing ann),for_v(for getting threshold),snn(for testing snn)

command to test ann:
python run_class_finetuning.py --eval_data_path your_data_path --nb_classes classes_of_dataset --data_set dataset_name --model your_model_name --model_path your_model_path --input_size your_input_size --batch_size you_batch_size --test_mode ann

command to get threshold:
python run_class_finetuning.py --eval_data_path your_data_path --nb_classes classes_of_dataset --data_set dataset_name --model your_model_name --model_path your_model_path --input_size your_input_size --batch_size you_batch_size --test_mode for_v

command to test threshold:
python run_class_finetuning.py --eval_data_path your_data_path --nb_classes classes_of_dataset --data_set dataset_name --model your_model_name --model_path your_model_path --input_size your_input_size --batch_size you_batch_size --test_mode snn --test_T time_step

```

**eg. Command to test ViT-S/16 on cifar10**

```
1.download model ViTS/16 on Imagenet dataset
2.set correspongding variable in fintune_model.py
	model_path = '../models/vit_small_patch16_224.pth'
	model_name = 'vit_small_patch16_224_cifar10'
	save_model_name = 'vit_small_patch16_224_cifar10_'+str(i)
and run 'python fintune_model.py' on terminal
3. choose the best model rename as 'vit_tiny_patch16_224_cifar10_test64_best.pth'
4. run the following command to get model with threshold : "python run_class_finetuning.py --eval_data_path ../datasets --nb_classes 10 --data_set CIFAR10 --model vit_tiny_patch16_224_cifar10 --model_path ../models/vit_tiny_patch16_224_cifar10_test64_best.pth --input_size 224 --batch_size 64 --test_mode for_v --savename vit_tiny_cifar10_for_v"
5. run the following command to get test result on CIFAR10: "python run_class_finetuning.py --eval_data_path ../datasets --nb_classes 10 --data_set CIFAR10 --model vit_tiny_patch16_224_cifar10 --model_path ../models/vit_tiny_cifar10_for_v.pth --input_size 224 --batch_size 64 --test_mode snn --test_T 10"
```



## 4. The role of each file

**4.1 run_class_finetuning.py :** This module is primarily responsible for: parsing input parameters, creating the model, loading the dataset, loading parameters, and invoking functions from engine_for_finetuning.py for model testing.

**4.2 datasets.py :** Used to load datasets

**4.3 enine_for_finetuning.py :** It mainly includes evaluate and evaluate_snn two functions

​		***evaluate：***Used to test artificial neural networks (ANNs) or get thresholds for spiking neural networks (SNNs), the cumulative accuracy calculation results will be output and appended to the log_ann.txt document.

​		***evaluate_snn：***Used to test SNNs , the cumulative accuracy calculation results will be output and appended to the log_snn.txt document.

**4.4 model_eva.py&model_vit.py :** The code of the modified model.

**4.5 utils.py : **Some auxiliary tools are used for loading, saving models, and calculating average accuracy.

**4.6 trans_utils.py :**

​		***MyTestPlace：***The class used to return input values in ANN testing

​		***TestNeuron，replace_test_by_testneuron：***The class that evaluates the threshold and  the function that replaces MyTestPlace with TestNeuron.

​		***TwoSideNeuron，replace_testneuron_by_twosideneuron：***The class that calculates the spike output of SNNs and  the function that replaces TestNeuron with TwoSideNeuron.

​		***exp_comp_neuron，replace_nonlinear_by_neuron：***The class that represents the expectation compensation modules and the function that replace some nonlinear modules with this module.

​		***MyAt，AtNeuron，replace_at_by_neuron：***The class that represents the expectation compensation modules  for matrix product and the function that replace matricx product with this module.

​		***get_modules：***Print out modules in the model

​		***reset_net：***Reset all modules in the model

​		***BaseMonitor，SOPMonitor：***Monitor fire rate and calculate energy consumption
