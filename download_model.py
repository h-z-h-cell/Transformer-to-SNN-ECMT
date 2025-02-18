import torch
from timm.models import create_model
import timm

model_name_list = ['vit_small_patch16_224.augreg_in21k_ft_in1k','vit_base_patch16_224.augreg2_in21k_ft_in1k','vit_large_patch16_224.augreg_in21k_ft_in1k']
save_name_list = ['vit_small_patch16_224','vit_base_patch16_224','vit_large_patch16_224']

model_name_list2 = ['eva02_tiny_patch14_336.mim_in22k_ft_in1k', 'eva02_small_patch14_336.mim_in22k_ft_in1k','eva02_base_patch14_448.mim_in22k_ft_in1k',  'eva02_large_patch14_448.mim_in22k_ft_in1k']
save_name_list2 = ['eva02_tiny_patch14_336', 'eva02_small_patch14_336', 'eva02_base_patch14_448', 'eva02_large_patch14_448']

model_name_list_1_2 = ['eva_giant_patch14_336.m30m_ft_in22k_in1k']
save_name_list3_1_2 = ['eva_giant_patch14_336']



# vit_small_patch16_224
def save_model(model, model_name):
    to_save = {'model': model.state_dict()}
    torch.save(to_save, '../models/vit_model/' + model_name.replace('.', '_') + '.pth')

if __name__ == '__main__':
    # model_names = timm.list_models(pretrained=True)
    # print(model_names)
    # # for i in range(len(model_name_list2)):
    # for i in range(0, 2):
    #     model = create_model(model_name_list2[i], pretrained=True)
    #     save_model(model, save_name_list2[i])
    #     print("download",model_name_list2[i],"successfully")

    import timm
    model = timm.create_model(
        'eva02_base_patch14_448',
        pretrained=False,
        num_classes=0,  # remove classifier nn.Linear
    )
    print(model)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(transforms)
    print(timm.data.IMAGENET_DEFAULT_STD, timm.data.OPENAI_CLIP_MEAN)