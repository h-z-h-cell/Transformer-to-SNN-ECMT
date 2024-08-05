import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import create_model
import model_vit
import utils

# 1. 环境准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 2. 加载CIFAR-10数据集

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
transform = transforms.Compose([transforms.Resize(224, interpolation=3),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean, std)])
batch_size = 64
root_path = "../datasets"
train_dataset = datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 加载预训练的ViT模型
model_path = '../models/vit_small_patch16_224.pth'
model_name = 'vit_small_patch16_224_cifar10'
model_dict = torch.load(model_path, map_location='cpu')['model']
print("Load ckpt from %s" % model_path)
model = create_model(model_name,pretrained=False,img_size=224,num_classes=10)
# 冻结除所有层
# for param in model.parameters():
#     param.requires_grad = False
# 修改最后的全连接层，改层不会被冻结
num_ftrs = model.head.in_features
model.head = torch.nn.Linear(num_ftrs, 10)

utils.load_state_dict(model, model_dict, prefix='')

model.to(device)

# 4. 模型微调
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
# 微调模型（这里只示意，实际应包含完整的训练循环）
for i in range(10):
    model.train()
    j = 1
    print(i,'start')
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j%100==0 :
            print(j)
        j=j+1
    # 5. 评估
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)[0]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the CIFAR-10 test set: {100 * correct / total}%')
    save_model_name = 'vit_small_patch16_224_cifar10_'+str(i)
    def save_model(model):
        to_save = {'model': model.state_dict()}
        torch.save(to_save, '../models/'+save_model_name.replace('.','_')+'.pth')
    save_model(model)
