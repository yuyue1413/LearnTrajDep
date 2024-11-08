import random

import torch
from net import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

BOOT_TEST = r'E:\Coding\GitHub\AlexNet\data\test'



# 1、定义一个数据预处理的转换
# 将图像的像素值归一化到[-1, 1]之间
# normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片尺寸
    # transforms.RandomVerticalFlip(),  # 以一定的概率随机垂直翻转图像，一种数据增强技术，增加训练数据的多样性
    transforms.ToTensor()  # 将图像转换为张量，并将像素值缩放到[0, 1]的范围
    # normalize  # 归一化
])

test_dataset = ImageFolder(root=BOOT_TEST, transform=test_transforms)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyAlexNet().to(device)



# 上面都是对数据的处理，下面的是测试的代码
# 加载模型
# torch.load()函数加载指定路径的模型权重文件，load_state_dict()方法将权重加载到模型
model.load_state_dict(torch.load("E:\\Coding\\GitHub\\AlexNet\\save_model\\last_model.pth"))

# 获取预测结果
# 定义一个类名列表classes，用于映射模型预测的索引到实际的类名
classes = ["cat", "dog"]

# 把张量转化为照片的格式
# 创建一个ToPILImage实例，用于将Pytorch张量转换成PIL图像格式，便于可视化
show = ToPILImage()

# 随机选择 10 张图片进行测试
num_samples = 10
indices = random.sample(range(len(test_dataset)), num_samples)  # 随机选择索引

# 初始化正确预测计数
correct_predictions = 0

# 进入到验证阶段
model.eval()  # 将模型设置为评估模式。这会关闭 Dropout 和 BatchNorm 的训练行为，使模型以推理模式运行
for i in indices:
    x, y = test_dataset[i][0], test_dataset[i][1]
    # show(x).show()  # 将张量 x 转换为图像格式并显示出来。这样用户可以直观地看到模型正在处理的图像

    # to(device)已经知道是将其移动到GPU，unsqueeze(x, dim=0)表示在第0为加一个维度，即将一张图像转换成一个批次的形式比如原本是(224, 224, 3)，执行后就变成(224, 224, 3, 32)
    # x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    x = torch.unsqueeze(x, dim=0).float().to(device)
    # x = torch.tensor(x).to(device)

    with torch.no_grad():  # 禁用梯度计算
        pred = model(x)  # 将输入图像 x 传递给模型，得到预测输出pred，pred
        pridicted = classes[torch.argmax(pred[0])]  # torch.argmax(pred[0]) 获取概率最大的类别索引，如果猫概率最大，则索引为0
        actual = classes[y]
        print(f'pridicted: {pridicted}, actual: {actual}')
        # print(pred)

        # 统计正确预测
        if pridicted == actual:
            correct_predictions += 1

# 计算并打印预测准确率
accuracy = correct_predictions / num_samples
print(f'Accuracy: {accuracy * 100:.2f}%')
