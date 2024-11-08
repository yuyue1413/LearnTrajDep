import torch
from torch import nn

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        # 是一个容器将多个层（如卷积层、激活函数、池化层等）组合在一起，形成一个有序的序列，负责特征提取，处理原始输入并提取出有用的信息
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # inplace=True表示在计算过程中直接修改输入张量，这样可以节省内存
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二层
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三层
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 第四层
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 第五层
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 是一个容器，用于将多个层（如全连接层、激活函数等）按顺序组合在一起，负责分类，将提取的特征映射到具体的类别标签。
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=1000, out_features=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)  # 展平卷积层的输出
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)
    print(y.size())