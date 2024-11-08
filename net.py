import torch
from torch import nn
import torch.nn.functional as F

class MyAlexNet(nn.Module):
    def __init__(self):
        # 下面一条语句解读：这个类继承自nn.Module，使用下面一条语句的作用就是继承了父类后我这个子类需要一些额外的属性
        super(MyAlexNet, self).__init__()

        # 定义激活函数
        self.ReLU = nn.ReLU()

        # 定义网络的每一层，in_channels，out_channels是通道这一维度的大小，即只看通道，不用看宽高`Z!D'nü'g'm'b'h
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.f7 = nn.Linear(in_features=4096, out_features=4096)
        self.f8 = nn.Linear(in_features=4096, out_features=1000)
        self.f9 = nn.Linear(in_features=1000, out_features=2)


    # 重新定义父类的forward函数
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)

        x = self.flatten(x)  # 将卷积层或池化层的输出展平，以便输入到‌全连接层处理

        x = self.f6(x)
        x = F.dropout(x, p=0.5)  # 随机失活，目前认为是一种正则化技术，为了防止过拟合
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x

# if __name__ == "__main__": 是 Python 中常用的一种条件判断语句，主要作用是在当前模块作为程序入口时执行一些特定的代码，而在被其它模块引入时不执行这些特定的代码。
# 也就是说用了if __name__ == “__main__“，在import的时候就不会运行if __name__ == "__main__":中的代码，否则你在执行import的时候会将被import的所有文件都执行。
# 这个例子中的X为四维，因为使用的是随机批量梯度下降，所以第一维是批量大小维，第二维是通道维，后面两维是高宽维
if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)