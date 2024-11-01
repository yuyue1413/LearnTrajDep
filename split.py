import os
import random
import shutil

def split_train_test(dir):
    # 检查目录是否存在
    if not os.path.exists(dir):
        raise Exception("Directory does not exist!")

    # 创建输出目录
    output_dir = dir + "_split_train_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录下的所有文件夹
    class_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    # 遍历每个类别的文件夹
    for class_dir in class_dirs:
        class_dir_path = os.path.join(dir, class_dir)

        # 获取类别文件夹下的所有文件
        files = [f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))]
        print(len(files))
        # 计算测试集数量
        test_size = int(0.2 * len(files))

        # 随机选择测试集文件
        test_files = random.sample(files, test_size)
        print(len(test_files))
        # 创建训练集和测试集目录
        train_dir = os.path.join(output_dir, "train", class_dir)
        test_dir = os.path.join(output_dir, "test", class_dir)
        print(train_dir)
        print(test_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 将文件复制到训练集或测试集目录
        for file in files:
            source_file = os.path.join(class_dir_path, file)
            if file in test_files:
                destination_file = os.path.join(test_dir, file)
            else:
                destination_file = os.path.join(train_dir, file)
            shutil.copy(source_file, destination_file)

if __name__ == "__main__":
    split_train_test('F:\Coding\AlexNet\data')
#按照80%train,20%test划分数据集