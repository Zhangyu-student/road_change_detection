import os
import shutil
import random


def split_change_detection_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    分割变化检测数据集为训练集、验证集和测试集（使用随机种子确保可复现）

    参数:
    data_dir: 数据集根目录路径
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    seed: 随机种子（默认为42）
    """
    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"

    # 设置随机种子确保可复现性
    random.seed(seed)

    # 原始数据目录
    a_dir = os.path.join(data_dir, 'A')
    b_dir = os.path.join(data_dir, 'B')
    label_dir = os.path.join(data_dir, 'label')

    # 获取所有文件名（假设A、B和label中的文件名一一对应）
    filenames = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]

    # 随机打乱文件名顺序（使用固定种子）
    random.shuffle(filenames)

    # 计算各集合的数量
    total_count = len(filenames)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    # 创建目标目录
    for split in ['train', 'val', 'test']:
        for folder in ['A', 'B', 'label']:
            os.makedirs(os.path.join(data_dir, split, folder), exist_ok=True)

    # 分割并复制文件
    for i, filename in enumerate(filenames):
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'

        # 复制A文件夹中的文件
        src_path = os.path.join(a_dir, filename)
        dst_path = os.path.join(data_dir, split, 'A', filename)
        shutil.copy2(src_path, dst_path)

        # 复制B文件夹中的文件
        src_path = os.path.join(b_dir, filename)
        dst_path = os.path.join(data_dir, split, 'B', filename)
        shutil.copy2(src_path, dst_path)

        # 复制label文件夹中的文件
        src_path = os.path.join(label_dir, filename)
        dst_path = os.path.join(data_dir, split, 'label', filename)
        shutil.copy2(src_path, dst_path)

    print(f"数据集分割完成！(随机种子: {seed})")
    print(f"总样本数: {total_count}")
    print(f"训练集: {train_count} 个样本")
    print(f"验证集: {val_count} 个样本")
    print(f"测试集: {test_count} 个样本")


# 使用示例
if __name__ == "__main__":
    data_dir = r"F:\BaiduNetdiskDownload\road_detection\Wuhan\2012_2014"
    split_change_detection_dataset(data_dir, seed=42)  # 使用固定随机种子