import os
import numpy as np
from osgeo import gdal
import warnings
from tqdm import tqdm
import sys
from PIL import Image


def process_mask_block(block):
    """处理掩膜数据块：非零值设为255"""
    # 处理多维度数组（单波段和多波段情况）
    if len(block.shape) == 2:
        return np.where(block != 0, 255, 0).astype(np.uint8)
    else:
        # 多波段处理：每个波段单独处理
        processed = np.zeros_like(block, dtype=np.uint8)
        for b in range(block.shape[0]):
            processed[b] = np.where(block[b] != 0, 255, 0)
        return processed


def crop_tif_to_patches(img_paths, output_dir, mask_path=None):
    """
    将多个TIFF图像裁剪为256x256的小块

    参数:
    img_paths (list): 需要裁剪的TIFF文件路径列表
    output_dir (str): 输出目录路径
    mask_path (str): [可选] 指定道路掩膜文件路径，进行特殊处理
    """
    # 创建分类输出目录
    output_dirs = {
        'A': os.path.join(output_dir, 'A'),  # 2012图像
        'B': os.path.join(output_dir, 'B'),  # 2014图像
        'label': os.path.join(output_dir, 'label')  # 掩膜
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 存储所有图像的基本信息
    img_info = []
    roles = {}  # 记录每个文件的角色

    # 规范化掩膜路径以便比较
    if mask_path:
        mask_path = os.path.abspath(mask_path)

    # 步骤1: 验证所有图像的尺寸一致性
    base_width, base_height = None, None
    for i, path in enumerate(img_paths):
        ds = gdal.Open(path)
        if ds is None:
            raise RuntimeError(f"无法打开文件: {path}")

        width = ds.RasterXSize
        height = ds.RasterYSize

        # 确定文件角色
        if mask_path and os.path.abspath(path) == mask_path:
            role = 'label'
        elif i == 0:  # 第一个文件默认为2012图像
            role = 'A'
        else:  # 第二个文件默认为2014图像
            role = 'B'
        roles[path] = role
        base_width, base_height = width, height
        # # 验证尺寸一致性 (使用第一个文件作为基准)
        # if base_width is None and base_height is None:
        #     base_width, base_height = width, height
        # elif width != base_width or height != base_height:
        #     raise ValueError(f"图像尺寸不一致: {path} ({width}x{height}) 与基准尺寸 ({base_width}x{base_height}) 不同")

        # 检测是否为掩膜文件
        is_mask = (role == 'label')

        img_info.append({
            'ds': ds,
            'path': path,
            'filename': os.path.basename(path),
            'width': width,
            'height': height,
            'bands': ds.RasterCount,
            'dtype': gdal.GetDataTypeName(ds.GetRasterBand(1).DataType),
            'geotrans': ds.GetGeoTransform(),
            'proj': ds.GetProjection(),
            'is_mask': is_mask,
            'role': role
        })

    # 步骤2: 计算裁剪位置（考虑边界处理）
    # X方向的起始坐标列表
    x_starts = []
    x = 0
    while x < base_width:
        x_end = x + 256
        if x_end > base_width:
            x = max(0, base_width - 256)  # 从边界回退
        x_starts.append(x)
        if x_end > base_width:
            break
        x = x_end

    # Y方向的起始坐标列表
    y_starts = []
    y = 0
    while y < base_height:
        y_end = y + 256
        if y_end > base_height:
            y = max(0, base_height - 256)  # 从边界回退
        y_starts.append(y)
        if y_end > base_height:
            break
        y = y_end

    print(
        f"图像尺寸: {base_width}x{base_height}, 将切成 {len(x_starts)}x{len(y_starts)}={len(x_starts) * len(y_starts)}个块")

    # 步骤3: 逐个文件裁剪
    for info in img_info:
        ds = info['ds']
        filename = info['filename']
        role = info['role']
        is_mask = info['is_mask']

        # 根据文件类型决定处理逻辑
        file_type = "道路掩膜" if is_mask else "真彩色图像"
        print(f"\n处理文件: {filename} (角色: {role}, {file_type})")

        # 计算裁剪块数
        total_blocks = len(x_starts) * len(y_starts)
        pbar = tqdm(total=total_blocks, desc="裁剪进度", file=sys.stdout)

        # 遍历所有裁剪位置
        for y_start in y_starts:
            for x_start in x_starts:
                # 读取图像块
                block = ds.ReadAsArray(x_start, y_start, 256, 256)

                # 特殊处理掩膜数据
                if is_mask:
                    block = process_mask_block(block)
                    output_dtype = gdal.GDT_Byte  # 强制输出为Byte类型
                else:
                    output_dtype = gdal.GetDataTypeByName(info['dtype'])

                # 创建输出文件名 (y_x.png)
                patch_name = f"{y_start//256}_{x_start//256}_1.png"
                out_path = os.path.join(output_dirs[role], patch_name)

                # 保存为PNG格式
                # 处理单波段数据
                if len(block.shape) == 2:
                    img_data = block.astype(np.uint8)
                # 处理多波段数据
                elif len(block.shape) == 3:
                    # 将通道顺序调整为HxWxC
                    if block.shape[0] == 3:  # RGB图像
                        img_data = np.transpose(block, (1, 2, 0)).astype(np.uint8)
                    else:
                        img_data = block[0].astype(np.uint8)  # 多通道取第一个波段
                else:
                    img_data = block.astype(np.uint8)

                # 创建并保存PNG图像
                img = Image.fromarray(img_data)
                img.save(out_path)

                pbar.update(1)

        pbar.close()
        ds = None  # 关闭原始数据集

    print(f"\n处理完成! 文件已分类保存到以下目录:")
    print(f"2012图像: {output_dirs['A']}")
    print(f"2014图像: {output_dirs['B']}")
    print(f"掩膜标签: {output_dirs['label']}")


if __name__ == "__main__":
    # 用户输入配置
    image1_path = r"F:\BaiduNetdiskDownload\Wuhan road change detection dataset\2014image.tif"  # 2012图像
    image2_path = r"F:\BaiduNetdiskDownload\Wuhan road change detection dataset\2016image.tif"  # 2014图像
    mask_path = r"F:\BaiduNetdiskDownload\Wuhan road change detection dataset\change_label_2014to2016.tif"  # 道路掩膜
    output_directory = r"F:\BaiduNetdiskDownload\road_detection\Wuhan\2012_2014"  # 替换为输出目录路径

    # 执行裁剪操作
    crop_tif_to_patches(
        img_paths=[image1_path, image2_path, mask_path],
        output_dir=output_directory,
        mask_path=mask_path
    )