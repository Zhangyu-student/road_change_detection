import torch
from torch.utils.data import DataLoader, Dataset

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler
try:
    from thop import profile, clever_format
except ImportError:
    print("警告：未安装thop库，无法计算MACs。请执行: pip install thop")
    profile = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = cdp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your datasets)
    siam_encoder=True,  # whether to use a siamese encoder
    fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
).to(DEVICE)


def print_model_complexity(model, input_shape=(3, 512, 512)):
    """打印模型参数数量和计算复杂度"""
    # 计算参数量
    params = sum(p.numel() for p in model.parameters())

    # 计算MACs（如果可用）
    if profile:
        dummy_input1 = torch.randn(1, *input_shape).to(DEVICE)
        dummy_input2 = torch.randn(1, *input_shape).to(DEVICE)
        macs, _ = profile(model, inputs=(dummy_input1, dummy_input2), verbose=False)
        macs_str = f", MACs: {macs / 1e9:.2f}G"
    else:
        macs_str = ""

    print(f"模型参数: {params / 1e6:.2f}M{macs_str}")


# 初始模型复杂度
print("=" * 50)
print("初始模型复杂度:")
print_model_complexity(model)
print("=" * 50)

train_dataset = LEVIR_CD_Dataset(r'F:\BaiduNetdiskDownload\road_detection\Wuhan\2012_2014',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir=r'F:\BaiduNetdiskDownload\road_detection\Wuhan\2012_2014/label',
                                 debug=False)

valid_dataset = LEVIR_CD_Dataset('../LEVIR-CD/test',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir='../LEVIR-CD/test/label',
                                 debug=False,
                                 test_mode=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

loss = cdp.utils.losses.CrossEntropyLoss()
metrics = [
    cdp.utils.metrics.Fscore(activation='argmax2d'),
    cdp.utils.metrics.Precision(activation='argmax2d'),
    cdp.utils.metrics.Recall(activation='argmax2d'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = cdp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = cdp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 60 epochs

max_score = 0
MAX_EPOCH = 60

for i in range(MAX_EPOCH):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    scheduler_steplr.step()

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        torch.save(model, './best_model.pth')
        print('Model saved!')

# save results (change maps)
"""
Note: if you use sliding window inference, set: 
    from change_detection_pytorch.datasets.transforms.albu import (
        ChunkImage, ToTensorTest)
    
    test_transform = A.Compose([
        A.Normalize(),
        ChunkImage({window_size}}),
        ToTensorTest(),
    ], additional_targets={'image_2': 'image'})

"""
valid_epoch.infer_vis(valid_loader, save=True, slide=False, save_dir='./res')
