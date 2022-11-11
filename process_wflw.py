# import os.path as osp
# from glob import glob

import mmcv
from mmcv import Config

from mmpose.datasets import build_dataset

config = 'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/' \
    'hrnetv2_w18_wflw_256x256_2.py'

cfg = Config.fromfile(config)

train_dataset = build_dataset(cfg.data.train, dict(test_mode=True))
test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))

for sample in train_dataset:
    image_file = sample['image_file']
    # print(image_file)
    paths = image_file.split('/')
    paths[1] = 'wflw_new'
    new_image_file = '/'.join(paths)
    # print(new_image_file)
    print(f'processing {new_image_file}')

    img = sample['img']
    # print(img.shape)

    mmcv.imwrite(img, new_image_file)

for sample in test_dataset:
    image_file = sample['image_file']
    # print(image_file)
    paths = image_file.split('/')
    paths[1] = 'wflw_new'
    new_image_file = '/'.join(paths)
    # print(new_image_file)
    print(f'processing {new_image_file}')

    img = sample['img']
    # print(img.shape)

    mmcv.imwrite(img, new_image_file)

# print(len(train_dataset))
# print(len(test_dataset))
# print(train_dataset[0]['img'].shape)
# print(train_dataset[0]['scale'])
# print(train_dataset[0]['image_file'])
# print(train_dataset[0]['image_file'].split('/'))

# image_file = train_dataset[1]['image_file']
# print(image_file)
# paths = image_file.split('/')
# paths[1] = 'wflw_new'
# new_image_file = '/'.join(paths)
# print(new_image_file)

# img = train_dataset[1]['img']
# img_ = mmcv.rgb2bgr(img)

# print(img.shape)

# mmcv.imwrite(img, new_image_file)

# pattern = 'data/wflw/images/*/*.jpg'
# images = glob(pattern)
# print(len(images))

# sample = train_dataset[0]
# for sample in train_dataset:
#     image_file = sample['image_file']
#     paths = image_file.split('/')
#     paths[1] = 'wflw_new'
#     new_image_file = '/'.join(paths)
#     print(new_image_file)
