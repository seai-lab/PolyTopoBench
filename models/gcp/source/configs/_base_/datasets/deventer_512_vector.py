# dataset settings
import os

dataset_type = 'PolyTopoBenchVectorDataset'
data_root = os.environ.get('GCP_DEVENTER_DATA_ROOT', '../../data_processed/deventer_512_valtest_as_val/gcp/road')
train_ann_file = os.environ.get('GCP_DEVENTER_TRAIN_ANN', 'train/annotation.json')
val_ann_file = os.environ.get('GCP_DEVENTER_VAL_ANN', 'val/annotation.json')
test_ann_file = os.environ.get('GCP_DEVENTER_TEST_ANN', 'test/annotation.json')
train_img_prefix = os.environ.get('GCP_DEVENTER_TRAIN_IMG_PREFIX', 'train/images')
val_img_prefix = os.environ.get('GCP_DEVENTER_VAL_IMG_PREFIX', 'val/images')
test_img_prefix = os.environ.get('GCP_DEVENTER_TEST_IMG_PREFIX', 'test/images')
train_batch_size = int(os.environ.get('GCP_DEVENTER_TRAIN_BATCH_SIZE', '24'))
train_num_workers = int(os.environ.get('GCP_DEVENTER_TRAIN_NUM_WORKERS', '8'))
eval_num_workers = int(os.environ.get('GCP_DEVENTER_EVAL_NUM_WORKERS', '4'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
backend_args = None
crop_size = (512, 512)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
        with_poly_json=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Rotate90', prob=0.75),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=True,
        poly2mask=False,
        with_poly_json=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        backend_args=backend_args,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_prefix),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=eval_num_workers,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_prefix),
        test_mode=True,
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=eval_num_workers,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_img_prefix),
        test_mode=True,
    )
)
