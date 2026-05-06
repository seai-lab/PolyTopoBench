import os

_base_ = [
    '../_base_/datasets/deventer_512_vector.py', '../_base_/default_runtime.py',
]
backend_args = None
single_class_name = os.environ.get('GCP_DEVENTER_SINGLE_CLASS', 'road')
singleclass_metainfo = dict(classes=(single_class_name,), palette=[(0, 0, 255)])
data_root = os.environ.get('GCP_DEVENTER_DATA_ROOT', '../../data_processed/deventer_512_valtest_as_val/gcp/road')
val_eval_ann = os.path.join(data_root, os.environ.get('GCP_DEVENTER_VAL_ANN', 'val/annotation.json'))
test_eval_ann = os.path.join(data_root, os.environ.get('GCP_DEVENTER_TEST_ANN', 'test/annotation.json'))

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
)

load_from = os.environ.get('GCP_STAGE1_LOAD_FROM', None)
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='PolyFormerV2',
    data_preprocessor=data_preprocessor,
    test_mode='slide_inference',
    frozen_parameters=[
        'backbone',
        'panoptic_head.pixel_decoder',
        'panoptic_head.transformer_decoder',
        'panoptic_head.decoder_input_projs',
        'panoptic_head.query_embed',
        'panoptic_head.query_feat',
        'panoptic_head.level_embed',
        'panoptic_head.cls_embed',
        'panoptic_head.mask_embed',
    ],
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    panoptic_head=dict(
        type='PolygonizerHeadV20',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=300,
        num_transformer_feat_level=3,
        poly_cfg=dict(
            num_inter_points=64,
            num_primitive_queries=64,
            apply_prim_pred=True,
            step_size=4,
            polygonized_scale=4.,
            max_offsets=5,
            use_coords_in_poly_feat=True,
            use_decoded_feat_in_poly_feat=True,
            use_point_feat_in_poly_feat=True,
            point_as_prim=True,
            pred_angle=False,
            prim_cls_thre=0.1,
            num_cls_channels=2,
            stride_size=64,
            use_ind_offset=True,
            poly_decode_type='dp',
            reg_targets_type='vertice',
            return_poly_json=False,
            use_gt_jsons=False,
            mask_cls_thre=0.0,
            lam=4,
            map_features=True,
            max_align_dis=15,
            align_iou_thre=0.5,
            num_min_bins=32,
            proj_gt=False,
            loss_weight_dp=0.01,
            max_match_dis=10,
            use_ref_rings=False,
            apply_poly_iou_loss=True,
            sample_points=True,
            max_step_size=128,
            polygonize_mode='cv2_single_mask',
            apply_right_angle_loss=False,
            apply_angle_loss=True
        ),
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        dp_polygonize_head=dict(
            return_intermediate=True,
            num_layers=3,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]
        ),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_dice_wn=dict(
            type='DiceLoss',
            use_sigmoid=False,
            activate=False,
            reduction='mean',
            naive_dice=False,
            loss_weight=10),
        loss_poly_reg=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=1.
        ),
        loss_poly_vec=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=10.
        ),
        loss_poly_ts=dict(
            type='MSELoss',
            reduction='mean',
            loss_weight=5.
        ),
        loss_poly_ang=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=1.
        ),
        loss_poly_right_ang=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=0.
        )),
    panoptic_fusion_head=dict(
        type='PolyFormerFusionHeadV2',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        prim_assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='PointL1Cost', weight=0.1),
            ]),
        sampler=dict(type='MaskPseudoSampler'),
        add_target_to_data_samples=True,
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=200,
        iou_thr=0.8,
        filter_low_score=False,
        stride=(400, 400),
        scale_factor=4,
        crop_size=(512, 512),
        crop_up_size=(512, 512),
        out_size_scale=1,
        out_crop_size=(512, 512)
    ),
    init_cfg=None)

train_dataloader = dict(dataset=dict(metainfo=singleclass_metainfo))
val_dataloader = dict(dataset=dict(metainfo=singleclass_metainfo))
test_dataloader = dict(dataset=dict(metainfo=singleclass_metainfo))

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=val_eval_ann,
        metric=['segm'],
        mask_type='polygon',
        backend_args=backend_args,
        calculate_mta=True,
        calculate_iou_ciou=True,
        score_thre=0.5
    )
]
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=test_eval_ann,
        metric=['segm'],
        mask_type='polygon',
        backend_args=backend_args,
        calculate_mta=True,
        calculate_iou_ciou=True,
        score_thre=0.5
    )
]

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

max_epochs = 24
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=2,
        interval=1),
)

visualizer = None
auto_scale_lr = dict(enable=False, base_batch_size=8)
