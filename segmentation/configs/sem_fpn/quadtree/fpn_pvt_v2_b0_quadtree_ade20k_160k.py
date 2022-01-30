_base_ = [
    "../../_base_/models/fpn_r50.py",
    "../../_base_/datasets/ade20k.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k_adamw.py",
]
# model settings

norm_cfg = dict(type="SyncBN", requires_grad=True)
# model settings
model = dict(
    type="EncoderDecoder",
    pretrained="https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention/b0.pth",
    backbone=dict(type="quadtree_b0", style="pytorch"),
    neck=dict(in_channels=[32, 64, 160, 256]),
    decode_head=dict(num_classes=150),
)


optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={"pos_block": dict(decay_mult=0.0), "norm": dict(decay_mult=0.0), "head": dict(lr_mult=10.0)}
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)


data = dict(samples_per_gpu=2)
evaluation = dict(interval=16000, metric="mIoU")
