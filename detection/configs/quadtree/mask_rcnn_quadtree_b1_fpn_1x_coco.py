_base_ = [
    "../_base_/models/mask_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
# optimizer
model = dict(
    pretrained="https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention/b1.pth",
    backbone=dict(type="quadtree_b1", style="pytorch", topks=[32, 32, 32, 32]),
    neck=dict(type="FPN", in_channels=[64, 128, 320, 512], out_channels=256, num_outs=5),
)
# optimizer
optimizer = dict(_delete_=True, type="AdamW", lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
