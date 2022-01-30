import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor
import torch
import pvt
import quadtree
import pvt_v2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim, num_heads=1):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2

def quadtree_flops(h, w, dim, scale=4, topk=8):
    flops=0
    for i in range(scale):
        if i!=scale-1:
            flops+=h*w*topk*4*dim*2
        else:
            flops+=h*w*h*w*dim*2
        h=h//2
        w=w//2
    return flops

def get_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    backbone=net.backbone
    backbone_name = type(net.backbone).__name__
    

    _, H, W = input_shape
    if 'li' in backbone_name:  # calculate flops of PVTv2_li
        stage1 = li_sra_flops(H // 4, W // 4,
                                backbone.block1[0].attn.dim) * len(backbone.block1)
        stage2 = li_sra_flops(H // 8, W // 8,
                                backbone.block2[0].attn.dim) * len(backbone.block2)
        stage3 = li_sra_flops(H // 16, W // 16,
                                backbone.block3[0].attn.dim) * len(backbone.block3)
        stage4 = li_sra_flops(H // 32, W // 32,
                                backbone.block4[0].attn.dim) * len(backbone.block4)
    elif 'quadtree' in backbone_name:
        stage1 = quadtree_flops(H // 4, W // 4,
                                backbone.block1[0].attn.dim, scale=4) * len(backbone.block1)
        stage2 = quadtree_flops(H // 8, W // 8,
                                backbone.block2[0].attn.dim, scale=3) * len(backbone.block2)
        stage3 = quadtree_flops(H // 16, W // 16,
                                backbone.block3[0].attn.dim, scale=2) * len(backbone.block3)
        stage4 = quadtree_flops(H // 32, W // 32,
                                backbone.block4[0].attn.dim, scale=1) * len(backbone.block4)
    else:  # calculate flops of PVT/PVTv2
        stage1 = sra_flops(H // 4, W // 4,
                            backbone.block1[0].attn.sr_ratio,
                            backbone.block1[0].attn.dim) * len(backbone.block1)
        stage2 = sra_flops(H // 8, W // 8,
                            backbone.block2[0].attn.sr_ratio,
                            backbone.block2[0].attn.dim) * len(backbone.block2)
        stage3 = sra_flops(H // 16, W // 16,
                            backbone.block3[0].attn.sr_ratio,
                            backbone.block3[0].attn.dim) * len(backbone.block3)
        stage4 = sra_flops(H // 32, W // 32,
                            backbone.block4[0].attn.sr_ratio,
                            backbone.block4[0].attn.dim) * len(backbone.block4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # from IPython import embed; embed()
    if hasattr(model.backbone, 'block1'):
        print('#### get transformer flops ####')
        with torch.no_grad():
            flops, params = get_tr_flops(model, input_shape)
    else:
        print('#### get CNN flops ####')
        flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()