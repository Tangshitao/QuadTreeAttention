import argparse
import torch
from timm.models import create_model
import pvt
import pvt_v2
import quadtree

try:
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Get FLOPS of a classification model')
    parser.add_argument('model', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim


def li_sra_flops(h, w, dim):
    return 2 * h * w * 7 * 7 * dim

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

def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

    _, H, W = input_shape
   
    if 'quadtree' in model.name:
        stage1 = quadtree_flops(H // 4, W // 4,
                            model.block1[0].attn.dim, scale=4) * len(model.block1)
        stage2 = quadtree_flops(H // 8, W // 8,
                            model.block2[0].attn.dim, scale=3) * len(model.block2)
        stage3 = quadtree_flops(H // 16, W // 16,
                            model.block3[0].attn.dim, scale=2) * len(model.block3)
        stage4 = quadtree_flops(H // 32, W // 32,
                            model.block4[0].attn.dim, scale=1) * len(model.block4)
        
    elif 'li' in model.name:  # calculate flops of PVTv2_li
        stage1 = li_sra_flops(H // 4, W // 4,
                            model.block1[0].attn.dim) * len(model.block1)
        stage2 = li_sra_flops(H // 8, W // 8,
                            model.block2[0].attn.dim) * len(model.block2)
        stage3 = li_sra_flops(H // 16, W // 16,
                            model.block3[0].attn.dim) * len(model.block3)
        stage4 = li_sra_flops(H // 32, W // 32,
                            model.block4[0].attn.dim) * len(model.block4)
    else:  # calculate flops of PVT/PVTv2
        stage1 = sra_flops(H // 4, W // 4,
                        model.block1[0].attn.sr_ratio,
                        model.block1[0].attn.dim) * len(model.block1)
        stage2 = sra_flops(H // 8, W // 8,
                        model.block2[0].attn.sr_ratio,
                        model.block2[0].attn.dim) * len(model.block2)
        stage3 = sra_flops(H // 16, W // 16,
                        model.block3[0].attn.sr_ratio,
                        model.block3[0].attn.dim) * len(model.block3)
        stage4 = sra_flops(H // 32, W // 32,
                        model.block4[0].attn.sr_ratio,
                        model.block4[0].attn.dim) * len(model.block4)
        print(flops_to_string(flops))
        
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000
    )
    model.name = args.model
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    flops, params = get_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
