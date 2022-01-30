For environmnet and data setup, please refer to [PVT](https://github.com/whai362/PVT/blob/v2/classification/README.md).

## Training
To train quadtree-B-b0 on ImageNet on a single node with 8 gpus for 300 epochs run:

```
bash dist_train.sh configs/quadtree/b0.py 8 --data-path /path/to/imagenet
```

## Testing
To test quadtree-B-b0 on ImageNet on a single node with 8 gpus run:

```
bash dist_train.sh configs/quadtree/b0.py 8 --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```

## Calculating FLOPS & Params
```
python get_flops.py quadtree_b0
```