For environmnet and data setup, please refer to [PVT](https://github.com/whai362/PVT/blob/v2/segmentation/README.md).

## Training
To train quadtree-B-b0 + Semantic FPN on a single node with 8 gpus run:

```
bash dist_train.sh configs/sem_fpn/quadtree/fpn_pvt_v2_b0_quadtree_ade20k_160k.py 8
```

## Testing
To test quadtree-B-b0 + Semantic FPN on a single node with 8 gpus run:

```
bash dist_test.sh configs/sem_fpn/quadtree/fpn_pvt_v2_b0_quadtree_ade20k_160k.py 8 /path/to/checkpoint
```
