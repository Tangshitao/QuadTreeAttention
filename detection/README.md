For environmnet and data setup, please refer to [PVT](https://github.com/whai362/PVT/blob/v2/detection/README.md).

## Data
```
mkdir data&&ln -s /path/to/coco data/coco
```

## Training
To train quadtree-B-b0 + RetinaNet (640x) on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
bash dist_train.sh configs/quadtree/retinanet_quadtree_b0_fpn_1x_coco.py 8
```

## Testing
To test quadtree-B-b0 + RetinaNet (640x) on COCO val2017:

```
bash dist_test.sh configs/quadtree/retinanet_quadtree_b0_fpn_1x_coco.py 8 /path/to/checkpoint
```
