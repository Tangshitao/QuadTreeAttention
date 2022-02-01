For environment and data setup, please refer to [LoFTR](https://github.com/zju3dv/LoFTR).

## Training
The results can be reproduced when training with 8 gpus. Please run the following commands.
```
sh scripts/reproduce_train/indoor_quadtree_ds.sh
```
The parameter top K can be reduced for speeding up. The performance won't drop too much. Please set this parameter in cfg.LOFTR.COARSE.TOPKS.

## Testing
```
sh scripts/reproduce_test/indoor_ds_quadtree.sh
```
