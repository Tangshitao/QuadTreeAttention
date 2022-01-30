#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CHECKPOINT=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox --launcher pytorch ${@:4}
