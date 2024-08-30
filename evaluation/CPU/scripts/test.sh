#!/bin/bash

SCRIPT=$1
NUM_THREADS_START=$2
NUM_THREADS_END=$3

echo TILE
for i in $(seq $NUM_THREADS_START $NUM_THREADS_END); do ./${SCRIPT} 1 $i TILE; done
echo IMAGE
for i in $(seq $NUM_THREADS_START $NUM_THREADS_END); do ./${SCRIPT} 1 $i IMAGE; done
echo TILES IN IMAGE
for i in $(seq $NUM_THREADS_START $NUM_THREADS_END); do ./${SCRIPT} 1 $i TILES_IN_IMAGE; done

exit 0
