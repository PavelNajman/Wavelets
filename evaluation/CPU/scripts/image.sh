#!/bin/bash

source utils.sh

NUM_THREADS=$1
TILE_SIZE=$2
METHOD=$3

if [ -z "$1" ] 
then
	echo "Usage: ./image.sh NUM_THREADS [TILE_SIZE] [METHOD]"
	exit 1
fi

if [ -z "$2" ]
then
	TILE_SIZE=512
fi

if [ -z "$3" ]
then
	METHOD=0
fi

MAX_IMAGE_SIZE=16384

if (( METHOD == 0 ))
then
	RES_OUT=IMAGES.out
else
	RES_OUT=TILES_IN_IMAGES.out
fi

echo "image-size	cdf53-sep-lift-gen	cdf53-nsp-lift-gen	cdf53-nsp-conv-at-gen	cdf53-nsp-conv-star-gen	cdf97-sep-lift-gen	cdf97-nsp-lift-gen	cdf97-nsp-conv-at-gen	cdf97-nsp-conv-star-gen	cdf97-nsp-poly-gen	dd137-sep-lift-gen	dd137-nsp-lift-gen	dd137-nsp-conv-at-gen	dd137-nsp-conv-star-gen	cdf53-sep-lift-sse	cdf53-nsp-lift-sse	cdf53-nsp-conv-at-sse	cdf53-nsp-conv-star-sse	cdf97-sep-lift-sse	cdf97-nsp-lift-sse	cdf97-nsp-conv-at-sse	cdf97-nsp-conv-star-sse	cdf97-nsp-poly-sse	dd137-sep-lift-sse	dd137-nsp-lift-sse	dd137-nsp-conv-at-sse	dd137-nsp-conv-star-sse" | tee $RES_OUT

SIZES="1024 2048 3072 4096 5120 8192 11264 16384 23552 32768"

for IMAGE_SIZE in $SIZES
do
	if (( TILE_SIZE/2 < NUM_THREADS ))
	then
		continue
	fi

	echo -n "$IMAGE_SIZE"

	RESULT=$(run_test $NUM_THREADS $TILE_SIZE $IMAGE_SIZE $METHOD)
	echo "	$RESULT"
	
	echo "${IMAGE_SIZE}	${RESULT}" >> $RES_OUT
done

exit 0
