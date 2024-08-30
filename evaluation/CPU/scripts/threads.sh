#!/bin/bash

# Finds optimal number of threads for tile of a given size.

source utils.sh

TILE_SIZE=$1

if [ -z "$1" ]
then
	echo "Usage: ./threads.sh TILE_SIZE"
	exit 1
fi

METHOD=0
MAX_THREADS=`grep --count ^processor /proc/cpuinfo`
	
THREADS=()

RESULTS=()

NUM_SOCKETS=1

RES_OUT=THREADS.out

echo "number-of-threads	cdf53-sep-lift-gen	cdf53-nsp-lift-gen	cdf53-nsp-conv-at-gen	cdf53-nsp-conv-star-gen	cdf97-sep-lift-gen	cdf97-nsp-lift-gen	cdf97-nsp-conv-at-gen	cdf97-nsp-conv-star-gen	cdf97-nsp-poly-gen	dd137-sep-lift-gen	dd137-nsp-lift-gen	dd137-nsp-conv-at-gen	dd137-nsp-conv-star-gen	cdf53-sep-lift-sse	cdf53-nsp-lift-sse	cdf53-nsp-conv-at-sse	cdf53-nsp-conv-star-sse	cdf97-sep-lift-sse	cdf97-nsp-lift-sse	cdf97-nsp-conv-at-sse	cdf97-nsp-conv-star-sse	cdf97-nsp-poly-sse	dd137-sep-lift-sse	dd137-nsp-lift-sse	dd137-nsp-conv-at-sse	dd137-nsp-conv-star-sse" | tee $RES_OUT

for (( I=0, NUM_THREADS=1; NUM_THREADS <= MAX_THREADS; NUM_THREADS+=1, I+=1 ))
do
	if (( TILE_SIZE/2 < NUM_THREADS ))
	then
		continue
	fi

	THREADS[$I]=$NUM_THREADS

	echo -n "$NUM_THREADS"

	TMP=$(run_test $NUM_THREADS $TILE_SIZE $TILE_SIZE $METHOD)
	echo "	$TMP"
	RESULTS[$I]=$TMP
done

for (( I=0, NUM_THREADS=1; NUM_THREADS <= MAX_THREADS; NUM_THREADS+=1, I+=1 ))
do
	echo "${THREADS[$I]}	${RESULTS[$I]}" >> $RES_OUT
done

exit 0
