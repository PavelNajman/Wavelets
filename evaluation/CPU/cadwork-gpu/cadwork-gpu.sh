#!/bin/bash

OMP_PLACES="{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {24}, {25}, {26}, {27}, {28}, {29}, {30}, {31}"
OMP_PROC_BIND="close"
#OMP_PLACES="cores"

export OMP_PLACES
export GOMP_CPU_AFFINITY

cd ../scripts

if [ "$2" == "IMAGE" ]
then
	./image.sh $1 1024
fi

if [ "$2" == "TILE" ]
then
	./tile.sh $1
fi

if [ "$2" == "TILES_IN_IMAGE" ]
then
	./image.sh $1 1024 2
fi

if [ "$2" == "THREADS" ]
then
	./threads.sh 1024
fi

exit 0

