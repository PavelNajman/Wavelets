function run_test {
	NUM_THREADS=$1
	TILE_SIZE=$2
	IMAGE_SIZE=$3
	METHOD=$4
	
	../../../implementation/dwt/build/dwt-benchmark --threads=${NUM_THREADS} --image-width=${IMAGE_SIZE} --image-height=${IMAGE_SIZE} --tile-width=${TILE_SIZE} --tile-height=${TILE_SIZE} --benchmark-type=${METHOD}
}

