#!/bin/bash

METHOD=$1
NUM_THREADS_START=$2
NUM_THREADS_END=$3

TMP="tmp.out"
RES_OUT=$METHOD.out

if [ "$METHOD" == "TILE" ]
then
	echo "Threads	Tile Size	nsp	sep	nsp	sep	nsp	sep	nsp	sep" > $RES_OUT
else
	echo "Threads	Image Size	Tile Size	nsp	sep	nsp	sep	nsp	sep	nsp	sep" > $RES_OUT
fi

for i in $(seq $NUM_THREADS_START $NUM_THREADS_END)
do
	FILE=1_${i}_${METHOD}.out
	NUM_LINES=`cat $FILE | wc -l`
	NUM_LINES=`echo $NUM_LINES - 1 | bc` 
	tail -$NUM_LINES $FILE > $TMP

	while read -r line
	do
		if [ -z "$line" ]
		then
			continue
		fi
		echo "$i	$line" >> $RES_OUT
	done < "$TMP"
done

rm -f $TMP

exit 0
