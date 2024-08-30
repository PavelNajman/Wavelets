#!/bin/bash

pushd .

./make_all.sh

cd ../../../implementation/dwt

cp build/dwt-test .

for i in 1 2 4 16 256 512 4096
do
	./dwt-test --threads=$i --image-width=512 --image-height=512 --tile-width=512 --tile-height=512 --blocks
	if [ $? -ne 0 ]
	then
		exit 1
	fi
done

popd

exit 0
