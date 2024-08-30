#!/bin/bash

pushd .

cd ../../../implementation/dwt/build
rm -rf *
CC=gcc cmake $@ ..
make -j `nproc --all`

popd
