#!/bin/bash

rm -f conv_para
rm -f log
for ((i=1; i<190; i=i+1))
do
	sed -n ''$i'p' conv_para.bak > conv_para
	nvprof ./convolution `cat conv_para` > log 2>&1
	sed -n '/1101/p' log | awk '{printf("%s ", $4);}'
	sed -n '/cudnn/p' log | awk '{printf("%s", $4);}'
	echo
done
