#!/bin/sh

model=$1

if [ $# -eq 0 ];then
echo "mobilenet or mobilenet_v2 ?"
exit -1
fi

caffe=/export/gpudata/surui/caffe_lib/MobileNet-YOLO/build/tools/caffe

if [ $model = "mobilenet" ];then
  $caffe train -solver ./prototxt/m1_solver.prototxt -weights ckpt/pretrain/mobilenet.caffemodel -gpu 2 
else
  $caffe train -solver ./prototxt/m2_solver.prototxt -weights ckpt/pretrain/mobilenet_v2.caffemodel
fi
