#!/bin/sh
data_dir=$1

if [ $# -eq 0 ];then
   echo "Please input data directory path."
   exit 1;
fi

ls $data_dir | sed "s:^:`pwd`/$data_dir/:" > img_path.txt
ls $data_dir | awk '{split($0, x, "_");print(x[1])}' > label.txt

paste -d" " img_path.txt label.txt > trainval.txt
