#!/bin/bash

haplotype_bin1=$1
haplotype_bin2=$2
output_csv=$3

script_dir=$(cd $(dirname $0);pwd)
command_path=$(cd ${script_dir}/../HaplotypeModel/;pwd)
echo "[-- step 5 --] Haplotype model prediction:"${command_path}"/predict.py"

python ${command_path}/predict.py \
-config ${command_path}/config/haplotype.yaml \
-model_path ${command_path}/model/cat45.epoch179.chkpt \
-data_tag1 ${haplotype_bin1} \
-data_tag2 ${haplotype_bin2} \
-output ${output_csv}