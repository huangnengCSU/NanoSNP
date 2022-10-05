#!/bin/bash

# haplotype_bin1=$1
# haplotype_bin2=$2
bin_paths=$1
ref=$2
output_csv=$3

script_dir=$(cd $(dirname $0);pwd)
command_path=$(cd ${script_dir}/../HaplotypeModel/;pwd)
echo "[-- step 5 --] Haplotype model prediction:"${command_path}"/predict_dev.py"

python ${command_path}/predict_dev.py \
-config ${command_path}/config/ont_haplotype.yaml \
-model_path ${command_path}/model/ont_haplotype.chkpt \
-bin_paths ${bin_paths} \
-reference_path ${ref} \
-output ${output_csv}