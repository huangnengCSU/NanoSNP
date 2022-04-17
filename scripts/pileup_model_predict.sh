#!/bin/bash

data_dir=$1
output_vcf=$2

script_dir=$(cd $(dirname $0);pwd)
command_path=$(cd ${script_dir}/../PileupModel/;pwd)
echo "[-- step 2 --] Pileup model prediction:"${command_path}"/predict.py"

python ${command_path}/predict.py \
-config ${command_path}/config/hg001_mix_without_balance.yaml \
-model ${command_path}/models/hg001_mix_without_balance.epoch13.chkpt \
-data ${data_dir} \
-output ${output_vcf}