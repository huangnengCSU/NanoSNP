#!/bin/bash

script_dir=$(cd $(dirname $0);pwd)

bam=$1
ref=$2
outdir=$3
threads=$4

command_path=$(cd ${script_dir}/../dna_sv_tensor/src/scripts/;pwd)
echo "[-- step 1 --] Pileup feature generation:"${command_path}"/make_predict_data.sh"

# bash ${command_path}/make_predict_data.sh -b ${bam} -f ${ref} -t ${threads} -o ${outdir}
bash ${command_path}/make_predict_data.sh -b ${bam} -f ${ref} -t ${threads} -o ${outdir} --usecontig
# rm ${outdir}/bin_predict_data/job_make_bin_predict_data.done