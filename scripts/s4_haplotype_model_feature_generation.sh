#!/bin/bash

ulimit -n 20000

script_dir=$(cd $(dirname $0);pwd)

# INPUT_VCF="/public/home/hpc164611151/projects/PileupModel/PileupModel/outputs/hg003_100G_hg001_mix_with_refcall.vcf"
# HAPLOTAG_DIR="haplotag_out"
# THREADS="80"
# HAPLOTAG_SPLIT_BAMS="haplotag_split_out"
# BIN_DIR1="edge_bins1"
# BIN_DIR2="edge_bins2"

INPUT_VCF=$1
HAPLOTAG_DIR=$2
THREADS=$3
HAPLOTAG_SPLIT_BAMS=$4
BIN_DIR1=$5
BIN_DIR2=$6
OUTPUT_DIR=$7

mkdir -p $HAPLOTAG_SPLIT_BAMS $BIN_DIR1 $BIN_DIR2

CHR=()
CHR[${#arr[*]}]=`ls $HAPLOTAG_DIR/*.bam | xargs -I file basename file .bam`

time parallel --joblog ${OUTPUT_DIR}/split_haplotag_bam.log -j$THREADS \
"python ${script_dir}/split_bam_by_tag.py -in_bam $HAPLOTAG_DIR/{1}.bam \
-h1 $HAPLOTAG_SPLIT_BAMS/{1}_TAG_1.bam \
-h2 $HAPLOTAG_SPLIT_BAMS/{1}_TAG_2.bam \
-tag HP" ::: ${CHR[@]}

time parallel -j$THREADS \
"samtools index $HAPLOTAG_SPLIT_BAMS/{1}_TAG_1.bam && samtools index $HAPLOTAG_SPLIT_BAMS/{1}_TAG_2.bam" ::: ${CHR[@]}

A=(1 2)
time parallel -j2 "samtools merge -@ 10 ${OUTPUT_DIR}/merged_TAG_{1}.bam $HAPLOTAG_SPLIT_BAMS/*_TAG_{1}.bam" ::: ${A[@]}

time parallel -j2 "samtools index -@ 10 ${OUTPUT_DIR}/merged_TAG_{1}.bam" ::: ${A[@]}

command_path=$(cd ${script_dir}/../HaplotypeModel/;pwd)

time parallel --joblog ${OUTPUT_DIR}/make_edge_matrix.log -j2 \
"python ${command_path}/make_predict_groups.py \
--bam ${OUTPUT_DIR}/merged_TAG_{1}.bam \
--pileup_vcf $INPUT_VCF \
--output ${OUTPUT_DIR}/edge_bins{1} \
--min_quality 19 \
--support_quality 14 \
-t ${THREADS} \
--adjacent_size 5" ::: ${A[@]}