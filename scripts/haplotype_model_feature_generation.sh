#!/bin/bash

ulimit -n 20000

INPUT_VCF="/public/home/hpc164611151/projects/PileupModel/PileupModel/outputs/hg003_100G_hg001_mix_with_refcall.vcf"
HAPLOTAG_DIR="haplotag_out"
THREADS="80"
HAPLOTAG_SPLIT_BAMS="haplotag_split_out"
BIN_DIR1="edge_bins1"
BIN_DIR2="edge_bins2"

mkdir -p $HAPLOTAG_SPLIT_BAMS $BIN_DIR1 $BIN_DIR2

CHR=()
CHR[${#arr[*]}]=`ls $HAPLOTAG_DIR/*.bam | xargs -I file basename file .bam`

time parallel --joblog split_haplotag_bam.log -j$THREADS \
"python split_bam_by_tag.py -in_bam $HAPLOTAG_DIR/{1}.bam \
-h1 $HAPLOTAG_SPLIT_BAMS/{1}_TAG_1.bam \
-h2 $HAPLOTAG_SPLIT_BAMS/{1}_TAG_2.bam \
-tag HP" ::: ${CHR[@]}

time parallel -j$THREADS \
"samtools index $HAPLOTAG_SPLIT_BAMS/{1}_TAG_1.bam && samtools index $HAPLOTAG_SPLIT_BAMS/{1}_TAG_2.bam" ::: ${CHR[@]}

A=(1 2)
time parallel -j2 "samtools merge -@ 10 merged_TAG_{1}.bam $HAPLOTAG_SPLIT_BAMS/*_TAG_{1}.bam" ::: ${A[@]}

time parallel -j2 "samtools index -@ 10 merged_TAG_{1}.bam" ::: ${A[@]}

time parallel --joblog make_edge_matrix.log -j2 \
"python /public/home/hpc164611151/projects/edge_snp_lastv/edge_snp/make_predict_groups2.py \
--bam merged_TAG_{1}.bam \
--pileup_vcf $INPUT_VCF \
--output edge_bins{1} \
--min_quality 19 \
--support_quality 14 \
-t 40 \
--adjacent_size 5" ::: ${A[@]}