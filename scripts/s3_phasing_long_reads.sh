#!/bin/bash

ulimit -n 10000

# INPUT_VCF="/public/home/hpc164611151/projects/PileupModel/PileupModel/outputs/hg003_100G_hg001_mix_with_refcall.vcf"
# REF="/public/data/biodata/compu_bio/member/huangneng/SNP_human_datasets/human_samples/GRCh38/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
# BAM="/public/data/biodata/compu_bio/member/huangneng/SNP_human_datasets/human_samples/hg003/low_coverage_test_dataset/hg003_100G.bam"
# THREADS="40"
# PHASED_DIR="phase_out"
# SPLITED_BAMS="splited_bams"
# SPLITED_VCFS="splited_vcfs"
# HAPLOTAG_DIR="haplotag_out"

INPUT_VCF=$1
REF=$2
BAM=$3
THREADS=$4
PHASED_DIR=$5
SPLITED_BAMS=$6
SPLITED_VCFS=$7
HAPLOTAG_DIR=$8
OUTPUT_DIR=$9

script_dir=$(cd $(dirname $0);pwd)


PHASED_PREFIX=$PHASED_DIR/`basename $PHASED_DIR`

CHR=()
CHR[${#arr[*]}]=`cat $REF.fai | xargs -I RD echo RD | cut -f1`

mkdir -p $PHASED_DIR $SPLITED_BAMS $SPLITED_VCFS $HAPLOTAG_DIR

## split alignment bam file by chromosomes.
time parallel --joblog ${OUTPUT_DIR}/splited_bams.log -j$THREADS \
"samtools view -b -h $BAM {1} > $SPLITED_BAMS/splited_{1}.bam && \
samtools index $SPLITED_BAMS/splited_{1}.bam" ::: ${CHR[@]}


## split vcf file by chromosomes.
bgzip -c $INPUT_VCF > $INPUT_VCF.gz && tabix -p vcf $INPUT_VCF.gz
# time parallel --joblog ${OUTPUT_DIR}/splited_vcf.log -j$THREADS \
# "bcftools view -r {1} $INPUT_VCF.gz > $SPLITED_VCFS/{1}.splited.vcf" ::: ${CHR[@]}
python ${script_dir}/select_high_quality_hetesnps.py --pileup_vcf $INPUT_VCF --support_quality 16 --output_dir $SPLITED_VCFS


## whatshap phase
time parallel --joblog ${OUTPUT_DIR}/whatshap_phase.log -j$THREADS \
"whatshap phase \
--output $PHASED_PREFIX.{1}.phased.vcf \
--reference $REF \
--chromosome {1} \
--distrust-genotypes \
--ignore-read-groups \
$SPLITED_VCFS/{1}.splited.vcf \
$SPLITED_BAMS/splited_{1}.bam" ::: ${CHR[@]}

parallel -j${THREADS} "bgzip -c $PHASED_PREFIX.{1}.phased.vcf > $PHASED_PREFIX.{1}.phased.vcf.gz && tabix -p vcf $PHASED_PREFIX.{1}.phased.vcf.gz" ::: ${CHR[@]}


## whatshap haplotag
time parallel --joblog ${OUTPUT_DIR}/whatshap_haplotag.log -j$THREADS \
"whatshap haplotag \
--output ${HAPLOTAG_DIR}/{1}.bam \
--reference $REF \
--ignore-read-groups \
--regions {1} \
$PHASED_PREFIX.{1}.phased.vcf.gz \
$SPLITED_BAMS/splited_{1}.bam" ::: ${CHR[@]}


CTGS=()
for ctg in ${CHR[@]}; do
    if [ -f ${HAPLOTAG_DIR}/${ctg}.bam ]
    then
        CTGS[${#CTGS[*]}]=${ctg}
    fi
done

parallel -j${THREADS} samtools index -@ 10 ${HAPLOTAG_DIR}/{1}.bam ::: ${CTGS[@]}