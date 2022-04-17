#!/bin/bash

pileup_calls=$1
haplotype_calls=$2
final_vcf=$3

python merge.py \
--pileup_vcf ${pileup_calls} \
--cat_predict ${haplotype_calls} \
--quality 19 \
--output ${final_vcf}