#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_PATH=`dirname "$0"`
Usage="Usage: ./${SCRIPT_NAME} --bam_fn=BAM --ref_fn=REF --output=OUTPUT_DIR --threads=THREADS "

set -e
#./make_train_data.sh -b tmp.bam -f ref.fasta -t 32 -o tmp
print_help_messages()
{
    echo $''
    echo ${Usage}
    echo $''
    echo $'Required parameters:'
    echo $'  -b, --bam_fn=FILE             BAM file input. The input file must be samtools indexed.'
    echo $'  -f, --ref_fn=FILE             FASTA reference file input. The input file must be samtools indexed.'
    echo $'  -t, --threads=INT             Max #threads to be used. The full genome will be divided into small chunks for parallel processing. Each chunk will use 4 threads. The #chunks being processed simultaneously is ceil(#threads/4)*3. 3 is the overloading factor.'
    echo $'  -o, --output=PATH             output directory.'
    echo $''
    echo $''
}

# ERROR="\\033[31m[ERROR]"
# WARNING="\\033[33m[WARNING]"
# NC="\\033[0m"

ARGS=`getopt -o b:f:t:o: -l bam_fn:,ref_fn:,threads:,output:,help -- "$@"`

[ $? -ne 0 ] && ${Usage}
eval set -- "${ARGS}"

while true; do
   case "$1" in
    -b|--bam_fn )
            BAM_FILE_PATH="$2"
            shift
            ;;
    -f|--ref_fn )
            REFERENCE_FILE_PATH="$2"
            shift
            ;;
    -t|--threads )
            THREADS="$2"
            shift
            ;;
    -o|--output )
            OUTPUT_FOLDER="$2"
            shift
            ;;
    -h|--help )
            print_help_messages
            ;;
    -- )
            shift
            break
            ;;
   esac
shift
done

if [ -z ${BAM_FILE_PATH} ] || [ -z ${REFERENCE_FILE_PATH} ] || [ -z ${THREADS} ] || [ -z ${OUTPUT_FOLDER} ]; then
      if [ -z ${BAM_FILE_PATH} ] && [ -z ${REFERENCE_FILE_PATH} ] && [ -z ${THREADS} ] && [ -z ${OUTPUT_FOLDER} ]; then print_help_messages; exit 0; fi
      if [ -z ${BAM_FILE_PATH} ]; then echo -e "${ERROR} Require to define index BAM input by --bam_fn=BAM${NC}"; fi
      if [ -z ${REFERENCE_FILE_PATH} ]; then echo -e "${ERROR} Require to define FASTA reference file input by --ref_fn=REF${NC}"; fi
      if [ -z ${THREADS} ]; then echo -e "${ERROR} Require to define max threads to be used by --threads=THREADS${NC}"; fi
      if [ -z ${OUTPUT_FOLDER} ]; then echo -e "${ERROR} Require to define output folder by --output=OUTPUT_DIR${NC}"; fi

      print_help_messages;
      exit 1;
fi

bam=${BAM_FILE_PATH}
ref=${REFERENCE_FILE_PATH}
output=${OUTPUT_FOLDER}
threads=${THREADS}

mkdir -p ${output}

bash scripts/s1_pileup_model_feature_generation.sh \
${bam} \
${ref} \
${output} \
${threads} > ${output}/s1.log 2>&1

bash scripts/s2_pileup_model_predict.sh \
${output}/bin_predict_data \
${output}/pileup.vcf > ${output}/s2.log 2>&1

bash scripts/s3_phasing_long_reads.sh \
${output}/pileup.vcf \
${ref} \
${bam} \
${threads} \
${output}/phase_out \
${output}/splited_bams \
${output}/splited_vcfs \
${output}/haplotag_out > ${output}/s3.log 2>&1

bash scripts/s4_haplotype_model_feature_generation.sh \
${output}/pileup.vcf \
${output}/haplotag_out \
${threads} \
${output}/haplotag_split_out \
${output}/edge_bins1 \
${output}/edge_bins2 > ${output}/s4.log 2>&1


bash scripts/s5_haplotype_model_predict.sh \
${output}/edge_bins1 \
${output}/edge_bins2 \
${output}/haplotype.csv > ${output}/s5.log 2>&1

bash scripts/s6_merge_pileup_haplotype_calls.sh \
${output}/pileup.vcf \
${output}/haplotype.csv \
${output}/merge.vcf > ${output}/s6.log 2>&1

