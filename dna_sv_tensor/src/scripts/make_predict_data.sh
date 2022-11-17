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
    echo $'  -g, --usecontig               Call SNPs using contigs as the reference genome.'
    echo $''
    echo $''
}

USE_CONTIG="false"

# ERROR="\\033[31m[ERROR]"
# WARNING="\\033[33m[WARNING]"
# NC="\\033[0m"

ARGS=`getopt -o b:f:t:o:g -l bam_fn:,ref_fn:,threads:,output:,usecontig,help -- "$@"`

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
    -g|--usecontig )
            USE_CONTIG="true"
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

############## The following options must be provided

#ALL_CHR_LIST=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)

BIN_EXEC_PATH=${SCRIPT_PATH}/../../Linux-amd64/bin/

MAKE_BIN_PREDICT_DATA_PY=${SCRIPT_PATH}/../make_bin_data/make_bin_predict_data.py

REFERENCE=${REFERENCE_FILE_PATH}

SORTED_BAM_PATH=${BAM_FILE_PATH}

WORK_DIR=${OUTPUT_FOLDER}

CPU_THREADS=${THREADS}

if $USE_CONTIG
then
    ALL_CHR_LIST[${#arr[*]}]=`cat ${REFERENCE}.fai | awk -F'\t' '{print $1}'`
    echo "CHR_LIST: ${ALL_CHR_LIST[@]}"
else
    ALL_CHR_LIST=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)
    echo "CHR_LIST: ${ALL_CHR_LIST[@]}"
fi

############# workding directory

PILEUP_DATA_DIR=${WORK_DIR}/pileup_data

CHR_PILEUP_DATA_DIR=${WORK_DIR}/chr_pileup_data

CANDIDATE_SNP_TENSOR_DIR=${WORK_DIR}/candidate_snp_tensor

TXT_PREDICT_DATA_DIR=${WORK_DIR}/predict_data

BIN_PREDICT_DATA_DIR=${WORK_DIR}/bin_predict_data

### options

SAMTOOS_MPILEUP_OPTIONS="--min-MQ 20 --min-BQ 0 --reverse-del --excl-flags 2316 --max-depth 144"

FLANKING_BASES=16
BED_EXTENDED_BASES=31

MIN_AF=0.12
SNP_MIN_AF=0.12
INDEL_MIN_AF=0.12
MIN_DEPTH=6

### dump config info

echo "========== config info"

echo "reference chr list"
echo "  ${ALL_CHR_LIST[@]}"

echo "binary executable path"
echo "  ${BIN_EXEC_PATH}"

echo "bam path"
echo "  ${SORTED_BAM_PATH}"

echo "working directory"
echo "  ${WORK_DIR}"

echo ""

###

mkdir -p ${PILEUP_DATA_DIR}
pileup_data_done=${PILEUP_DATA_DIR}/job_make_pileup_data.done 
if [ ! -f ${pileup_data_done} ]; then
    echo "[$(date)] ============> Step 1/5: making pileup data from ${SORTED_BAM_PATH}"
    CMD="samtools mpileup ${SAMTOOS_MPILEUP_OPTIONS} -o ${PILEUP_DATA_DIR}/pileup_data ${SORTED_BAM_PATH}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${pileup_data_done}
fi

mkdir -p ${CHR_PILEUP_DATA_DIR}
chr_pileup_data_done=${CHR_PILEUP_DATA_DIR}/job_chr_pileup_data.done 
if [ ! -f ${chr_pileup_data_done} ]; then
    echo "[$(date)] ============> Step 2/5: split chr pileup data"
    CMD="${BIN_EXEC_PATH}/DNA_ExtractChrPileupData ${PILEUP_DATA_DIR}/pileup_data ${CHR_PILEUP_DATA_DIR} ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${chr_pileup_data_done}
fi

ALL_CHR_LIST[${#arr[*]}]=`ls ${CHR_PILEUP_DATA_DIR}/*.mpileup | xargs -I file basename file .mpileup`
mkdir -p ${CANDIDATE_SNP_TENSOR_DIR}
make_can_snp_tensor_done=${CANDIDATE_SNP_TENSOR_DIR}/job_make_snp_tensor.done 
if [ ! -f ${make_can_snp_tensor_done} ]; then
    echo "[$(date)] ============> Step 3/5: Create tensor from pileup data"
    CMD="${BIN_EXEC_PATH}/DNA_CreateCanSnpTensor \
        -reference ${REFERENCE} \
        -chr_pileup_dir ${CHR_PILEUP_DATA_DIR} \
        -output_dir ${CANDIDATE_SNP_TENSOR_DIR} \
        -min_af ${MIN_AF} \
        -snp_min_af ${SNP_MIN_AF} \
        -indel_min_af ${INDEL_MIN_AF} \
        -min_coverage ${MIN_DEPTH} \
        -flanking_base ${FLANKING_BASES} \
        -num_threads ${CPU_THREADS} \
        ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${make_can_snp_tensor_done}
fi  

mkdir -p ${TXT_PREDICT_DATA_DIR}
make_predict_data_done=${TXT_PREDICT_DATA_DIR}/job_make_predict_data.done 
if [ ! -f ${make_predict_data_done} ]; then
    echo "[$(date)] ============> Step 4/5: making predict data"
    CMD="${BIN_EXEC_PATH}/DNA_CreatePredictData \
        -chr_tensor_dir ${CANDIDATE_SNP_TENSOR_DIR} \
        -reference ${REFERENCE} \
        -output_dir ${TXT_PREDICT_DATA_DIR} \
        -num_threads ${CPU_THREADS} \
        ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${make_predict_data_done}    
fi 

mkdir -p ${BIN_PREDICT_DATA_DIR}
make_bin_predict_data_done=${BIN_PREDICT_DATA_DIR}/job_make_bin_predict_data.done
if [ ! -f ${make_bin_predict_data_done} ]; then
    echo "[$(date)] ============> Step 5/5: making binary predict data"
    for chr in ${ALL_CHR_LIST[@]}
    do
        echo "make binary predict data for ${chr}"
        python ${MAKE_BIN_PREDICT_DATA_PY} ${BIN_PREDICT_DATA_DIR}/${chr}.pd.bin ${TXT_PREDICT_DATA_DIR}/${chr}.pd
        if [ $? -ne 0 ]; then
            echo "[$(date)] Fail at making binary predict data for ${chr}"
            exit 1
        fi
    done 
    touch ${make_bin_predict_data_done}
fi