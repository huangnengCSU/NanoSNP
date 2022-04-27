# NanoSNP: A progressive and haplotype-aware SNP caller on low coverage Nanopore sequencing data

## Installation

NanoSNP can be installed using Docker or Singularity:  
* docker installation with gpu device:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP/
docker pull huangnengcsu/nanosnp:v1-gpu
```
* singularity installation with gpu devices:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP/
singularity pull docker://huangnengcsu/nanosnp:v1-gpu
```


## Usage

For whole genome SNP calling on each chromosome including chr1-chr22,chrX,chrY,chrM, use `run_caller.sh` to call SNPs.

Singularity:
```
singularity exec --nv --containall \
-B "[INPUT_DIR]":"[INPUT_DIR]","[OUTPUT_DIR]":"[OUTPUT_DIR]" \  ## Absolute path
nanosnp_v1-gpu.sif run_caller.sh \
-b "[INPUT_DIR]/[BAM_FILE]" \   ## Input bam file are stored in the directory of [INPUT_DIR].
-f "[INPUT_DIR]/[REFERENCE_FILE]" \    ## Input reference file are stored in the directory of [INPUT_DIR].
-t "[THREADS]" \
-o "[OUTPUT_DIR]"
```

Docker:
```
docker run \
-v "[INPUT_DIR]":"[INPUT_DIR]" \    ## Absolute path
-v "[OUTPUT_DIR]":"[OUTPUT_DIR]" \  ## Absolute path
--gpus all \
huangnengcsu/nanosnp:v1-gpu \
run_caller.sh \
-b "[INPUT_DIR]/[BAM_FILE]" \   ## Input bam file are stored in the directory of [INPUT_DIR].
-f "[INPUT_DIR]/[REFERENCE_FILE]" \    ## Input reference file are stored in the directory of [INPUT_DIR].
-t "[THREADS]" \
-o "[OUTPUT_DIR]"
```