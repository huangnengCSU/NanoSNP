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
singularity build nanosnp_v1-gpu.sif nanosnp.def
```


## Usage

For whole genome SNP calling on each chromosome including chr1-chr22,chrX,chrY,chrM, use `run_caller.sh` to call SNPs.

Singularity:
```
cd NanoSNP/
singularity exec -B "Directory on the host system":"Directory inside the container" \
nanosnp_v1-gpu.sif bash run_caller.sh \
-b "[BAM_FILE]" \
-f "[REFERENCE_FILE]" \
-t "[THREADS]" \
-o "[OUTPUT_DIR]"
```

Docker:
```
cd NanoSNP/
docker run \
-v `pwd`:`pwd` \
-v  "Directory on the host system":"Directory inside the container" \
--gpus all \
huangnengcsu/nanosnp:v1-gpu \
bash `pwd`/run_caller.sh \
-b "[BAM_FILE]" \
-f "[REFERENCE_FILE]" \
-t "[THREADS]" \
-o "[OUTPUT_DIR]"
```