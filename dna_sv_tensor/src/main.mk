ifeq "$(strip ${BUILD_DIR})" ""
  BUILD_DIR    := ../$(OSTYPE)-$(MACHINETYPE)/obj
endif
ifeq "$(strip ${TARGET_DIR})" ""
  TARGET_DIR   := ../$(OSTYPE)-$(MACHINETYPE)/bin
endif

TARGET       := libdnasv.a

SOURCES      := ./common/bed_intv_list.cpp \
                ./common/cpp_aux.cpp \
                ./common/genotype.cpp \
                ./common/kstring.c \
                ./common/line_reader.cpp \
                ./common/ref_reader.cpp \
                ./common/tensor.cpp

SRC_INCDIRS  :=

SUBMAKEFILES := ./extract_chr_pileup_data/main.mk \
                ./extend_bed/main.mk \
                ./split_vcf/main.mk \
                ./make_candidate_snp_tensor/main.mk \
                ./make_predict_data/main.mk \
                ./make_train_data/main.mk \
                ./split_sam/main.mk