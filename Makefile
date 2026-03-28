# HYDRA KANGAROO — Optimized Pollard's Kangaroo Solver
# Target: Bitcoin Puzzle #135

NVCC = nvcc
NVCC_FLAGS = -O3 -I include --expt-relaxed-constexpr
CUDA_ARCH ?= sm_89  # RTX 5070 Ti / Ada. Change for your GPU:
                     # sm_86 = RTX 3090, sm_89 = RTX 4090/5070Ti, sm_90 = H100

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

TARGET = hydra

SOURCES = $(SRC_DIR)/hydra_kangaroo.cu
HEADERS = $(INC_DIR)/field.cuh $(INC_DIR)/ec.cuh

.PHONY: all clean test

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(CUDA_ARCH) -o $@ $(SOURCES)
	@echo "Built $(BUILD_DIR)/$(TARGET) for $(CUDA_ARCH)"

# Build for specific GPUs
rtx5070ti: CUDA_ARCH=sm_89
rtx5070ti: all

rtx4090: CUDA_ARCH=sm_89
rtx4090: all

rtx3090: CUDA_ARCH=sm_86
rtx3090: all

h100: CUDA_ARCH=sm_90
h100: all

# Test field arithmetic correctness
test: $(BUILD_DIR)/test_field
	$(BUILD_DIR)/test_field

$(BUILD_DIR)/test_field: tests/test_field.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(CUDA_ARCH) -o $@ tests/test_field.cu

# Python prototype (no build needed)
test-python:
	python3 scripts/kangaroo.py --test

clean:
	rm -rf $(BUILD_DIR)
