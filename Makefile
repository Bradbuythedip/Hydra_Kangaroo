# HYDRA KANGAROO — Optimized Pollard's Kangaroo Solver
# Target: Bitcoin Puzzle #135
#
# Key innovation: Batch inversion + Z=1 specialized EC adds +
# Galbraith-Ruprai equivalence classes = ~32x effective speedup

NVCC = nvcc
NVCC_FLAGS = -O3 -Iinclude --expt-relaxed-constexpr -lineinfo
CUDA_ARCH ?= sm_89  # RTX 5070 Ti / Ada. Change for your GPU:
                     # sm_86 = RTX 3090, sm_89 = RTX 4090/5070Ti, sm_90 = H100

BUILD_DIR = build

.PHONY: all clean test bench

all: $(BUILD_DIR)/hydra

$(BUILD_DIR)/hydra: src/hydra_kangaroo.cu include/field.cuh include/ec.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(CUDA_ARCH) -o $@ src/hydra_kangaroo.cu
	@echo ""
	@echo "Built $@ for $(CUDA_ARCH)"
	@echo "Run: ./$@ [--dp-bits 25] [--blocks 2048]"

# Register usage analysis (critical for occupancy)
reginfo: src/hydra_kangaroo.cu include/field.cuh include/ec.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(CUDA_ARCH) --ptxas-options=-v -o $(BUILD_DIR)/hydra src/hydra_kangaroo.cu 2>&1 | grep -E "(registers|bytes)"

# Build for specific GPUs
rtx5070ti: CUDA_ARCH=sm_89
rtx5070ti: all

rtx4090: CUDA_ARCH=sm_89
rtx4090: all

rtx3090: CUDA_ARCH=sm_86
rtx3090: all

h100: CUDA_ARCH=sm_90
h100: all

# Test field arithmetic + EC operations
test: $(BUILD_DIR)/test_field
	$(BUILD_DIR)/test_field

$(BUILD_DIR)/test_field: tests/test_field.cu include/field.cuh include/ec.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(CUDA_ARCH) -o $@ tests/test_field.cu

clean:
	rm -rf $(BUILD_DIR)
