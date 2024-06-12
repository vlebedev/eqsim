# Compiler and Tools
NVCC = nvcc
MAVEN = mvn

# Directory settings
SRC_DIR = src/main/java
CUDA_DIR = cuda
BIN_DIR = target

# CUDA settings
CUDA_TARGET = $(BIN_DIR)/computeLosses.ptx

# Find CUDA source files
CUDA_SOURCES = $(wildcard $(CUDA_DIR)/*.cu)

# Build rules
all: java cuda

$(BIN_DIR)/%.ptx: $(CUDA_DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) -O3 -ptx $< -o $@

cuda: $(CUDA_TARGET)

java:
	$(MAVEN) clean package

run:
	$(MAVEN) exec:java

clean:
	$(MAVEN) clean
	rm -f $(BIN_DIR)/*.ptx

.PHONY: all java cuda clean run

