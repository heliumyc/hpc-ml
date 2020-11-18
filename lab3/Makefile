all: ./lab3

CUDADIR :=  /usr/local/cuda/
CUDNNDIR := /usr/local/cuda

CPPFLAGS := -g --std=c++11 -I$(CUDADIR)/include -I$(CUDNNDIR)/include
LDFLAGS := -g -L$(CUDADIR)/lib -L$(CUDNNDIR)/lib64
LDLIBS := -lcublas -lcudnn

NVCC := nvcc
CC := $(NVCC)

%.o: %.cu
	$(NVCC) $(CPPFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f ./lab3
