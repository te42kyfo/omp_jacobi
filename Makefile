NVCC := nvcc

TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)

# internal flags
NVCCFLAGS   :=  --compiler-options="-march=native -O3 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" -Xcompiler "-Wall"
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -lcuda
NAME 		:= cuda-jacobi
PREFIX		:= .
INCLUDES 	:= 			

$(PREFIX)/$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)
