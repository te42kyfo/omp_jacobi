NVCC := nvc++


# internal flags
NVCCFLAGS   :=  -Minfo=all -mp=gpu
CCFLAGS     := 
LDFLAGS     :=
NAME 		:= omp-jacobi

INCLUDES 	:= 			

main7: main7.cu Makefile
	nvcc main7.cu -o main7

%: %.cpp Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

main%-aomp: main%.cpp
	/opt/rocm/llvm/bin/clang++  -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908  -Wl,-rpath,/opt/rocm/lib-lamdhip64  $< -o $@

clean:
	rm  main1 main2 main3 main4 main5 main6 main7
	rm  *.qdrep *.ncu-rep
