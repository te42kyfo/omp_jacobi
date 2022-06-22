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


clean:
	rm  main1 main2 main3 main4 main5 main6 main7
	rm  *.qdrep *.ncu-rep
