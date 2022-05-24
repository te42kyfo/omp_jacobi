NVCC := nvc++


# internal flags
NVCCFLAGS   :=  -Minfo=all -mp=gpu
CCFLAGS     := 
LDFLAGS     :=
NAME 		:= omp-jacobi

INCLUDES 	:= 			

%: %.cpp Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm  main1 main2 main3 main4 main5 main6
	rm  *.qdrep *.ncu-rep
