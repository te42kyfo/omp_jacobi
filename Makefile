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
	rm -f ./$(NAME)
