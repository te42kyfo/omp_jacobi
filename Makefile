NVCC := nvc++


# internal flags
NVCCFLAGS   :=  -Minfo=all -mp=gpu
CCFLAGS     := 
LDFLAGS     :=
NAME 		:= omp-jacobi
PREFIX		:= .
INCLUDES 	:= 			

$(PREFIX)/$(NAME): main.cpp Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)
