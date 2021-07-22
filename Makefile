NVCC := nvc++


# internal flags
NVCCFLAGS   :=  -Minfo=mp -mp=gpu -O3
CCFLAGS     :=
LDFLAGS     :=
NAME 		:= omp-jacobi
PREFIX		:= .
INCLUDES 	:= 			

$(PREFIX)/$(NAME): main.cpp Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)
