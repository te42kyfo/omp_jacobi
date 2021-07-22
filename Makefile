NVCC := nvc++


# internal flags
NVCCFLAGS   :=  -Minfo=all -mp=gpu -gpu=managed
CCFLAGS     := 
LDFLAGS     :=
NAME 		:= cuda-jacobi
PREFIX		:= .
INCLUDES 	:= 			

$(PREFIX)/$(NAME): main.cpp Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)
