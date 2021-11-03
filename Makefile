.PHONY: all
all: cuda

.PHONY: cuda
cuda:
	nvcc -O3 -o main_cu nn.cu
	./main_cu
.PHONY: run
	
run:
	g++ -O3 -o main nn.cpp
	./main
.PHONY: run2

.PHONY: clean
clean:
	rm main.*

.PHONY: winclean
winclean:

	del main*