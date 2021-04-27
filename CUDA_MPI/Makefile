all:
	mpixlc -g highlifeMpi.c -c -o highlifeMpi.o
	nvcc -g -G -arch=sm_70 highlifeCuda.cu -c -o highlifeCuda.o 
	mpicc -g highlifeMpi.o highlifeCuda.o -o highlife-exe -L/usr/local/cuda-10.2/lib64/ -lcudadevrt -lcudart -lstdc++

