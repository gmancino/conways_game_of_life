all: highlife.c highlife-cuda.cu
mpixlc g highlife-mpi.c -c -o highlife-mpi.o
nvcc -g -G -arch=sm_70 highlife-cuda.cu -c -o highlife-cuda.o mpicc -g highlife-mpi.o highlife-cuda.o -o highlife-exe \
-L/usr/local/cuda-10.2/lib64/ -lcudadevrt -lcudart -lstdc++