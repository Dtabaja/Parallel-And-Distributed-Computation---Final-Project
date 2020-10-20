build:
	mpicxx -fopenmp -c Project.c -o Project.o
	#mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c CudaFunction.cu -o CudaFunction.o
	mpicxx -fopenmp -o mpiCudaOpemMP  Project.o  CudaFunction.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP

run2:
	mpiexec -np 2 -machinefile  mfile  -map-by  node  ./mpiCudaOpemMP


