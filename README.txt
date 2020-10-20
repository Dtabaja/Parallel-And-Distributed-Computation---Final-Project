Daniel Tabaja 204217533

How to use:

For 1 machine only:

Open the ZIP file in a directory
Then, start a terminal in the directory that been extracted from the ZIP file

First, change the number of threads used by omp:
export OMP_NUM_THREADS=4

Then, clean old builds:
make clean

Then, compile:
make

Then run:
make run



For 2 machines:

Open the ZIP file in a directory in both machines
Then, start a terminal in the directory that been extracted from the ZIP file in both machines

First, find the ip of both machines (two options):
hostname -I
or
ifconfig

Then, put it in the machine file "mf" in both machines (so you can start a run from both)
Then, clean old builds in both machines:
make clean

Then, compile in both machines:
make

Then run from one machine and it will run both:
make run2


