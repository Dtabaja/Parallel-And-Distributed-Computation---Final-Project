#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

#define CON_SIZE 9
#define CON_LEN 5
#define SEMI_SIZE 11
#define SEMI_LEN 7
#define SQ1_LEN 3000
#define SQ2_LEN 2000
#define WEIGHTS 4

__constant__ char conservativeGroup[CON_SIZE][CON_LEN] = { "NDEQ","MILV","FYW", "NEQK","QHRK", "HY", "STA", "NHQK","MILF"};
__constant__ char semiConservativeGroup[SEMI_SIZE][SEMI_LEN] = {"SAG","SGND","NEQHRK","ATV","STPA","NDEQHK","HFY","CSA","STNK","SNDEQK","FVLIM"};

__device__ int matchingGroups(char ch1, char ch2, char *str) {
	int c1 = 0, c2 = 0;

	while (*str != '\0') {
		if (ch1 == *str)
			c1 = 1;
		if (ch2 == *str)
			c2 = 1;
		if (c1 && c2)
			return 1;
		str++;
	}
	return 0;
}

__device__ void copyConservativeGroup(char group1[][CON_LEN],
		char group2[][CON_LEN], int r) {

	char *group;
	char *str;
	group = group1[r];
	str = group2[r];

	while (*group != '\0') {
		*str = *group;
		str++;
		group++;
	}

}

__device__ void copySemiConservativeGroup(char group1[][SEMI_LEN],
		char group2[][SEMI_LEN], int r)

{
	char *group;
	char *str;
	group = group1[r];
	str = group2[r];

	while (*group != '\0') {
		*str = *group;
		str++;
		group++;
	}

}

__global__ void cudaFormation(char *sq1, char *sq2, int sq1Len, int sq2Len,
		int sq2num, int s, int e, double *weights, double *scores,
		int *mutantArr, int *offsetArr, int exponent2) {


	__shared__
	char conservativeGroupShared[CON_SIZE][CON_LEN];
	__shared__
	char semiConservativeShared[SEMI_SIZE][SEMI_LEN];
	__shared__
	double scoreShared[SQ2_LEN];
	__shared__
	double weightShare[WEIGHTS];
	__shared__
	char seq1[SQ1_LEN];
	__shared__
	char seq2[SQ2_LEN];

	int mutant, offset, chosenMutant, chosenOffset;
	int i, j, k, l, w;
	int id = threadIdx.x;
	double bestScore = -INFINITY;

	if (id >= sq2Len)
		return;
	if (id == 0) {
		for (i = 0; i < sq1Len; i++)
			seq1[i] = sq1[i];
		for (i = 0; i < sq2Len; i++)
			seq2[i] = sq2[i];
	}


	if (id < CON_SIZE)
		copyConservativeGroup(conservativeGroup, conservativeGroupShared, id);
	if (id < SEMI_SIZE)
		copySemiConservativeGroup(semiConservativeGroup, semiConservativeShared,
				id);
	if (id < WEIGHTS)
		weightShare[id] = weights[id];

	for (i = s; i < e; i++) {
		__syncthreads();
		unsigned int temp;

		//location of mark - in each loaction between the chars
		mutant = i / (sq1Len - sq2Len - 1);
		//location of the offset
		offset = i % (sq1Len - sq2Len - 1);
		scoreShared[id] = 0;

		for (k = id; k < sq2Len; k += blockDim.x) {
			w = 0;
			l = k;
			if (k > mutant)
				l++;
			if (seq2[k] == seq1[l + offset]) {
				scoreShared[id] += weightShare[0];
				w = 1;
				continue;
			}

			for (j = 0; j < CON_SIZE; j++) {
				if (matchingGroups(seq2[k], seq1[l + offset],
						&conservativeGroupShared[j][0])) {
					scoreShared[id] -= weightShare[1];
					w = 1;
					break;
				}
			}

			if (w)
				continue;

			for (j = 0; j < SEMI_SIZE; j++) {
				if (matchingGroups(seq2[k], seq1[l + offset],
						&semiConservativeShared[j][0])) {
					scoreShared[id] -= weightShare[2];
					w = 1;
					break;
				}
			}
			if (w)
				continue;

			scoreShared[id] -= weightShare[3];

		}

		__syncthreads();
		if (exponent2 / 2 != blockDim.x) {
			temp = exponent2 / 2;

			if (id < temp) {

				if (id + temp < sq2Len)
					scoreShared[id] += scoreShared[id + temp];

			}
			__syncthreads();
		} else
			temp = blockDim.x;

		for (temp = temp / 2; temp > 0; temp = temp / 2) {
			if (id < temp) {
				scoreShared[id] += scoreShared[id + temp];
			}
			__syncthreads();

		}
		if (id == 0) {
			scoreShared[0] -= weightShare[3];
			if (scoreShared[0] > bestScore) {
				chosenMutant = mutant;
				chosenOffset = offset;
				bestScore = scoreShared[0];

			}
		}

	}

	if (id == 0) {
		mutantArr[sq2num] = chosenMutant;
		offsetArr[sq2num] = chosenOffset;
		scores[sq2num] = bestScore;

	}

}

cudaError_t cudaErrorAssistant(char *sq1, char *sq2, int sq1Len, int sq2Len,
		int sq2size, int start, int end, double *weights, double *scores,
		int *mutants, int *offsets, int pow_of_2) {
	cudaError_t status;
	cudaDeviceProp prop;
	int numOftread, numberOfblock, tempBlock;

	cudaGetDeviceProperties(&prop, 0);


	if (prop.maxThreadsPerBlock < sq2Len) {
		numOftread = prop.maxThreadsPerBlock;
	} else {
		numOftread = sq2Len;
	}

	numberOfblock = sq2Len / numOftread;
	tempBlock = sq2Len % numOftread != 0;

	if(numberOfblock + tempBlock == 1)
	cudaFormation<<<numberOfblock + tempBlock, numOftread>>>(sq1, sq2,sq1Len,sq2Len, sq2size, start,end, weights, scores, mutants, offsets,pow_of_2);
	else
	cudaFormation<<<1, numOftread>>>(sq1, sq2,sq1Len,sq2Len, sq2size, start,end, weights, scores, mutants, offsets,pow_of_2);

	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(status));
		return status;
	}

	return status;
}

cudaError_t sq1Init(char *sq1, char **sq1FromCuda, int size) {
	cudaError_t status;

	status = cudaMalloc((void**) sq1FromCuda, size * sizeof(char));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Condition;
	}


	status = cudaMemcpy(*sq1FromCuda, sq1, size * sizeof(char),
			cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Condition;
	}

	return status;

	Condition: cudaFree(*sq1FromCuda);
	return status;

}

cudaError_t initSeq2(char **sq2Arr, char **sq2Cuda, int SeqNumber) {
	cudaError_t status;
	int i;
	char *temp;

	for (i = 0; i < SeqNumber; i++) {

		status = cudaMalloc((void**) (&temp),
				(strlen(sq2Arr[i]) + 1) * sizeof(char));
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Condtion;
		}


		status = cudaMemcpy(temp, sq2Arr[i],
				(strlen(sq2Arr[i]) + 1) * sizeof(char), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Condtion;
		}

		sq2Cuda[i] = temp;

	}

	return status;

	Condtion: for (i = 0; i < SeqNumber; i++)
		cudaFree(sq2Cuda[i]);
	return status;

}
cudaError_t weightInit(double *weights, double **cudaWieght, int weightNum) {

	cudaError_t status;


	status = cudaMalloc((void**) cudaWieght, weightNum * sizeof(double));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Condition;
	}


	status = cudaMemcpy(*cudaWieght, weights, weightNum * sizeof(double),
			cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Condition;
	}

	return status;

	Condition: cudaFree(*cudaWieght);
	return status;

}
cudaError_t resultFromThread(int seqNumber, double **score, int **mutants,
		int **offsets, double *procScore) {
	cudaError_t status;

	status = cudaMalloc((void**) (mutants), (seqNumber * sizeof(int)));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Condition;
	}
	status = cudaMalloc((void**) (offsets), (seqNumber * sizeof(int)));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Condition;
	}
	status = cudaMalloc((void**) (score), (seqNumber * sizeof(double)));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Condition;
	}
	

	status = cudaMemcpy(*score, procScore, seqNumber * sizeof(double),
			cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Condition;
	}
	return status;

	Condition: cudaFree(*score);
	cudaFree(*mutants);
	cudaFree(*offsets);
	return status;

}
cudaError_t seqResults(char **seqCuda, int size) {
	cudaError_t status;

	status = cudaMalloc((void**) seqCuda, size * sizeof(char));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return status;
	}
	return status;


}

cudaError_t copyFromGpuToCpu(double *threadScore, int *treadMutant,
		int *treadOffset, double *scoreFromCuda, int *cudaMutant,
		int *cudaOffset, int seqNum) {
	cudaError_t status;

	status = cudaMemcpy(treadOffset, cudaOffset, seqNum * sizeof(int),
			cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n%s\n", cudaGetErrorString(status));
	}
	status = cudaMemcpy(treadMutant, cudaMutant, seqNum * sizeof(int),
			cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");

	}

	status = cudaMemcpy(threadScore, scoreFromCuda, seqNum * sizeof(double),
			cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");

	}
	return status;
}

