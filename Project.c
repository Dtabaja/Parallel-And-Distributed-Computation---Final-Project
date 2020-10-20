#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <math.h>


#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"

#define ROOT 0
#define MAX_SQ1 3000
#define MAX_SQ2 2000
#define NUMBER_WEIGHT 4

#define NUMBER_OF_CON_GRUOP 9
#define CON_LEN 5
#define NUMBER_OF_SEMI_GROUP 11
#define SEMI_LEN 7
#define CUDA_PERCENTAGE 70

char conservativeGroup[NUMBER_OF_CON_GRUOP][CON_LEN] = { "NDEQ","MILV","FYW", "NEQK","QHRK", "HY", "STA", "NHQK","MILF"};
char semiConservativeGroup[NUMBER_OF_SEMI_GROUP][SEMI_LEN] = {"SAG","SGND","NEQHRK","ATV","STPA","NDEQHK","HFY","CSA","STNK","SNDEQK","FVLIM"};


void mpiFunc(int my_id, int num_procs);
void freeSeq2(char **seq2_arr, int seqNumber);
void sequenceFormation(int pid, int numOfprocess, char seq1[MAX_SQ1],char **seq2_arr, int numberofsequence, double weights[NUMBER_WEIGHT]);
void resultsToMainProcessRoot(int my_id, int processNumber, int seqNumber, int *bestOffset, int *bestMutant, double *bestScore);
double cpuFormation(char *seq1, char *seq2, int seq1len, int seq2len, int mutant, int offset, double weights[NUMBER_WEIGHT]);
int bothGroups(char ch1, char ch2, char *group);
int exponent2next(int n);
void freeAllocation();

//cuda functions
int matchingGroups(char ch1, char ch2, char* str);
void copyConservativeGroup(char group1[][CON_LEN], char group2[][CON_LEN],int r);
void copySemiConservativeGroup(char group1[][SEMI_LEN], char group2[][SEMI_LEN],int r);
cudaError_t cudaErrorAssistant(char* sq1,char* sq2,int sq1Len,int sq2Len,int sq2size,int start,int end,double* weights,double* scores,int* mutants,int* offsets,int pow_of_2);
cudaError_t sq1Init(char* sq1, char** sq1FromCuda, int size);
cudaError_t resultFromThread(int seqNumber,double** score,int ** mutants,int** offsets,double* procScore);
cudaError_t initSeq2(char** sq2Arr, char** sq2Cuda, int SeqNumber);
cudaError_t weightInit(double* weights, double** cudaWieght, int weightNum);
cudaError_t copyFromGpuToCpu(double* threadScore,int* treadMutant, int* treadOffset,double* scoreFromCuda,int* cudaMutant,int* cudaOffset,int seqNum);
cudaError_t seqResults(char** seqCuda, int size);




double cpuFormation(char *seq1, char *seq2, int seq1len, int seq2len, int mutant, int offset, double weights[NUMBER_WEIGHT])
{
	int i, j, k, l;
	double score = 0.0;

	for (i = 0; i < seq2len ; i++)
	{

		if (i <= mutant)
			k = i;
		else
			k = i + 1;

		l = 0;
		if (seq2[i] == seq1[k + offset])
		{
			score += weights[0];

			l = 1;
		}

		if (l)
			continue;

		for (j = 0; j < NUMBER_OF_CON_GRUOP; j++)
		{
			if (bothGroups(seq2[i], seq1[k + offset], &conservativeGroup[j][0]))
			{
				score -= weights[1];

				l = 1;
				break;
			}
		}

		if (l)
			continue;


		for (j = 0; j < NUMBER_OF_SEMI_GROUP; j++)
		{
			if (bothGroups(seq2[i], seq1[k + offset], &semiConservativeGroup[j][0]))
			{
				score -= weights[2];

				l = 1;
				break;
			}
		}

		if (l)
			continue;
		else
			score -= weights[3];

	}

	return score-weights[3];


}





void mpiFunc(int my_id, int num_procs)
{
	double weights[NUMBER_WEIGHT];
	char seq1[MAX_SQ1];
	char **seq2_arr;
	int numOfSeq;
	int i;
	int temp_len;


	if (my_id == ROOT)
	{

		FILE *f = fopen(INPUT_FILE, "r");
		if (f == NULL)
		{
			perror("Error");
			printf("Could not open file %s", INPUT_FILE);
			return;
		}
		fscanf(f, "%lf %lf %lf %lf", weights, weights+1, weights+2, weights+3);
		fscanf(f, "%s", seq1);
		fscanf(f, "%d", &numOfSeq);
		seq2_arr = (char **)malloc(numOfSeq * sizeof(char *));
		for (i = 0; i < numOfSeq; i++)
		{
			seq2_arr[i] = (char *)malloc(MAX_SQ2 * sizeof(char));
			fscanf(f, "%s", seq2_arr[i]);
		}
		fclose(f);
		temp_len = strlen(seq1) + 1;
	}

	MPI_Bcast(weights, NUMBER_WEIGHT, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&temp_len, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(seq1, temp_len, MPI_CHAR, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&numOfSeq, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	if (my_id != ROOT)
	{
		seq2_arr = (char **)malloc(numOfSeq * sizeof(char *));
		for (i = 0; i < numOfSeq; i++)
			seq2_arr[i] = (char *)malloc(MAX_SQ2 * sizeof(char));
	}

	for (i = 0; i < numOfSeq; i++)
	{
		if (my_id == ROOT)
			temp_len = strlen(seq2_arr[i]) + 1;
		MPI_Bcast(&temp_len, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(seq2_arr[i], temp_len, MPI_CHAR, ROOT, MPI_COMM_WORLD);
	}


	sequenceFormation(my_id, num_procs, seq1, seq2_arr, numOfSeq, weights);
	freeSeq2(seq2_arr, numOfSeq);
}

void freeAllocation(int i, int numberofsequence, int *cudaOffsets,
		double *scoreInCuda, char *arrForCudaSeq1, double *weightInCuda,
		int *cudaMutants, char **arrForCudaSeq2, double *threadScoreInCuda,
		double *processScore, int *processMutants, int *processOffsets,
		double *threadScore, int *treadMutant, int *threadOffsets,
		int *seq2_lens) {
	cudaFree(cudaOffsets);
	cudaFree(scoreInCuda);
	cudaFree(arrForCudaSeq1);
	cudaFree(weightInCuda);
	cudaFree(cudaMutants);
	for (i = 0; i < numberofsequence; i++)
		cudaFree(arrForCudaSeq2[i]);
	cudaFree(threadScoreInCuda);
	free(arrForCudaSeq2);
	free(processScore);
	free(processMutants);
	free(processOffsets);
	free(threadScore);
	free(treadMutant);
	free(threadOffsets);
	free(seq2_lens);
}

void sequenceFormation(int pid, int numOfprocess, char seq1[MAX_SQ1],char **seq2_arr, int numberofsequence, double weights[NUMBER_WEIGHT])
{
	int *processMutants, *processOffsets;
	int *treadMutant, *threadOffsets;
	int *seq2_lens,seq1_len;
	int i,j;
	int *cudaOffsets=0,*cudaMutants=0;
	double *processScore, *threadScore;
	double *weightInCuda=0,*scoreInCuda=0,*threadScoreInCuda=0;
	char** arrForCudaSeq2 = 0;
	char* arrForCudaSeq1 = 0;
	int ompMaxThread = omp_get_max_threads();

	// calculate the length sequence 2
	seq2_lens = (int*)malloc(numberofsequence*sizeof(int));
	seq1_len = strlen(seq1);
	// Allocation for threads
	treadMutant = (int *)malloc(numberofsequence * ompMaxThread * sizeof(int));
	threadOffsets = (int *)malloc(numberofsequence * ompMaxThread * sizeof(int));
	threadScore = (double *)malloc(numberofsequence * ompMaxThread * sizeof(double));
	// Allocation for processes
	processMutants = (int *)malloc(numberofsequence * sizeof(int));
	processOffsets = (int *)malloc(numberofsequence * sizeof(int));
	processScore = (double *)malloc(numberofsequence * sizeof(double));

	for(i=0;i<numberofsequence;i++)
	{
		processScore[i] = -INFINITY;
		seq2_lens[i] = strlen(seq2_arr[i]);

	}
	for(i=0;i<ompMaxThread*numberofsequence;i++)
		threadScore[i] = -INFINITY;
	// the start of the cuda

	arrForCudaSeq2 = (char**)malloc(numberofsequence*sizeof(char*));
	for(i=0;i<numberofsequence;i++)
		arrForCudaSeq2[i] = 0;

	sq1Init(seq1,&arrForCudaSeq1,seq1_len+1);
	initSeq2(seq2_arr,arrForCudaSeq2,numberofsequence);
	weightInit(weights,&weightInCuda,NUMBER_WEIGHT);
	resultFromThread(numberofsequence,&scoreInCuda,&cudaMutants,&cudaOffsets,processScore);



#pragma omp parallel
	{
		int processStart,processEnd,threadStart,threadEnd;
		int totalProcess,totalThread,totalOfCudaThread,numOfThread,tid;
		int mutant,offsets,exponent2;
		double score;
		int k,p;
		tid = omp_get_thread_num();
		numOfThread = omp_get_num_threads();
		if(tid == ROOT)
		{
			for(k=0;k<numberofsequence;k++)
			{
				totalProcess = (seq2_lens[k] * (seq1_len - seq2_lens[k]-1))/numOfprocess;
				processStart = totalProcess*pid;
				processEnd = totalProcess*(pid+1);
				totalThread = ((totalProcess)*(CUDA_PERCENTAGE))/100;
				threadStart = processStart+totalThread*tid;
				threadEnd = threadStart+totalThread;
				exponent2 = exponent2next(seq2_lens[k]);

				cudaErrorAssistant(arrForCudaSeq1,arrForCudaSeq2[k],seq1_len,seq2_lens[k],
						k,threadStart,threadEnd,weightInCuda,scoreInCuda,cudaMutants,cudaOffsets,exponent2);

			}
		}
		else
		{
			int localTid = tid-1;
			int localNumOfThread = numOfThread-1;
			for(k=0;k<numberofsequence;k++)
			{
				totalProcess = (seq2_lens[k] * (seq1_len - seq2_lens[k]-1))/numOfprocess;
				processStart = totalProcess*pid;
				processEnd = totalProcess*(pid+1);
				totalThread = ((totalProcess/localNumOfThread)*(100-CUDA_PERCENTAGE))/100;
				totalOfCudaThread = ((totalProcess)*(CUDA_PERCENTAGE))/100;
				threadStart = processStart+totalThread*localTid+totalOfCudaThread;
				threadEnd = threadStart+totalThread;

				for(p=threadStart;p<threadEnd;p++)
				{
					mutant = p/(seq1_len - seq2_lens[k]-1);
					offsets = p%(seq1_len - seq2_lens[k]-1);
					score = cpuFormation(seq1,seq2_arr[k],seq1_len,seq2_lens[k],mutant,offsets,weights);
					if(score == 0)
						printf("gucci\n");
					if(score > threadScore[localTid*numberofsequence+k])
					{
						threadOffsets[localTid*numberofsequence+k] = offsets;
						threadScore[localTid*numberofsequence+k] = score;
						treadMutant[localTid*numberofsequence+k] = mutant;
					}

				}

				if(localTid == k % localNumOfThread)
				{
					threadStart = (totalThread*localNumOfThread +totalOfCudaThread);
					for(p=threadStart;p<processEnd;p++)
					{
						score = cpuFormation(seq1,seq2_arr[k],seq1_len,seq2_lens[k],mutant,offsets,weights);
						offsets = p%(seq1_len - seq2_lens[k]-1);
						mutant = p/(seq1_len - seq2_lens[k]-1);

						if(score > threadScore[localTid*numberofsequence+k])
						{
							treadMutant[localTid*numberofsequence+k] = mutant;
							threadOffsets[localTid*numberofsequence+k] = offsets;
							threadScore[localTid*numberofsequence+k] = score;

						}

					}
				}
			}
		}

	}//end of OpenMP
	int idx =numberofsequence*(ompMaxThread-1);
	copyFromGpuToCpu(threadScore+idx,treadMutant+idx,threadOffsets+idx,scoreInCuda,cudaMutants,cudaOffsets,numberofsequence);


	for(i=0;i<numberofsequence;i++)
	{
		for(j=0;j<ompMaxThread;j++)
		{
			if( threadScore[i+numberofsequence*j] > processScore[i])
			{
				processMutants[i] = treadMutant[i+numberofsequence*j];
				processOffsets[i] = threadOffsets[i+numberofsequence*j];
				processScore[i] = threadScore[i+numberofsequence*j];

			}
		}

	}
	resultsToMainProcessRoot(pid,numOfprocess,numberofsequence,processOffsets,processMutants,processScore);

	freeAllocation(i, numberofsequence, cudaOffsets, scoreInCuda,
			arrForCudaSeq1, weightInCuda, cudaMutants, arrForCudaSeq2,
			threadScoreInCuda, processScore, processMutants, processOffsets,
			threadScore, treadMutant, threadOffsets, seq2_lens);
}

 int exponent2next(int n)
{
   	int count = 0;

    while( n != 0)
    {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}
int bothGroups(char ch1, char ch2, char *group)
{
	int inCh1 = 0;
	int inCh2 = 0;

	while (*group != '\0')
	{
		if (ch1 == *group)
			inCh1 = 1;
		if (ch2 == *group)
			inCh2 = 1;
		if (inCh1 && inCh2)
			return 1;
		group++;
	}

	return 0;
}

void resultsToMainProcessRoot(int my_id, int processNumber, int seqNumber, int *bestOffset, int *bestMutant, double *bestScore)
{
	int *offsets, *mutant;
	int i, j;
	double *alloftheScores;

	if (my_id == ROOT)
	{
		offsets = (int *)malloc(processNumber * seqNumber * sizeof(int));
		alloftheScores = (double *)malloc(processNumber * seqNumber * sizeof(double));
		mutant = (int *)malloc(processNumber * seqNumber * sizeof(int));
	}

	MPI_Gather(bestMutant, seqNumber, MPI_INT, mutant, seqNumber, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Gather(bestOffset, seqNumber, MPI_INT, offsets, seqNumber, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Gather(bestScore, seqNumber, MPI_DOUBLE, alloftheScores, seqNumber, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


	if (my_id == ROOT)
	{
		for (i = 0; i < processNumber * seqNumber; i += seqNumber)
		{
			for (j = 0; j < seqNumber; j++)
			{
				if (alloftheScores[i + j] > bestScore[j])
				{
					bestMutant[j] = mutant[i + j];
					bestOffset[j] = offsets[i + j];
					bestScore[j] = alloftheScores[i + j];
				}
			}
		}
		//save to the file
		FILE *f = fopen("output.txt", "w");
		if (f == NULL)
		{
		    printf("Error opening file!\n");
		    	return;
		}


		for (i = 0; i < seqNumber; i++)
		{
			printf("Best mutant: %d, Best offset :%d, \n",bestMutant[i]+1,bestOffset[i]);
			fprintf(f, "Best mutant: %d, Best offset :%d,\n",bestMutant[i]+1,bestOffset[i]);
		}

		fclose(f);


		free(alloftheScores);
		free(offsets);
		free(mutant);
	}
}



void freeSeq2(char **seq2_arr, int seqNumber)
{
	int i;
	for (i = 0; i < seqNumber; i++)
		free(*(seq2_arr + i));
	free(seq2_arr);
}

int main(int argc, char *argv[])
{

	int my_Id, num_Procs;
	double startTime, endTime;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_Id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_Procs);

	startTime = MPI_Wtime();
	mpiFunc(my_Id, num_Procs);
	endTime = MPI_Wtime();

	if (my_Id == ROOT)
		printf("Time : %lf\n", endTime - startTime);
	MPI_Finalize();
	return 0;
}

