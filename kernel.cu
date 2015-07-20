/* SorensonPar.cu
   Parallel Implementation of Algorithm 4.1
   as discussed in Sorenson and Parberry's
   1994 paper "Two Fast Parallel Prime Number
   Sieves".

   Authors:
   Daniel Anzaldo
   David Frank
   Antonio Lanfranchi

   Constant Memory Version.
*/

// Visual Studio Dependencies (Can be commented out)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// C dependencies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// C++ dependencies
#include <algorithm>

// Preprocessor Constants
#define CONST_MEM_SIZE 64000
#define SHARED_MEM_SIZE 48000

typedef unsigned long long big;

// GLOBAL VARIABLES--------------------------------------
bool * S;	// Global shared bit array of numbers up to N
int P;		// Global number of processors
__constant__ bool c_S[CONST_MEM_SIZE];	// table of the first 64000 primes

bool check_cuda_status = false; // turn to false when running on circe

/* These are for tracking time */
struct timezone myTimezone;	
struct timeval startTime, endTime;

// HOST FUNCTION HEADERS---------------------------------

/*	EratosthenesSieve
	HELPER: for Algorithm 4.1 Sequential Portion
	The most basic form of generating primes.
	Used to help find the first k primes.
	Returns the k-th prime.
*/
void EratosthenesSieve(big n);

/*	Algorithm 4.1 Sequential Portion
	Running Time: O(sqrt(n))
	Space: O(sqrt(n)) up to O(sqrt(n)/log log n)
*/
cudaError_t algorithm4_1(big n);

/*	Algorithm 4.1 Helper: Parallel Sieve
	All CUDA-related functionality goes here.
	This code will change for different kernel versions.
*/
cudaError_t parallelSieve(big n, big range);

/*	Frees the memory allocated on the device and returns any errors */
cudaError_t cleanup(bool *d_S, cudaError_t cudaStatus);

/*	Set a checkpoint and show the total running time in seconds */
double report_running_time(const char *arr);


// DEVICE MATH FUNCTIONS---------------------------------

/*	sqrt_d
	Device version of the Square Root Function
	Babylonian Method
*/
__device__ big sqrt_d(big a)
{
	big root = a/2;
   
#pragma unroll
	for (big n = 0; n < 10; n++)
	{
		root = 0.5 * (root + (a/root));
	}
   
	return root;
}

__device__ big min_d(big a, big b)
{
	return (a < b) ? a : b;
}

__device__ big max_d(big a, big b)
{
	return (a > b) ? a : b;
}


// ALGORITHM 4.1 KERNEL VERSIONS-------------------------
/*	
	Parallel Sieve Kernel
	With Constant Memory
*/
__global__ void parallelSieveKernel(
	big n, big range, bool *d_S)
{
	// Shared memory use for S in range of thread
	__shared__ bool sievingRange[SHARED_MEM_SIZE];

	// Thread id
	big i = threadIdx.x + blockIdx.x * blockDim.x;
	int j;	// Constant Memory Accessor
	big f;	// Multiplier of prime in const memory

	// Find left and right range
	big L = range * i;
	big R = range + L;

	if (i <= n)
	{
		// Thread Sieve
		for (j = 0; j < CONST_MEM_SIZE; j++)
		{
			// For each prime number in constant memory
			if (j <= CONST_MEM_SIZE && c_S[j])
			{
				/* Calculate smallest multiple j * f
				within the range of [L,R]*/
				f = L / j;

				// Write results to shared memory
				while ((j * f - CONST_MEM_SIZE) <= R)
				{
					sievingRange[j * f - CONST_MEM_SIZE] = false;
					f++;
				}
			}
		}

		// Commit range changes to global memory
		for (j = L; j < R; j++)
		{
			d_S[j] = sievingRange[j - L];
		}
		__syncthreads();
	}

	return;
}

/*	MAIN
	To run this add the ff. args:
	1. N = the number up to which you're sieving
*/
int main(int argc, char **argv)
{
	big N = (big)strtoull(argv[1], NULL, 10);
	
	printf("Find primes up to: %llu\n\n", N);

	/* Program's Limitation */
	if (sqrtl((long double)N) > CONST_MEM_SIZE)
	{
		printf("ERROR: sqrt(N) exceeds available constant memory.");
		return EXIT_FAILURE;
	}

	S = new bool[N]; //(bool*)malloc(N * sizeof(bool));

	/* start counting time */
	gettimeofday(&startTime, &myTimezone);

	cudaError_t x = algorithm4_1(N);

	/* check the total running time */ 
	report_running_time("CUDA Parallel Sieve");

	if (check_cuda_status)
	{
		if (x != cudaSuccess) {
			printf("CUDA Parallel Sieve failed to execute!");
			return EXIT_FAILURE;
		}
	}

	// Display the primes.
	for (int i = 0; i < N; i++)
		if (S[i]) printf("%llu ", i);

	delete[] S;
	return EXIT_SUCCESS;
}


// HOST FUNCTION DEFINITIONS-----------------------------

void EratosthenesSieve(big n)
{
	big kthPrime = 0;

	// 0 and 1 are non-primes.
	S[0] = S[1] = false;
	for (big i = 2; i < n; i++)
		S[i] = true;

	// Simple Sieving Operation. MODIFIED.
	for (big i = 2; i < (big)sqrtl(n); i++)
		if (S[i])
		{
			int j;
			for (j = i*i; j < CONST_MEM_SIZE; j += i)
				S[j] = false;
		}

	return;
}

cudaError_t algorithm4_1(big n)
{
	/* VARIABLES */
	big range;
	big sqrt_N = (big)sqrtl((long double)n);

	/* Find the first k primes up to sqrt(N) */
	EratosthenesSieve(n);

	/* Delta = ceil(n/p) */
	range = (big)ceill(n / (long double)P);

	/* PARALLEL PART */
	cudaError_t parallelStatus = parallelSieve(n, range);
	if (check_cuda_status)
	{
		if (parallelStatus != cudaSuccess) {
			fprintf(stderr, "parallelSieve() failed!");
		}
	}

	return parallelStatus;
}

cudaError_t parallelSieve(big n, big range)
{
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* The Number Field S
		will be migrated to GLOBAL memory
		OPTIMIZATION: ranges will be migrated to SHARED memory
		OPTIMIZATION: [0, sqrt(n)] will be migrated to CONSTANT memory
	*/
	bool * d_S = NULL;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
			return cudaStatus;
		}
	}

	// Measure start time for CUDA portion
	cudaEventRecord(start, 0);

	// CUDA Memory Allocations.
	cudaStatus = cudaMalloc((void**)&d_S, (n - CONST_MEM_SIZE) * sizeof(bool));
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed on number field S!\n");
			return cleanup(d_S, cudaStatus);
		}
	}

	// cudaMemCpy -> Device
	// Pointer Arithmetic Copy: Hope it works!
	cudaStatus = cudaMemcpy(d_S, S + CONST_MEM_SIZE, (n - CONST_MEM_SIZE) * sizeof(bool), cudaMemcpyHostToDevice);
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! S->d_S.\n");
			return cleanup(d_S, cudaStatus);
		}
	}

	// cudaMemCpyToSymbol -> Device Constant Memory
	cudaStatus = cudaMemcpyToSymbol(c_S, S, CONST_MEM_SIZE * sizeof(bool));
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpyToSymbol failed! S->c_S.\n");
			return cleanup(d_S, cudaStatus);
		}
	}

	// Kernel Call
	dim3 gridSize(ceill(ceill(sqrt(n))/256), 1, 1);
	dim3 blockSize(256, 1, 1);

	//parallelSieveKernel<<<gridSize, blockSize>>>(n, k, m, wheel, range, d_S);
	parallelSieveKernel<<<gridSize, blockSize>>>(n, range, d_S);

	cudaStatus = cudaGetLastError();
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "parallelSieveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cleanup(d_S, cudaStatus);
		}
	}

	cudaStatus = cudaDeviceSynchronize();
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return cleanup(d_S, cudaStatus);
		}
	}

	// cudaMemCpy -> Host
	// Pointer Arithmetic Return: Hope it works!
	cudaStatus = cudaMemcpy(S + CONST_MEM_SIZE, d_S, (n - CONST_MEM_SIZE) * sizeof(bool), cudaMemcpyDeviceToHost);
	if (check_cuda_status)
	{
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! d_S->S.\n");
			return cleanup(d_S, cudaStatus);
		}
	}

	// Measure stop time for CUDA portion
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms\n", elapsedTime);

	// cudaFree
	return cleanup(d_S, cudaStatus);
}

cudaError_t cleanup(bool *d_S, cudaError_t cudaStatus)
{
	cudaFree(d_S);
	return cudaStatus;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(const char *arr) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &myTimezone);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for %s: %ld.%06ld sec\n\n", arr, sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}
