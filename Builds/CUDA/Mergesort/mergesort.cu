#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *main = "main";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *cuda_memcpy_host_to_device = "cuda_memcpy_host_to_device";
const char *cuda_memcpy_device_to_host = "cuda_memcpy_device_to_host";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
int THREADS;
int BLOCKS;
int NUM_VALS;
int num_calls = 0;

#define SHARED 8000
// Device function called locally
__device__ void solve(int tempList[2][SHARED / sizeof(int)], int left_start, int right_start, int old_left_start, int my_start, int my_end, int left_end, int right_end, int headLoc, int walkLen)
{
	int minRemaining = min(right_end - right_start, left_end - left_start);

	for (int i = 0; i < minRemaining; i++)
	{
		if (tempList[current_list][left_start] < tempList[current_list][right_start])
		{
			tempList[!current_list][headLoc] = tempList[current_list][left_start];
			left_start++;
		}
		else
		{
			tempList[!current_list][headLoc] = tempList[current_list][right_start];
			right_start++;
		}

		headLoc++;
	}
}

/* Mergesort definition.  Takes a pointer to a list of floats, the length
  of the list, and the number of list elements given to each thread.
  Puts the list into sorted order */
__global__ void Device_Merge(int *d_list, int length, int elementsPerThread)
{
	int my_start, my_end;
	int left_start, right_start;
	int old_left_start;
	int left_end, right_end;
	int headLoc;
	short current_list = 0;

	__shared__ int tempList[2][SHARED / sizeof(int)];

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < elementsPerThread; i++)
	{
		if (index + i < length)
		{
			tempList[current_list][elementsPerThread * threadIdx.x + i] = d_list[index + i];
		}
	}

	__syncthreads();

	for (int walkLen = 1; walkLen < length; walkLen *= 2)
	{
		my_start = elementsPerThread * threadIdx.x;
		my_end = my_start + elementsPerThread;
		left_start = my_start;

		while (left_start < my_end)
		{
			old_left_start = left_start;

			if (left_start > my_end)
			{
				left_start = length;
				break;
			}

			left_end = left_start + walkLen;
			if (left_end > my_end)
			{
				left_end = length;
			}

			right_start = left_end;

			if (right_start > my_end)
			{
				right_end = length;
			}

			right_end = right_start + walkLen;

			if (right_end > my_end)
			{
				right_end = length;
			}

			solve(tempList, left_start, right_start, old_left_start, my_start, my_end, left_end, right_end, headLoc, walkLen);
			left_start = old_left_start + 2 * walkLen;
			current_list = !current_list;
		}
	}

	__syncthreads();

	for (int i = 0; i < elementsPerThread; i++)
	{
		if (index + i < length)
		{
			d_list[index + i] = tempList[current_list][elementsPerThread * threadIdx.x + i];
		}
	}

	__syncthreads();

	return;
}

/* Mergesort definition.  Takes a pointer to a list of floats.
  the length of the list, the number of threads per block, and
  the number of blocks on which to execute.
  Puts the list into sorted order in-place.*/
void MergeSort(int *h_list, int len, int threadsPerBlock, int blocks)
{

	// device copy
	int *d_list;
	// Allocate space for device copy
	cudaMalloc((void **)&d_list, len * sizeof(int));
	// copy input to device
	CALI_MARK_BEGIN(comm);
	CALI_MARK_BEGIN(comm_large);
	CALI_MARK_BEGIN(cuda_memcpy_host_to_device);
	cudaMemcpy(d_list, h_list, len * sizeof(int), cudaMemcpyHostToDevice);
	CALI_MARK_END(cuda_memcpy_host_to_device);
	CALI_MARK_END(comm_large);
	CALI_MARK_END(comm);
	int elementsPerThread = ceil(static_cast<float>(NUM_VALS) / (THREADS * BLOCKS));
	// Launch a Device_Merge kernel on GPU
	CALI_MARK_BEGIN(comp);
	CALI_MARK_BEGIN(comp_large);
	Device_Merge<<<BLOCKS, THREADS>>>(d_list, NUM_VALS, elementsPerThread);
	CALI_MARK_END(comp);
	CALI_MARK_END(comp_large);

	CALI_MARK_BEGIN(comm);
	CALI_MARK_BEGIN(comm_large);
	CALI_MARK_BEGIN(cuda_memcpy_device_to_host);
	cudaMemcpy(h_list, d_list, len * sizeof(int), cudaMemcpyDeviceToHost);
	CALI_MARK_END(cuda_memcpy_device_to_host);
	CALI_MARK_END(comm_large);
	CALI_MARK_END(comm);
	cudaFree(d_list);
}

int main(int argc, char *argv[])
{

	THREADS = atoi(argv[1]);
	NUM_VALS = atoi(argv[2]);
	BLOCKS = NUM_VALS / THREADS;
	cali::ConfigManager mgr;
	mgr.start();
	CALI_MARK_BEGIN(main);
	int *h_list = (int *)malloc(NUM_VALS * sizeof(int));
	CALI_MARK_BEGIN(data_init);
	for (int i = 0; i < NUM_VALS; i++)
	{
		h_list[i] = rand() % NUM_VALS;
	}
	CALI_MARK_END(data_init);

	MergeSort(h_list, NUM_VALS, THREADS, BLOCKS);

	CALI_MARK_BEGIN(correctness_check);
	for (int i = 0; i < NUM_VALS - 1; i++)
	{
		if (h_list[i] > h_list[i + 1])
		{
			cout << "Failed" << endl;
		}
	}
	cout << "Success" << endl;
	CALI_MARK_END(correctness_check);
	adiak::init(NULL);
	adiak::launchdate();							 // launch date of the job
	adiak::libraries();								 // Libraries used
	adiak::cmdline();								 // Command line used to launch the job
	adiak::clustername();							 // Name of the cluster
	adiak::value("Algorithm", "Mergesort");			 // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
	adiak::value("ProgrammingModel", "CUDA");		 // e.g., "MPI", "CUDA", "MPIwithCUDA"
	adiak::value("Datatype", "int");				 // The datatype of input elements (e.g., double, int, float)
	adiak::value("SizeOfDatatype", sizeof(int));	 // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("InputSize", NUM_VALS);			 // The number of elements in input dataset (1000)
	adiak::value("InputType", "Random");			 // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
	adiak::value("num_threads", THREADS);			 // The number of CUDA or OpenMP threads
	adiak::value("num_blocks", BLOCKS);				 // The number of CUDA blocks
	adiak::value("group_num", 18);					 // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	CALI_MARK_END(main);
	mgr.stop();
	mgr.flush();

	return 0;
}