/* CUDA C Program to to merge sort of a list in ascending order */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "helpers.cuh"
#include <assert.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void __device__ merge(float *list,int l,int m,int r);
void __global__ mergesort(float *list, int SIZE);
// Shared memory for the merge function

const char* dataInitialization = "data_init";
const char* commLarge = "comm_large";
const char* compLarge = "comp_large";
const char* correctnessCheck = "correctness_check";
const char* comm = "comm";
const char* comp = "comp";
const char* cudaMCPY = "cudaMemcpy";
const char* mainCali = "main";

//check whether a certain number is a power of 2
int isPowerOfTwo(int num){
	int i=0;
	int val=1;
	for(i=0;val<=num;i++){
		if((val=pow(2,i))==num){
			return 1;
		}
	}				
	return 0;	

}

//check if array is sorted
float correctness_check(float arr[], int size) {
  CALI_MARK_BEGIN(correctnessCheck);
  for (int i=0; i<size-1; i++) {
    if (arr[i+1] < arr[i]) {
      CALI_MARK_END(correctnessCheck);
      return 0;  // means it's not ordered correctly
    }
  }
  CALI_MARK_END(correctnessCheck);

  return 1;
}

int main(int argc, char **argv){

  CALI_MARK_BEGIN(mainCali);
  int SIZE = atoi(argv[2]);

	//check the condition that check that checks whether the size is a power of 2
	if(!isPowerOfTwo(SIZE)){
		fprintf(stderr,"This implementation needs the list size to be a power of two\n");
		exit(1);
	}
	
	//allocate a list
	float *list = (float *)malloc(sizeof(float)*SIZE);
	if(list==NULL){
		perror("Mem full");
		exit(1);
	}
	
	//generate some random values
  CALI_MARK_BEGIN(dataInitialization);
	for(int i=0;i<SIZE;i++){
		list[i]=rand()/(float)100000;
	}
  CALI_MARK_END(dataInitialization);	
 
	//print the input list
 /**
	printf("The input list is : \n");
	for(int i=0;i<SIZE;i++){
		printf("%.2f ",list[i]);
	} **/
	printf("\n\n");
	
	/********************************** CUDA stuff starts here *******************************/
  	
 
	//start measuring time
 /*
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);		
 */
	//pointers for memory allocation in cudaa
	float *list_cuda;
	
	//allocate memory in cuda
	checkCuda(cudaMalloc((void **)&list_cuda,sizeof(float)*SIZE));
	
	//copy memory from ram to cuda
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(commLarge);
	checkCuda(cudaMemcpy(list_cuda,list,sizeof(float)*SIZE,cudaMemcpyHostToDevice));
  CALI_MARK_END(commLarge);
  CALI_MARK_END(comm);
	
	//thread configurations
	int threadsPerBlock = atoi(argv[1]);
	int numBlocks=ceil(SIZE/(2.0f * threadsPerBlock));
 
	/* The reason to divide by 2 is because now we need a thread per two elements only*/
	
	//start measuring time for cuda kernel only
 /**
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);	
  **/	
	printf("Num Blocks: %d\n", numBlocks);
printf("Threads Per Block: %d\n", threadsPerBlock);

	//do sorting
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(compLarge);
	mergesort<<<numBlocks, threadsPerBlock, sizeof(float) * threadsPerBlock>>>(list_cuda, SIZE);


  CALI_MARK_END(compLarge);
  CALI_MARK_END(comp);	
	checkCuda(cudaGetLastError());

	//end measuring time for cuda kernel
 /**
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
 **/
		
	//copy the answer back from cuda ro ram
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(commLarge);
	checkCuda(cudaMemcpy(list,list_cuda,sizeof(float)*SIZE,cudaMemcpyDeviceToHost));
  CALI_MARK_END(commLarge);
  CALI_MARK_END(comm);

	//free the cuda memory
	checkCuda(cudaFree(list_cuda));
	
	//end measuring time
 /**
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
 **/
	
	/********************** CUDA stuff ends here ********************************/
  bool correct = correctness_check(list, SIZE);
  CALI_MARK_END(mainCali);
	
	//print the answer
  if (correct) {
  /**
    printf("The sorted list is : \n");
	    for(int i=0;i<SIZE;i++){
		    printf("%.2f ",list[i]);
	    } **/
    printf("sorted");
	  printf("\n\n");	
  }
	else {
    printf("The list is not sorted");
  }
	
	//print the time spent to stderr
 /**
	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000); 
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 
 **/
 
 
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", SIZE); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", threadsPerBlock); // The number of threads
    adiak::value("group_num", 10); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("data_init_time", dataInitialization);
    adiak::value("correctness_check", correctnessCheck);
    adiak::value("communication_time", comm);
    adiak::value("computation_time", comp);
    adiak::value("communication_large_time", commLarge);
    adiak::value("computation_large_time", compLarge);
    adiak::value("main_time", mainCali);
    adiak::value("cuda_memcpy", cudaMCPY);
    
	return 0;
}


/* merge two lists while sorting them in ascending order
* For example say there are two arrays 
* while one being 1 3 5 and the other being 2 4 7
* when merge they become 1 2 3 4 5 7
* When storing the two lists they are stored in same array and the
* two arrays are specified using the index of leftmost element, middle element and the last element
* For example say the two arrays are there in memory as a single array 1 3 5 2 4 7
* Here l=0 m=3 and r=5 specifies the two arrays separately
* */


__device__ void merge(float *list, float *temp, int left, int middle, int right) {

    // i is used for indexing elements in the left array and j is used for indexing elements in the right array
    int i = left;
    int j = middle;

    // k is the index for the temporary array
    int k = 0;

    // now merge the two lists in ascending order
    // check the first element remaining in each list and select the lowest one from them. Then put it into temp
    // put and increase the relevant index i or j
    while (i < middle && j <= right) {
        if (list[i] < list[j]) {
            temp[k] = list[i];
            i++;
        } else {
            temp[k] = list[j];
            j++;
        }
        k++;
    }

    // if there are remaining ones in an array, append those to the end
    while (i < middle) {
        temp[k] = list[i];
        i++;
        k++;
    }
    while (j <= right) {
        temp[k] = list[j];
        j++;
        k++;
    }

    // now copy back the sorted array in temp to the original
    for (i = left, k = 0; i <= right; i++, k++) {
        list[i] = temp[k];
    }
}

// carry out merge sort ascending
__global__ void mergesort(float *list, int SIZE) {
    // calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory for the merge function
    extern __shared__ float temp[];

    // step means the distance to the next list
    // loop till the merging happens for a list of the size of the original list
    int step = 1;
    while (step < SIZE - 1) {
        if (tid % step == 0 && tid * 2 < SIZE) {
            // calculate the index of the first element of the first list
            int left = 2 * tid;

            // calculate the index of the first element of the second list
            int middle = 2 * tid + step;

            // calculate the last element of the second list
            int right = 2 * tid + 2 * step - 1;

            // merge the two lists
            merge(list, temp, left, middle, right);
        }

        // next list size
        step = step * 2;

        // synchronize all threads
        __syncthreads();
    }
}