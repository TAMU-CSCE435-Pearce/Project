#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Define Caliper region names */
const char* mainFunction = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check ";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm _small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";

/* Data generation */
__global__ void generateData(int* dataArray, int size, int inputType) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        switch (inputType) {
        case 0: {//Random    
            if(idx < size) {
                unsigned int x = 12345687 + idx;
                x ^= (x << 16);
                x ^= (x << 25);
                x ^= (x << 4);
                dataArray[idx] = abs(static_cast<int>(x) % size);
            }
            break;
        }
        case 1: {//Sorted
            if (idx < size) {dataArray[idx] = idx;}
            break;
        }
        case 2: { //Reverse sorted
             if (idx < size) {dataArray[idx] = size - 1 - idx;}
            break;
        }
    }
}


/* Main Alg Stuff */
// CUDA kernel to select and gather samples from the array
__global__ void selectSamples(int* array, int* samples, int size, int sampleSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (sampleSize-1)) {
        int step = size / sampleSize;
        samples[tid] = array[((tid+1) * step)];
    }
}

// Compare function for sorting the samples
__device__ bool compareSamples(const int& a, const int& b) {
    return a > b;
}

// CUDA kernel to sort the samples
__global__ void sortSamples(int* samples, int sampleSize) {
    if (threadIdx.x < sampleSize - 1) {
        for (int i = threadIdx.x; i < sampleSize; i++) {
            for (int j = i + 1; j < sampleSize; j++) {
                if (compareSamples(samples[i], samples[j])) {
                    int temp = samples[i];
                    samples[i] = samples[j];
                    samples[j] = temp;
                }
            }
        }
    }
}

// CUDA kernel to calculate the data offsets for grouping
__global__ void partitionDataCalculation(int* array, int* samples, int* bucketOffsets, int size, int sampleSize) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sampleSize) {
        int my_bucket = tid;
        for(int i = 0; i < size; i++) {
            if(array[i] < samples[my_bucket]) {
                atomicAdd(&bucketOffsets[my_bucket], 1);
            }
        }   
    }
}

// CUDA kernel to partition the data into buckets based on the samples
__global__ void partitionData(int* unsortedData, int* groupedData, int* startPosition, int* pivots, int numThreads, int NUM_VALS, int* expandedPivots, int* expandedStarts) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numThreads) {
        for(int i = 0; i < numThreads-1; i++) {
            expandedPivots[i] = pivots[i];
        }
        expandedPivots[numThreads-1] = NUM_VALS; 

        for(int i = 1; i < numThreads; i++) {
            expandedStarts[i] = startPosition[i-1];
        }
        expandedStarts[0] = 0; 
        
        int previousCutoff = (tid == 0) ? 0 : expandedPivots[tid-1];

        for(int i = 0; i < NUM_VALS; i++) {
            if(unsortedData[i] < expandedPivots[tid] && unsortedData[i] >= previousCutoff){
                groupedData[expandedStarts[tid]] = unsortedData[i];
                expandedStarts[tid]++;
            }
        }
    }
    
}

// CUDA kernel to sort each bucket using insertion sort
__global__ void sortBuckets(int* array, int* bucketOffsets, int size, int NUM_VALS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int bucket = tid;
        int start = (bucket == 0) ? 0 : bucketOffsets[bucket - 1];
        int end = (bucket == (size-1)) ? (NUM_VALS) : bucketOffsets[bucket];
        for (int i = start + 1; i < end; i++) {
            int key = array[i];
            int j = i - 1;
            while (j >= start && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }
            array[j + 1] = key;
        }
    }
}


/* Verification */
// CUDA kernel to check if the array is sorted
__global__ void checkArraySorted(int* array, bool* isSorted, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
}


/* Program main */
int main(int argc, char *argv[]) {
    int sortingType;

    sortingType = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Input sorting type: %d\n", sortingType);
    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    /* Data generation */
    int* d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void**)&d_unsortedArray, NUM_VALS * sizeof(int));
    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS, sortingType);
    cudaDeviceSynchronize();
    CALI_MARK_END(data_init);


    /* Main Alg */
    //arraySize = NUM_VALS; 
    //blockSize = THREADS;     // CUDA block size
    int sampleSize = THREADS;     // Number of samples
    int* d_samples;
    int* d_bucketOffsets;
    int* d_groupedData;
    int* d_expandedPivots;
    int* d_expandedStarts;
   
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_samples, (sampleSize-1) * sizeof(int));
    cudaMalloc((void**)&d_bucketOffsets, sampleSize * sizeof(int));
    cudaMalloc((void**)&d_groupedData, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&d_expandedPivots, THREADS * sizeof(int));
    cudaMalloc((void**)&d_expandedStarts, THREADS * sizeof(int));

    // Launch the kernel to select and gather samples
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    selectSamples<<<BLOCKS, THREADS>>>(d_unsortedArray, d_samples, NUM_VALS, sampleSize);

    // Launch the kernel to sort the samples
    sortSamples<<<1, 1>>>(d_samples, (sampleSize-1));

    // Launch the kernel to count the data in each bucket
    partitionDataCalculation<<<BLOCKS, THREADS>>>(d_unsortedArray, d_samples, d_bucketOffsets, NUM_VALS, (sampleSize-1));
    CALI_MARK_END(comp_small);

    CALI_MARK_BEGIN(comp_large);
    // Launch the kernel to partition the data into buckets
    partitionData<<<BLOCKS, THREADS>>>(d_unsortedArray, d_groupedData, d_bucketOffsets, d_samples, THREADS, NUM_VALS, d_expandedPivots, d_expandedStarts);

    // Launch the kernel to sort each bucket using insertion sort
    sortBuckets<<<BLOCKS, THREADS>>>(d_groupedData, d_bucketOffsets, THREADS, NUM_VALS);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // Copy data back to the host
    int sortedArray[NUM_VALS];
    cudaMemcpy(sortedArray, d_groupedData, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);


    CALI_MARK_BEGIN(correctness_check);
    /* Verify Correctness */ 
    bool isSorted[NUM_VALS - 1];
    bool* d_isSorted;
    cudaMalloc((void**)&d_isSorted, (NUM_VALS - 1) * sizeof(bool));
    checkArraySorted<<<BLOCKS, THREADS>>>(d_groupedData, d_isSorted, NUM_VALS);
    cudaDeviceSynchronize();

    cudaMemcpy(isSorted, d_isSorted, (NUM_VALS - 1) * sizeof(bool), cudaMemcpyDeviceToHost);

   // Verify if the array is sorted
    bool sorted = true;
    for (int i = 0; i < NUM_VALS - 1; i++) {
        if (!isSorted[i]) {
            sorted = false;
            break;
        }
    }
    CALI_MARK_END(correctness_check);

    // Free GPU memory
    cudaFree(d_samples);
    cudaFree(d_bucketOffsets);
    cudaFree(d_unsortedArray);
    cudaFree(d_groupedData);
    cudaFree(d_expandedPivots);
    cudaFree(d_expandedStarts);
    cudaFree(d_isSorted);

    CALI_MARK_END(mainFunction);

    if (sorted) {printf("The array is sorted!\n" );} 
    else {printf("The array is not sorted!\n");}

    string inputType;
    switch (sortingType) {
    case 0: {
        inputType = "Randomized";
        break; }
    case 1: {
        inputType = "Sorted";
        break; }
    case 2: {
        inputType = "Reverse Sorted";
        break; }
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 16); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI & Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}

