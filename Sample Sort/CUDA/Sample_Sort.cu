#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

using std::vector;

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
__global__ void generateData(int* dataArry, int size, int inputType) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        switch (inputType) {
        case 0: {//Random    
            if(idx < size) {
                curandState state;
                curand_init(12345678, idx, 0, &state);
                dataArray[idx] = curand(&state) % NUM_VALS;
            }
            break;
        }
        case 1: {//Sorted
            if (idx < size) {dataArray[idx] = idx;}
            break;
        }
        case 2: { //Reverse sorted
             if (idx < size) {dataArrau[idx] = size - 1 - idx;}
            break;
        }
    }
}


/* Main Alg Stuff */
// CUDA kernel to select and gather samples from the array
__global__ void selectSamples(int* array, int* samples, int size, int sampleSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sampleSize) {
        int step = size / (sampleSize - 1);
        samples[tid] = array[tid * step];
    }
}

// Compare function for sorting the samples
__device__ bool compareSamples(const int& a, const int& b) {
    return a < b;
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

// CUDA kernel to partition the data into buckets based on the samples
__global__ void partitionData(int* array, int* samples, int* bucketOffsets, int size, int sampleSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int value = array[tid];
        int bucket = 0;
        while (bucket < sampleSize - 1 && value >= samples[bucket]) {
            bucket++;
        }
        int nextBucketOffset = (bucket == 0) ? 0 : bucketOffsets[bucket - 1];
        int indexInBucket = tid - nextBucketOffset;
        array[tid] = bucket;
    }
}

// CUDA kernel to sort each bucket using insertion sort
__global__ void sortBuckets(int* array, int* bucketOffsets, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int bucket = array[tid];
        int start = (bucket == 0) ? 0 : bucketOffsets[bucket - 1];
        int end = bucketOffsets[bucket];
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

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();


    /* Data generation */
    int* d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void**)&d_unsortedArray, NUM_VALS * sizeof(int));
    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS);
    cudaDeviceSynchronize(); //Leaves "d_unsortedArray" as an array with data on GPU


    /* Main Alg */
    //arraySize = NUM_VALS; 
    //blockSize = THREADS;     // CUDA block size
    int sampleSize = 4 * THREADS;     // Number of samples
    int samples[sampleSize];
    int* d_samples;
    int bucketOffsets[sampleSize - 1];
    int* d_bucketOffsets;
    int sortedArray[NUM_VALS];

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_samples, sampleSize * sizeof(int));
    cudaMalloc((void**)&d_bucketOffsets, (sampleSize - 1) * sizeof(int));

    // Launch the kernel to select and gather samples
    selectSamples<<<BLOCKS, THREADS>>>(d_unsortedArray, d_samples, NUM_VALS, sampleSize);

    // Launch the kernel to sort the samples
    sortSamples<<<1, 1>>>(d_samples, sampleSize);

    // Launch the kernel to partition the data into buckets
    partitionData<<<BLOCKS, THREADS>>>(d_unsortedArray, d_samples, d_bucketOffsets, NUM_VALS, sampleSize);

    // Launch the kernel to sort each bucket using insertion sort
    sortBuckets<<<BLOCKS, THREADS>>>(d_unsortedArray, d_bucketOffsets, NUM_VALS);
    cudaDeviceSynchronize();



    // Copy data back to the host
    cudaMemcpy(sortedArray, d_unsortedArray, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);

    /* Verify Correctness */ 
    bool isSorted[NUM_VALS - 1];
    bool* d_isSorted;
    cudaMalloc((void**)&d_isSorted, (NUM_VALS - 1) * sizeof(bool));
    checkArraySorted<<<BLOCKS, THREADS>>>(d_unsortedArray, d_isSorted, NUM_VALS);

    // Free GPU memory
    cudaFree(d_samples);
    cudaFree(d_bucketOffsets);
    cudaFree(d_unsortedArray);
    cudaFree(d_isSorted);

   // Verify if the array is sorted
    bool sorted = true;
    for (int i = 0; i < NUM_VALS - 1; i++) {
        if (!isSorted[i]) {
            sorted = false;
            break;
        }
    }

    if (sorted) {printf("The array is sorted!" );} 
    else {printf("The array is not sorted!");}

    printf("Array: ");
    for(int i = 0; i < NUM_VALS; i++) {
        printf("%d, ", sortedArray[i]);
    }

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}

