#include "common.h"

__global__ void quicksort_step(float *dev_values, unsigned int* partitionBegin, unsigned int* partitionEnd)
{
    unsigned int start;
    //start = threadIdx.x + blockDim.x * blockIdx.x;
    start = partitionBegin[threadIdx.x];
    unsigned int end = partitionEnd[start];
    if(start>=end)
        return;
    unsigned int pivot = start;
    unsigned int left = start;
    for (int i=start;i<end;i++) {
        if(dev_values[i] < dev_values[pivot]) {
            int tmp = dev_values[i];
            //dev_values[i] = dev_values[left];
            //dev_values[left] = tmp;
            left = left+1;
        }
    }
    int tmp = dev_values[pivot];
    dev_values[left] = dev_values[pivot];
    dev_values[pivot] = tmp;

    partitionEnd[start] = left;
    //partitionEnd[left+1] = end;
}

/**
 * Inplace bitonic sort using CUDA.
 */
void quicksort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS)
{
    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */
    unsigned int* partitionEnd;
    unsigned int* partitionBegin;
    size_t partsize = NUM_VALS * sizeof(unsigned int);
    cudaMalloc((void**) &partitionEnd, partsize);
    cudaMalloc((void**) &partitionBegin, partsize);
    cudaMemset(partitionEnd, 0, partsize);
    cudaMemset(partitionEnd, NUM_VALS, partsize);
    
    // Start event
    CALI_MARK_BEGIN(bitonic_sort_step_region);
    int i=0;
    while(i<=NUM_VALS) {
        //unsigned int size = int(NUM_VALS / pow(double(2), double(i)));
        quicksort_step<<<blocks,threads>>>(dev_values, partitionBegin, partitionEnd);
        cudaDeviceSynchronize();
        i ++;
    }
    // End Event
    cudaDeviceSynchronize();
    cudaFree(partitionEnd);
    cudaFree(partitionBegin);
    CALI_MARK_END(bitonic_sort_step_region);
}