#include "common.h"


/*__device__ int partition(int start, int end, int partitionIdx, float* dev_values, unsigned int*partitionBegin, unsigned int*partitionEnd, unsigned int* newBegin, unsigned int* newEnd) {
    unsigned int pivot = start;
    unsigned int left = start;
    for (int i=start;i<end;i++) {
        if(dev_values[i] < dev_values[pivot]) {
            int tmp = dev_values[i];
            dev_values[i] = dev_values[left];
            dev_values[left] = tmp;
            left = left+1;
        }
    }
    int tmp = dev_values[pivot];
    dev_values[left] = dev_values[pivot];
    dev_values[pivot] = tmp; // Pivot value is now at index left
    if(left>start) {
        newBegin[partitionIdx] = start;
        newBegin[partitionIdx] = left;
        partitionIdx += 1;
    }
    if(left<end-1) {
        newBegin[partitionIdx] = left+1;
        newBegin[partitionIdx] = end;
        partitionIdx += 1;
    }
    return partitionIdx;
}

__global__ void quicksort_step(float *dev_values, unsigned int* partitionBegin, unsigned int* partitionEnd, unsigned int* newBegin, unsigned int* newEnd)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    //while idx < block

    unsigned int i = idx;
    unsigned int partitionIdx = idx;
    while (i < idx + blockDim.x) {
        if (partitionEnd[i] == 0) {
            break;
        }
        unsigned int start = partitionBegin[i];
        unsigned int end = partitionEnd[i];
        if(end>start) {
            partitionIdx = partition(start, end, partitionIdx, dev_values, partitionBegin, partitionEnd, newBegin, newEnd);
        }
        i ++;
    }
}

__global__ void memset2(int* buffer, int val, size_t size) {
    memset(buffer, val, size);
}

__global__ void printfd(unsigned int* buffer, int n) {
    for(int i=0;i<n;i++) {
        printf("%u  ", buffer[i]);
    }
    printf("\n");
}

__global__ void initBuffer(unsigned int* partitionEnd, unsigned int n) {
    unsigned int idx = threadIdx.x * blockDim.x;
    printf("%u, %u, %u, %u\n", threadIdx.x, blockDim.x, blockIdx.x, idx);
    partitionEnd[idx] = n;
}*/

__global__ void oddeven_dev(float* data, unsigned int N, bool odd) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(odd && (idx%2==0))
        return;
    if(!odd && (idx)%2!=0)
        return;
    
    unsigned int nxt = idx+1;
    if(nxt>=N)
        return;

    if(data[idx]>data[nxt]) {
        float tmp = data[idx];
        data[idx] = data[nxt];
        data[nxt] = tmp;
    }
}

void oddeven(float* values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS) {
    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    for(int i=0;i<NUM_VALS;i++) {
        bool isOdd = i%2!=0;
        oddeven_dev<<<blocks, threads>>>(dev_values, NUM_VALS, isOdd);
    }
}


/*void quicksort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS)
{
    dim3 blocks(BLOCKS,1);    
    dim3 threads(THREADS,1);  
    unsigned int* partitionEnd;
    unsigned int* partitionBegin;
    unsigned int* beginBuffer;
    unsigned int* endBuffer;
    size_t partsize = NUM_VALS * sizeof(unsigned int);
    cudaMalloc((void**) &partitionEnd, partsize);
    cudaMalloc((void**) &partitionBegin, partsize);
    cudaMalloc((void**) &beginBuffer, partsize);
    cudaMalloc((void**) &endBuffer, partsize);
    cudaMemset(partitionBegin, 0, partsize);
    cudaMemset(partitionEnd, 0, partsize);
    cudaMemset(beginBuffer, 0, partsize);
    cudaMemset(endBuffer, 0, partsize);
    cudaDeviceSynchronize();
    initBuffer<<<blocks,1>>>(partitionEnd, NUM_VALS);
    cudaDeviceSynchronize();
    // Start event
    //CALI_MARK_BEGIN(bitonic_sort_step_region);
    unsigned int i=0;
    while(i<NUM_VALS) {
        cudaMemset(beginBuffer, 0, partsize);
        cudaMemset(endBuffer, 0, partsize);
        cudaDeviceSynchronize();
        quicksort_step<<<blocks,threads>>>(dev_values, partitionBegin, partitionEnd, beginBuffer, endBuffer);
        cudaDeviceSynchronize();
        printfd<<<1,1>>>(beginBuffer, NUM_VALS);
        printfd<<<1,1>>>(endBuffer, NUM_VALS);
        printf("\n");
        cudaMemcpy(partitionBegin, beginBuffer, partsize, cudaMemcpyHostToHost);
        cudaMemcpy(partitionEnd, endBuffer, partsize, cudaMemcpyHostToHost);
        cudaDeviceSynchronize();
        i ++;
    }
    // End Event
    cudaDeviceSynchronize();
    cudaFree(partitionEnd);
    cudaFree(partitionBegin);
    cudaFree(beginBuffer);
    cudaFree(endBuffer);
    //CALI_MARK_END(bitonic_sort_step_region);
}*/