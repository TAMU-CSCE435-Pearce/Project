#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

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


/* Main Alg */




/* Verification */



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
    int unsortedArray[NUM_VALS];
    int* d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void**)&d_unsortedArray, NUM_VALS * sizeof(int));
    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS);
    cudaDeviceSynchronize();

    // Copy the result back to the host and free memory
    cudaMemcpy(unsortedArray, d_unsortedArray, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_unsortedArray);  //Leaves "unsortedArray" as the array with all the data on CPU

   
    //Main Alg

    //Verify



    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}

}