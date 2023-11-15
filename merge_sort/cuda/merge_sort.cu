#include <iostream>
#include <vector>
#include <algorithm>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <stdlib.h>
#include <stdio.h>

int THREADS;
int BLOCKS;
int NUM_VALS;  

// Generate data
void generate_data(size_t size, int *data) {
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand() % (size * 10);
    }
}

// Correctness check
bool is_correct(size_t size, int *data) {
    for (size_t i = 1; i < size; i++) {
        if (data[i - 1] > data[i]) {
            return false;
        }
    }
    return true;
}

__device__ void deviceMerge(int *array, int *temp, int left, int right, int middle) {
    int left_idx = left;
    int merged_idx = left;
    int right_idx = middle+1;
    int k;

    // Sort left and right side of array into temp
    while ((left_idx <= middle) && (right_idx <= right)) {
        if (array[left_idx] <= array[right_idx]) {
            temp[merged_idx] = array[left_idx];
            left_idx++;
        } else {
            temp[merged_idx] = array[right_idx];
            right_idx++;
        }
        merged_idx++;
    }

    // Copy remaining elements into temp array
    if (left_idx > middle) {
        for (k=right_idx; k<=right; k++) {
            temp[merged_idx] = array[k];
            merged_idx++;
        }
    } else {
        for (k=left_idx; k<=middle; k++) {
            temp[merged_idx] = array[k];
            merged_idx++;
        }
    }

    // Put sorted temp back into array
    for (k=left; k<=right; k++) {
        array[k] = temp[k];
    }
}

__device__ void deviceMergeSort(int *array, int *temp, int left, int right) {
    if (left < right) {
        int middle = (left+right)/2;
        deviceMergeSort(array, temp, left, middle);
        deviceMergeSort(array, temp, middle+1, right);
        deviceMerge(array, temp, left, right, middle);
    }
}

__global__ void mergeSortKernel(int *data, int num_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blockDim.x * gridDim.x;

    int size = num_vals / threads;
    int left = tid * size;
    int right = (tid + 1) * size - 1;

    int *temp = (int*) malloc(num_vals * sizeof(int));
    deviceMergeSort(data, temp, left, right);
    free(temp);
}

void merge(int *array, int *temp, int left, int right, int middle) {
    int left_idx = left;
    int merged_idx = left;
    int right_idx = middle+1;
    int k;

    // Sort left and right side of array into temp
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    while ((left_idx <= middle) && (right_idx <= right)) {
        if (array[left_idx] <= array[right_idx]) {
            temp[merged_idx] = array[left_idx];
            left_idx++;
        } else {
            temp[merged_idx] = array[right_idx];
            right_idx++;
        }
        merged_idx++;
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Copy remaining elements into temp array
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    if (left_idx > middle) {
        for (k=right_idx; k<=right; k++) {
            temp[merged_idx] = array[k];
            merged_idx++;
        }
    } else {
        for (k=left_idx; k<=middle; k++) {
            temp[merged_idx] = array[k];
            merged_idx++;
        }
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Put sorted temp back into array
    for (k=left; k<=right; k++) {
        array[k] = temp[k];
    }
}

void finalMerge(int *array, int *temp, int left, int right, int num_sub_arrays) {
    int middle = (left+right)/2;
    if (num_sub_arrays != 2) {
        //call final merge again
        finalMerge(array, temp, left, middle, num_sub_arrays/2);
        finalMerge(array, temp, middle+1, right, num_sub_arrays/2);
        merge(array, temp, left, right, middle);
    }
    else {
        merge(array, temp, left, right, middle);
    }
}

int main(int argc, char **argv)
{
    NUM_VALS = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "MergeSort";
    std::string programmingModel = "CUDA";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    std::string inputType = "Random";
    int group_number = 13;
    std::string implementation_source = "AI";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", NUM_VALS);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", THREADS);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("main");

    int *d_data;
    size_t d_size = NUM_VALS * sizeof(int);

    // Generate Data
    CALI_MARK_BEGIN("data_init");
    int *h_data = (int*) malloc(NUM_VALS * sizeof(int));
    generate_data(NUM_VALS, h_data);
    CALI_MARK_END("data_init");

    // Cuda overhead
    cudaMalloc((void **)&d_data, d_size);
    cudaMemcpy(d_data, h_data, d_size, cudaMemcpyHostToDevice);

    // Merge sort
    mergeSortKernel<<<BLOCKS, 1>>>(d_data, NUM_VALS);

    // Get data from device
    cudaMemcpy(h_data, d_data, d_size, cudaMemcpyDeviceToHost);

    // Final merge
    int *temp = (int*) malloc(NUM_VALS * sizeof(int));
    finalMerge(h_data, temp, 0, NUM_VALS-1, THREADS);

    // Correctness check
    CALI_MARK_BEGIN("correctness_check");
    bool correct = is_correct(NUM_VALS, h_data);
    CALI_MARK_END("correctness_check");
    std::cout << "is_correct: " << correct << "\n";
    /*
    printf("Sorted array:\n");
    for (int i = 0; i < NUM_VALS; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    */

    // Clean memory
    cudaFree(d_data);
    free(h_data);

    CALI_MARK_END("main");
    return 0;
}