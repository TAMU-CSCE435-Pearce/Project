#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;


void printArr(int arr[], int n) {
    for (int i = 0; i < n; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}

__device__ int d_size;

__global__ void partition(int *arr, int *arr_l, int *arr_h, int n) {
    
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z < n) {
        int h = arr_h[z];
        int l = arr_l[z];
        int x = arr[h];
        int i = (l - 1);
        int temp;
        for (int j = l; j <= h - 1; j++) {
            if (arr[j] <= x) {
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        temp = arr[i + 1];
        arr[i + 1] = arr[h];
        arr[h] = temp;
        int p = (i + 1);
        if (p - 1 > l) {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p - 1;
        }
        if (p + 1 < h) {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p + 1;
            arr_h[ind] = h;
        }
    }
    
}

// void quickSortIterative(int arr[], int l, int h, int threads_per_block, int size) {
    
    
//     int lstack[h - l + 1], hstack[h - l + 1];

//     int top = -1, *d_d, *d_l, *d_h;

//     lstack[++top] = l;
//     hstack[top] = h;

//     CALI_MARK_BEGIN("comm");
//     CALI_MARK_BEGIN("comm_large");

//     cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
//     cudaMemcpy(d_d, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

//     cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
//     cudaMemcpy(d_l, lstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

//     cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
//     cudaMemcpy(d_h, hstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

//     CALI_MARK_END("comm_large");
//     CALI_MARK_END("comm");

//     int n_t = threads_per_block;
//     int n_b = size / threads_per_block;
//     int n_i = 1;
    
//     CALI_MARK_BEGIN("comp");
//     CALI_MARK_BEGIN("comp_large");
//     while (n_i > 0) {
//         partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
//         int answer;
//         CALI_MARK_BEGIN("comm");
//         CALI_MARK_BEGIN("comm_small");
//         cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
//         n_t = threads_per_block;
//         n_i = answer;
//         cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//         CALI_MARK_END("comm_small");
//         CALI_MARK_END("comm");
//     }
//     CALI_MARK_END("comp_large");
//     CALI_MARK_END("comp");
    
// }

void data_init(int arr[], int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }
}

bool isSorted(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

void correctness_check(int arr[], int n) {
    if (isSorted(arr, n)) {
        printf("Array is correctly sorted.\n");
    } else {
        printf("Array is NOT correctly sorted.\n");
    }
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");
    
    if (argc != 3) {
        printf("Usage: %s <threads_per_block> <input_size>\n", argv[0]);
        return 1;
    }

    int threads_per_block = atoi(argv[1]);
    int input_size = atoi(argv[2]);

    cali::ConfigManager mgr;
    mgr.start();

    int arr[input_size];

    CALI_MARK_BEGIN("data_init");
    data_init(arr, input_size);
    CALI_MARK_END("data_init");

    int n = sizeof(arr) / sizeof(*arr);
    printf("Number of threads per block: %d\n", threads_per_block);
    printf("Input size: %d\n", input_size);

    // printf("Array before sorting: ");
    // printArr(arr, n);

    // quickSortIterative(arr, 0, n - 1, threads_per_block, input_size);

    int l = 0;
    int h = n-1;
    int size = input_size;

    int lstack[h - l + 1], hstack[h - l + 1];

    int top = -1, *d_d, *d_l, *d_h;

    lstack[++top] = l;
    hstack[top] = h;

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");

    cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_d, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_l, lstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_h, hstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    int n_t = threads_per_block;
    int n_b = size / threads_per_block;
    int n_i = 1;
    
    
    while (n_i > 0) {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        CALI_MARK_BEGIN("kernal_partition");
        partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
        CALI_MARK_END("kernal_partition");
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");
        int answer;
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_small");
        CALI_MARK_BEGIN("cudaMemcpySymbol");
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
        CALI_MARK_END("cudaMemcpySymbol");
        CALI_MARK_END("comm_small");
        
        n_t = threads_per_block;
        n_i = answer;
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("cudaMemcpySortedinLoop");
        cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        CALI_MARK_END("cudaMemcpySortedinLoop");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    }
    

    string algorithm = "QuickSort";
    string programmingModel = "CUDA";
    string datatype = "int";
    int sizeOfDatatype = sizeof(int);
    int inputSize = input_size;
    string inputType = "Random";
    int num_threads = threads_per_block;
    int num_blocks = input_size / threads_per_block;
    string group_number = "18";
    string implementation_source = "Online";

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // printf("Array after sorting: ");
    // printArr(arr, n);
    CALI_MARK_BEGIN("correctness_check");
    correctness_check(arr, n);
    CALI_MARK_END("correctness_check");
    CALI_MARK_END("main");
    
    mgr.stop();
    mgr.flush();

    return 0;
}

