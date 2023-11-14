/*
 * CUDA Implementation of Odd-Even Transposition Sort
 * Adapted for CUDA from the MPI implementation
 * Original MPI Source: https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/oddEvenSort/oddEven.html
 * Peter S. Pacheco, An Introduction to Parallel Programming,
 * Morgan Kaufmann Publishers, 2011
 * IPP: Section 3.7 (pp. 131)
 *
 * CUDA Adaptation Source: ChatGPT
 *
 *  cuda_odd_even_sort.cu
 *
 * Usage: ./odd_even_sort <number of threads> <number of values>
 *
 *         - number of threads: Number of threads per block in CUDA
 *         - number of values: Total number of values to sort
 *
 * This CUDA implementation is adapted to work on GPU devices,
 * performing the odd-even transposition sort algorithm in parallel.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/* Local functions */
void printArrayPortion(const int* arr, int size, const char* prefix);
void generateData(int* arr, int size);
bool isSortedAndPrint(const int* arr, int size);

/* Functions involving communication */
__global__ void oddEvenSortStepKernel(int* d_A, int n, bool isOddPhase);
void oddEvenSort(int* h_A, int n, int threads, int blocks);

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number of threads> <number of values>\n", argv[0]);
        return 1;
    }

    CALI_MARK_BEGIN("main");

    int threads = atoi(argv[1]);
    int num_vals = atoi(argv[2]);
    int blocks = (num_vals + threads - 1) / threads;

    int* h_A = (int*) malloc(num_vals * sizeof(int));

    // Generate data
    CALI_MARK_BEGIN("data_init");
    generateData(h_A, num_vals);
    CALI_MARK_END("data_init");


    // Print a portion of the array before sorting
    #ifdef OUTPUT
    printArrayPortion(h_A, num_vals, "Before Sorting");
    #endif

    // Perform the sorting
    oddEvenSort(h_A, num_vals, threads, blocks);

    // Print a portion of the array after sorting
    #ifdef OUTPUT
    printArrayPortion(h_A, num_vals, "After Sorting");
    #endif

    // Check if the array is sorted and print the result
    CALI_MARK_BEGIN("correctness_check");
    isSortedAndPrint(h_A, num_vals);
    CALI_MARK_END("correctness_check");


    free(h_A);

    CALI_MARK_END("main");

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "OddEvenSort";
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
    adiak::value("InputSize", num_vals);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", threads);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}

/*-------------------------------------------------------------------
 * Kernel Function: oddEvenSortStepKernel
 * Purpose: Perform one step of the odd-even sort in parallel on the GPU.
 * Input args: 
 *   - d_A: array to sort
 *   - n: size of the array
 *   - isOddPhase: flag to indicate if the current phase is odd or even
 */
__global__ void oddEvenSortStepKernel(int* d_A, int n, bool isOddPhase) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idx2 = idx + 1;

    if (idx2 < n && ((isOddPhase && (idx % 2 == 0)) || (!isOddPhase && (idx % 2 != 0)))) {
        if (d_A[idx] > d_A[idx2]) {
            int temp = d_A[idx];
            d_A[idx] = d_A[idx2];
            d_A[idx2] = temp;
        }
    }
}

/*-------------------------------------------------------------------
 * Function: oddEvenSort
 * Purpose: Perform the odd-even sort on the GPU.
 * Input args: 
 *   - h_A: host array to sort
 *   - n: size of the array
 *   - threads: number of threads per block
 *   - blocks: number of blocks
 */
void oddEvenSort(int* h_A, int n, int threads, int blocks) {
    int* d_A;
    cudaMalloc(&d_A, n * sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < n; ++i) {
        oddEvenSortStepKernel<<<blocks, threads>>>(d_A, n, i % 2 == 0);
        cudaDeviceSynchronize();
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(h_A, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(d_A);
}


/*-------------------------------------------------------------------
 * Function: printArrayPortion
 * Purpose: Print a portion of an array.
 * Input args: 
 *   - arr: array to print
 *   - size: size of the array
 *   - prefix: prefix string for the printout
 */
void printArrayPortion(const int* arr, int size, const char* prefix) {
    printf("%s: ", prefix);
    int numElementsToShow = 10; // Number of elements to show at the start and end
    for (int i = 0; i < numElementsToShow; ++i) {
        printf("%d ", arr[i]);
    }
    printf("... ");
    for (int i = size - numElementsToShow; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/*-------------------------------------------------------------------
 * Function: generateData
 * Purpose: Generate random data for an array.
 * Input args: 
 *   - arr: array to fill with data
 *   - size: size of the array
 */
void generateData(int* arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; // Generate random values
    }
}

/*-------------------------------------------------------------------
 * Function: isSortedAndPrint
 * Purpose: Check if an array is sorted and print the result.
 * Input args: 
 *   - arr: array to check
 *   - size: size of the array
 * Output: Returns true if the array is sorted, false otherwise.
 */
bool isSortedAndPrint(const int* arr, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            printf("Error: Array is not sorted.\n");
            return false;  // Array is not sorted
        }
    }
    printf("Array is sorted.\n");
    return true;  // Array is sorted
}