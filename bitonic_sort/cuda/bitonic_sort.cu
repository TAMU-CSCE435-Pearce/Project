/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <random>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

// Global variables
int THREADS;
int BLOCKS;
size_t NUM_VALS;

const char *cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char *cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Function to generate sorted data
std::vector<int> generate_sorted_data()
{
    std::vector<int> data(NUM_VALS);
    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(i);
    }
    return data;
}

// Function to generate reverse sorted data
std::vector<int> generate_reverse_sorted_data()
{
    std::vector<int> data(NUM_VALS);
    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(NUM_VALS - i - 1);
    }
    return data;
}

// Function to generate random data
std::vector<int> generate_random_data()
{
    std::vector<int> data(NUM_VALS);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(NUM_VALS) * 10);

    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(dis(gen));
    }
    return data;
}

// Function to generate 1% perturbed data
std::vector<int> generate_perturbed_data()
{
    std::vector<int> data = generate_sorted_data();
    size_t perturb_count = std::max(1UL, NUM_VALS / 100);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(NUM_VALS) * 10);
    std::uniform_int_distribution<size_t> index_dis(0, NUM_VALS - 1);

    for (size_t i = 0; i < perturb_count; ++i)
    {
        data[index_dis(gen)] = static_cast<int>(dis(gen));
    }
    return data;
}

// Check if the data is correctly sorted
bool is_correct(const std::vector<int> &data)
{
    for (size_t i = 1; i < data.size(); i++)
    {
        if (data[i - 1] > data[i])
        {
            return false;
        }
    }
    return true;
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0)
        {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values)
{

    float *dev_values;
    size_t size = NUM_VALS * sizeof(int);

    cudaMalloc((void **)&dev_values, size);

    // MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    dim3 blocks(BLOCKS, 1);   /* Number of blocks   */
    dim3 threads(THREADS, 1); /* Number of threads  */

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    int j, k;
    int calls = 0;

    /* Major step */
    for (k = 2; k <= NUM_VALS; k <<= 1)
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            calls++;
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(dev_values);
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN("main");

    // Check for correct number of arguments
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <size> <numThreads> <inputType>\n";
        return 1;
    }

    // Parse the command line arguments
    NUM_VALS = std::stoul(argv[1]);
    THREADS = std::stoi(argv[2]);
    BLOCKS = (NUM_VALS + THREADS - 1) / THREADS;
    std::string inputTypeShort = argv[3];

    std::string inputType;
    if (inputTypeShort == "r")
    {
        inputType = "Random";
    }
    else if (inputTypeShort == "s")
    {
        inputType = "Sorted";
    }
    else if (inputTypeShort == "rs")
    {
        inputType = "Reverse Sorted";
    }
    else if (inputTypeShort == "p")
    {
        inputType = "1% Perturbed";
    }
    else
    {
        std::cerr << "Invalid input type. Use 'r', 's', 'rs', or 'p'.\n";
        return 1;
    }

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "BitonicSort";
    std::string programmingModel = "CUDA";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    int group_number = 13;
    std::string implementation_source = "AI";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", NUM_VALS);
    adiak::value("InputType", inputType);
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("data_init");
    std::vector<int> data;
    if (inputType == "Sorted")
    {
        data = generate_sorted_data();
    }
    else if (inputType == "Reverse Sorted")
    {
        data = generate_reverse_sorted_data();
    }
    else if (inputType == "Random")
    {
        data = generate_random_data();
    }
    else if (inputType == "1% Perturbed")
    {
        data = generate_perturbed_data();
    }
    else
    {
        std::cerr << "Invalid input type. Use 'Sorted', 'Reverse Sorted', 'Random', or '1% Perturbed'.\n";
        return 1;
    }
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    bitonic_sort(data.data());

    CALI_MARK_BEGIN("correctness_check");
    bool correct = is_correct(data);
    if (!correct)
    {
        std::cerr << "Error: The algorithm did not sort the data correctly." << std::endl;
    }
    CALI_MARK_END("correctness_check");

    CALI_MARK_END("main");
    return 0;
}