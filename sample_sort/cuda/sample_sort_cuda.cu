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

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>

// Global variables
int THREADS;
int BLOCKS;
size_t NUM_VALS;

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

__global__ void selectSamplesKernel(int *d_data, int *d_samples, size_t group_size, size_t samples_per_group, int blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < blocks)
    {
        for (size_t j = 0; j < samples_per_group; ++j)
        {
            size_t sample_idx = idx * group_size + j * (group_size / samples_per_group);
            d_samples[idx * samples_per_group + j] = d_data[sample_idx];
        }
    }
}

__global__ void assignToBucketsKernel(int *d_data, int *d_buckets, int *d_splitters, int num_splitters, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int data_val = d_data[idx];
        int bucket_id = 0;

        for (int i = 0; i < num_splitters; ++i)
        {
            if (data_val >= d_splitters[i])
            {
                bucket_id = i + 1;
            }
            else
            {
                break;
            }
        }

        d_buckets[idx] = bucket_id;
    }
}

// Sample sort function
void sample_sort(int *h_data, size_t size)
{
    // Copy data to device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    thrust::device_vector<int> d_data(h_data, h_data + size);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // 1. Split data into groups and sort each group
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    size_t group_size = size / BLOCKS;
    for (int i = 0; i < BLOCKS; ++i)
    {
        auto start_itr = d_data.begin() + i * group_size;
        auto end_itr = (i != BLOCKS - 1) ? start_itr + group_size : d_data.end();
        thrust::sort(thrust::device, start_itr, end_itr);
    }
    CALI_MARK_END("comp_large");

    // 2. Select samples and sort them
    CALI_MARK_BEGIN("comp_small");
    size_t samples_per_group = static_cast<int>(size / sqrt(static_cast<double>(size)));
    thrust::device_vector<int> d_samples(BLOCKS * samples_per_group);
    selectSamplesKernel<<<BLOCKS, THREADS>>>(thrust::raw_pointer_cast(&d_data[0]), thrust::raw_pointer_cast(&d_samples[0]), group_size, samples_per_group, BLOCKS);
    cudaDeviceSynchronize();
    thrust::sort(thrust::device, d_samples.begin(), d_samples.end());
    CALI_MARK_END("comp_small");

    // 3. Determine splitters from sorted samples
    int num_buckets = BLOCKS;
    int num_splitters = num_buckets - 1;
    thrust::device_vector<int> d_splitters(num_splitters);
    CALI_MARK_BEGIN("comp_small");
    for (int i = 0; i < num_splitters; ++i)
    {
        size_t splitter_idx = ((i + 1) * d_samples.size()) / num_buckets - 1;
        d_splitters[i] = d_samples[splitter_idx];
    }
    CALI_MARK_END("comp_small");

    // 4. Assign elements to buckets
    CALI_MARK_BEGIN("comp_large");
    thrust::device_vector<int> d_bucket_ids(size);
    assignToBucketsKernel<<<(size + THREADS - 1) / THREADS, THREADS>>>(thrust::raw_pointer_cast(&d_data[0]), thrust::raw_pointer_cast(&d_bucket_ids[0]), thrust::raw_pointer_cast(&d_splitters[0]), num_splitters, size);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");

    std::vector<thrust::device_vector<int>> buckets(num_buckets);

    // Distribute elements into buckets
    CALI_MARK_BEGIN("comp_large");
    for (size_t idx = 0; idx < size; ++idx)
    {
        int bucket_id = d_bucket_ids[idx];
        buckets[bucket_id].push_back(d_data[idx]);
    }
    CALI_MARK_END("comp_large");

    // 5. Sort within each bucket
    CALI_MARK_BEGIN("comp_large");
    for (auto &bucket : buckets)
    {
        thrust::sort(thrust::device, bucket.begin(), bucket.end());
    }

    // Flatten buckets into d_data
    auto d_data_itr = d_data.begin();
    for (auto &bucket : buckets)
    {
        d_data_itr = thrust::copy(bucket.begin(), bucket.end(), d_data_itr);
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Copy sorted data back to host
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    thrust::copy(d_data.begin(), d_data.end(), h_data);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
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

    std::string algorithm = "SampleSort";
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

    sample_sort(data.data(), data.size());

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
