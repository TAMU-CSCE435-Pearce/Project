#include <iostream>
#include <vector>
#include <algorithm>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// Global variables
int THREADS;
int BLOCKS;
int NUM_VALS;

const char *bitonic_sort_step_region = "bitonic_sort_step";
const char *cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char *cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Generate data
std::vector<int> generate_data(size_t size)
{
    std::vector<int> data(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand() % (size * 10);
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

// CUDA kernel to pick samples from the sorted subarrays
__global__ void pick_samples(const int *data, int *samples, int stride, int num_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples)
    {
        samples[idx] = data[idx * stride];
    }
}

// CUDA kernel to partition data based on splitters
__global__ void partition_data(int *data, int *splitters, int *buckets, int n, int num_splitters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int val = data[idx];
        int bucket = 0;
        while (bucket < num_splitters && val >= splitters[bucket])
        {
            bucket++;
        }
        buckets[idx] = bucket;
    }
}

// CUDA kernel to scatter elements into their correct positions
__global__ void scatter_elements(int *data, int *bucket_indices, int *bucket_start_indices, int *scattered_data, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        int bucket_index = bucket_indices[index];
        int pos = atomicAdd(&bucket_start_indices[bucket_index], 1);
        scattered_data[pos] = data[index];
    }
}

// CUDA kernel to count elements per bucket and initialize gather indices
__global__ void count_elements_and_prepare_gather(int *data, int *bucket_indices, int *bucket_counts, int *gather_indices, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        int bucket_index = bucket_indices[index];
        atomicAdd(&bucket_counts[bucket_index], 1);
        // Initialize gather_indices with -1 or a marker value to indicate unprocessed elements
        gather_indices[index] = -1;
    }
}

// Sample sort host function
void sample_sort(int *h_data, size_t size)
{
    // Allocate memory and copy data to the device
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    thrust::device_vector<int> d_data(h_data, h_data + size);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Perform local sort on the device using Thrust
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    thrust::sort(d_data.begin(), d_data.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Determine the number of samples to pick
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    int num_samples = sqrt(size);
    int stride = size / num_samples;
    thrust::device_vector<int> d_samples(num_samples);

    // Define grid and block sizes for CUDA kernels
    int threadsPerBlock = 256;
    int blocks = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    // Use a kernel to pick splitters from the sorted data
    pick_samples<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()),
                                              thrust::raw_pointer_cast(d_samples.data()), stride, num_samples);
    cudaDeviceSynchronize();

    // Sort the samples on the device to get splitters
    thrust::sort(d_samples.begin(), d_samples.end());
    thrust::device_vector<int> d_splitters(num_samples - 1);
    thrust::copy(d_samples.begin() + 1, d_samples.end(), d_splitters.begin());

    // Allocate memory for buckets
    thrust::device_vector<int> d_buckets(size);

    // Partition the data into buckets according to the splitters
    blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    partition_data<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()),
                                                thrust::raw_pointer_cast(d_splitters.data()),
                                                thrust::raw_pointer_cast(d_buckets.data()), size, num_samples - 1);
    cudaDeviceSynchronize();

    // Count elements per bucket and prepare for gathering
    thrust::device_vector<int> d_bucket_counts(num_samples, 0);
    thrust::device_vector<int> d_gather_indices(size);
    blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    count_elements_and_prepare_gather<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()),
                                                                   thrust::raw_pointer_cast(d_buckets.data()),
                                                                   thrust::raw_pointer_cast(d_bucket_counts.data()),
                                                                   thrust::raw_pointer_cast(d_gather_indices.data()),
                                                                   size);
    cudaDeviceSynchronize();

    // Compute the starting indices of each bucket
    thrust::device_vector<int> d_bucket_starts(num_samples, 0);
    thrust::exclusive_scan(d_bucket_counts.begin(), d_bucket_counts.end(), d_bucket_starts.begin());
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Scatter the elements into their correct positions
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    thrust::device_vector<int> d_scattered_data(size);
    scatter_elements<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_data.data()),
                                                  thrust::raw_pointer_cast(d_buckets.data()),
                                                  thrust::raw_pointer_cast(d_bucket_starts.data()),
                                                  thrust::raw_pointer_cast(d_scattered_data.data()),
                                                  size);
    cudaDeviceSynchronize();

    // Now each bucket in d_scattered_data can be sorted individually
    for (int i = 0; i < num_samples; ++i)
    {
        int start_index = d_bucket_starts[i];
        int bucket_size = (i == num_samples - 1) ? size - start_index : d_bucket_starts[i + 1] - start_index;

        if (bucket_size > 0)
        {
            thrust::sort(d_scattered_data.begin() + start_index, d_scattered_data.begin() + start_index + bucket_size);
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Copy the sorted data back to host
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    thrust::copy(d_scattered_data.begin(), d_scattered_data.end(), h_data);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
}

int main(int argc, char **argv)
{
    NUM_VALS = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "SampleSort";
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
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("main");

    CALI_MARK_BEGIN("data_init");
    std::vector<int> data = generate_data(NUM_VALS);
    CALI_MARK_END("data_init");

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
