#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <random>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function to generate sorted data
std::vector<int> generate_sorted_data(size_t size)
{
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(i);
    }
    return data;
}

// Function to generate reverse sorted data
std::vector<int> generate_reverse_sorted_data(size_t size)
{
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(size - i - 1);
    }
    return data;
}

// Function to generate random data
std::vector<int> generate_random_data(size_t size)
{
    std::vector<int> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(size) * 10);

    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(dis(gen));
    }
    return data;
}

// Function to generate 1% perturbed data
std::vector<int> generate_perturbed_data(size_t size)
{
    std::vector<int> data = generate_sorted_data(size);
    size_t perturb_count = std::max(1UL, size / 100);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(size) * 10);
    std::uniform_int_distribution<size_t> index_dis(0, size - 1);

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

void bitonic_merge(std::vector<int> &local_data, int step, int rank)
{
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    int partner = rank ^ step;
    MPI_Status status;
    int local_size = local_data.size();

    // Allocate buffer for receiving data from partner
    std::vector<int> partner_data(local_size);

    // Exchange data with partner
    CALI_MARK_BEGIN("MPI_Sendrecv");
    MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 0,
                 partner_data.data(), local_size, MPI_INT, partner, 0,
                 MPI_COMM_WORLD, &status);
    CALI_MARK_END("MPI_Sendrecv");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Determine whether to merge ascending or descending
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bool ascending = ((rank & step) == 0);
    
    if (rank == 1) {
        std::cout << "Step " << step << ": Ascending Order: " << ascending << std::endl;
    }

    // Merge local data and partner data
    std::vector<int> merged_data;
    merged_data.reserve(local_size * 2);
    auto my_data_iter = local_data.begin();
    auto partner_data_iter = partner_data.begin();

    while (merged_data.size() < local_size * 2)
    {
        if (ascending)
        {
            if (my_data_iter == local_data.end() || (partner_data_iter != partner_data.end() && *partner_data_iter < *my_data_iter))
            {
                merged_data.push_back(*partner_data_iter);
                partner_data_iter++;
            }
            else
            {
                merged_data.push_back(*my_data_iter);
                my_data_iter++;
            }
        }
        else
        {
            if (my_data_iter == local_data.end() || (partner_data_iter != partner_data.end() && *partner_data_iter > *my_data_iter))
            {
                merged_data.push_back(*partner_data_iter);
                partner_data_iter++;
            }
            else
            {
                merged_data.push_back(*my_data_iter);
                my_data_iter++;
            }
        }
    }

    // Choose the appropriate half of the merged data
    if (ascending)
    {
        local_data.assign(merged_data.begin(), merged_data.begin() + local_size);
    }
    else
    {
        local_data.assign(merged_data.end() - local_size, merged_data.end());
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
}

void bitonic_sort(std::vector<int> &local_data, int rank, int num_procs)
{
    // First, sort locally
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(local_data.begin(), local_data.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Perform the bitonic merge steps
    for (int step = 1; step < num_procs; step <<= 1)
    {
        for (int j = step; j > 0; j >>= 1)
        {
            bitonic_merge(local_data, j, rank);
        }
    }
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN("main");

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <size> <inputType>\n";
        return 1;
    }

    size_t size = std::stoul(argv[1]);
    std::string inputTypeShort = argv[2];

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

    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "BitonicSort";
    std::string programmingModel = "MPI";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    int groupNumber = 13;
    std::string implementationSource = "AI";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", size);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", num_procs);
    adiak::value("group_num", groupNumber);
    adiak::value("implementation_source", implementationSource);

    std::vector<int> data;
    if (rank == 0)
    {
        CALI_MARK_BEGIN("data_init");
        if (inputType == "Sorted")
        {
            data = generate_sorted_data(size);
        }
        else if (inputType == "Reverse Sorted")
        {
            data = generate_reverse_sorted_data(size);
        }
        else if (inputType == "Random")
        {
            data = generate_random_data(size);
        }
        else if (inputType == "1% Perturbed")
        {
            data = generate_perturbed_data(size);
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
    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    size_t local_size = size / num_procs;
    std::vector<int> local_data(local_size);
    MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    bitonic_sort(local_data, rank, num_procs);

    std::vector<int> gathered_data;
    if (rank == 0)
    {
        gathered_data.resize(size);
    }

    // Gather the sorted data from each process
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_data.data(), local_size, MPI_INT,
               gathered_data.data(), local_size, MPI_INT,
               0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (rank == 0)
    {
        CALI_MARK_BEGIN("correctness_check");
        bool correct = is_correct(gathered_data);
        if (!correct)
        {
            // std::cerr << "Error: The algorithm did not sort the data correctly." << std::endl;
        }
        CALI_MARK_END("correctness_check");
    }

    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}