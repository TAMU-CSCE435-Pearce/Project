#include <iostream>
#include <vector>
#include <algorithm>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Generate data
void generate_data(size_t size, int *data)
{
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand() % (size * 10);
    }
}

void mergeSort(int *array, int *temp, int left, int right) {
    if (left < right) {
        int middle = (left+right)/2;
        mergeSort(array, temp, left, middle);
        mergeSort(array, temp, (middle+1), right);
        merge(array, temp, left, right, middle);
    }
}

void merge(int *array, int *temp, int left, int right, int middle) {
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

int main(int argc, char **argv)
{
    size_t inputSize;
    if (argc > 1)
    {
        inputSize = std::stoi(argv[1]);
    }

    int num_procs, rank;

    // Initialize
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "SampleSort";
    std::string programmingModel = "MPI";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    std::string inputType = "Random";
    int group_number = 13;
    std::string implementation_source = "AI";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", inputSize);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", num_procs);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("main");

    // Generate Data
    CALI_MARK_BEGIN("generate_data");
    int *data = new int[inputSize];-
    generate_data(inputSize, data);
    CALI_MARK_END("generate_data");

    // Divide data into equal chunks then send to each process
    int sub_array_size = inputSize/num_procs;
    int *sub_array = new int[sub_array_size];
    MPI_Scatter(data, sub_array_size, MPI_INT, sub_array, sub_array_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Merge sort
    int *temp_array = new int[sub_array_size];
    mergeSort(sub_array, temp_array, 0, (sub_array_size-1))

    // Clean memory
    delete [] data;
    delete [] sub_array;
    delete [] temp_array;

    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}