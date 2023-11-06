#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define bitonic_sort_step_region "bitonic_sort_step"
#define cudaMemcpy_host_to_device "cudaMemcpy_host_to_device"
#define cudaMemcpy_device_to_host "cudaMemcpy_device_to_host"
#define array_fill_name "array_fill"

void bitonic_sort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);
void quicksort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);