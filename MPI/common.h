#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>
#include <utility>
#include <bits/stdc++.h> 

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

#define PRINT_DEBUG 0

#define ARRAY_FILL_NAME "array_fill"
#define SORT_CHECK_NAME "sort_check"
#define SAMPLE_SORT_NAME "sample_sort"

void sample_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank, int sample_size);