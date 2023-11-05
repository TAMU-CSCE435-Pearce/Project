# CSCE 435 Group 16 project

## 0. Group number: 16

## 1. Group members:

1. Anjali Kumar
2. Shawn Mathen
3. Ashrita Vadlapatla
4. Robert Eads

## 2. Project topic (e.g., parallel sorting algorithms)
Parallel Sorting Algorithms

## 2a. Communication Method
Our team's primary method of communication will be GroupMe with Slack as a secondary method.

## 2b. Brief project description (what algorithms will you be comparing and on what architectures)
Each of the selected sort algorithms, Bubble, Merge, Quick, & Sample will be run in parallel using MPI and CUDA separately. 

## 2c. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

### Algorithm 1: Bubble Sort

```
begin BubbleSort(list)

   for all elements of list
      if list[i] > list[i+1]
         swap(list[i], list[i+1])
      end if
   end for

   return list

end BubbleSort
```


### Algorithm 2: Merge Sort
Sequential:
```python
function merge(left, right):
  result = []
  i, j = 0, 0
  while i < length(left) and j < length(right):
    if left[i] < right[j]:
      result.append(left[i])
      i += 1
    else:
      result.append(right[j])
      j += 1

  result += left[i:]
  result += right[j:]
  return result

function mergeSort(x):
  if length(x) <= 1:
    return x

  
   middle = len(x)//2
   left = mergeSort(x[:middle])
   right = mergeSort(x[middle:])
   return merge(left, right)
```

Parallel:
The following parallel Merge Sort algorithms will be implemented using MPI.
```python
function parallel_merge_sort(arr, num_threads)
    if length(arr) <= 1
        return arr

    mid = length(arr) // 2
    left = arr[0...mid-1]
    right = arr[mid...]

    if num_threads > 1
        left_thread = start_thread(parallel_merge_sort(left, num_threads/2))
        right_thread = start_thread(parallel_merge_sort(right, num_threads/2))
        join_thread(left_thread)
        join_thread(right_thread)

    else
        left = merge_sort(left)
        right = merge_sort(right)

    return parallel_merge(left, right)
```
### Algorithm 3: Quick Sort

```
int partition(int arr[], int low, int high) {

   int pivot = arr[high];
   int i = (low-1);

   for (int j = low; j <= high; j++) {
    if (arr[j] < pivot) {
      i++;
      swap(arr[i], arr[j]);
    }
   }
   swap(arr[i+1], arr[high]);
   return (i+1);
}

void quickSort(int arr[], int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);

    quickSort(arr, low, pi-1);
    quickSort(arr, pi+1, high);
  }
}
```
### Algorithm 4: Sample Sort
  #### MPI
  ```
  Sample 4 splitters
  Send splitters to all processes using MPI_Allgather
  Sort the splitters and decide upon bucket cutoffs
  Evaluate local elements and place into send buffers
  Calculate and communicate buffer sizes using MPI_Gather
  Allocate receive buffer for incoming data
  Send data in buffers to the appropriate process with MPI_Gatherv
  Sort with sequential quicksort
  ```

  #### CUDA
  ```
  ```

## 2d. Citations
- https://www.tutorialspoint.com/data_structures_algorithms/bubble_sort_algorithm.htm
- https://compucademy.net/algorithmic-thinking-with-python-part-3-divide-and-conquer-strategy/#:~:text=There%20is%20a%20really%20clever%20trick%20that,the%20same%20type%20as%20the%20original%20problem.
- https://teivah.medium.com/parallel-merge-sort-in-java-e3213ae9fa2c
- https://www.geeksforgeeks.org/quick-sort/
- https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Nicolas-Barrios-Fall-2021.pdf
- https://en.wikipedia.org/wiki/Samplesort
- https://www.geeksforgeeks.org/cpp-program-for-quicksort/#



## 2e. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU 

Each algorithm with be run with the input types of sorted, reverse sorted, and randomized. The input sizes are planned to be 2^16, 2^20, and 2^24.
Both weak and strong scaling will be analyzed and compared against the other algorithms being tested.
Block size will be... 


## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
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
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
S