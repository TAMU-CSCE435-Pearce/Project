# CSCE 435 Group project

## 0. Group number: 13

## 1. Group members:
1. Kyle Diano
2. Connor Bowling
3. Chris Anand
4. Connor McLean

## 2. Project topic: Parallel Sorting Algorithms Comparison

### 2a. Brief project description

We will compare the performance of four parallel sorting algorithms implemented using MPI and CUDA. The algorithms to be compared are:

- Parallel Bitonic Sort
- Parallel Mergesort
- Parallel Odd-Even Transposition Sort
- Parallel Sample Sort

Each of these algorithms will be assessed on multi-core CPU architectures using MPI and on NVIDIA GPUs using CUDA.

### 2b. Pseudocode for each parallel algorithm

- For **Parallel Bitonic Sort**:
  - MPI: We will use `MPI_Send` and `MPI_Recv` for comparison exchanges between processes.
  - CUDA: The compare-exchange operations will be performed in a CUDA kernel, with data transfer to/from the GPU occurring before and after the sort.

```
procedure BITONIC_SORT(label, d)
begin
    for i := 0 to d - 1 do
        for j := i downto 0 do
            if (i + 1)st bit of label = j th bit of label then
                comp exchange max(i);
            else
                comp exchange min(j);
        end for
    end for
end BITONIC_SORT
```

- For **Parallel Mergesort**:
  - MPI: Data partitioning and merging will involve `MPI_Scatter` and `MPI_Gather`.
  - CUDA: Sorting within each partition will be handled by a CUDA kernel.

```
procedure PARALLEL MERGE SORT(id, n, data, newdata)
begin
    data := sequentialmergesort(data)
    for dim := 1 to n do
        begin
            data := parallelmerge(id, dim, data)
        end
    newdata := data
end PARALLEL MERGE SORT
```

- For **Parallel Odd-Even Transposition Sort**:
  - MPI: The compare-exchange operations will involve `MPI_Sendrecv` for pairwise comparison exchanges.
  - CUDA: Sorting steps will be carried out within CUDA kernels, with the necessary data shuttling to/from the GPU.

```
procedure ODD-EVEN PAR(n)
begin
    id := process's label
    for i := 1 to n do
        begin
            if i is odd then
                if id is odd then
                    compare-exchange min(id + 1);
                else
                    compare-exchange max(id - 1);
            if i is even then
                if id is even then
                    compare-exchange min(id + 1);
                else
                    compare-exchange max(id - 1);
            end if
        end for
    end for
end ODD-EVEN PAR
```

- For **Parallel Sample Sort**:
  - MPI: We will use `MPI_Gather` to collect samples, and MPI_Bcast for broadcasting splitters.
  - CUDA: Local sorting and data assignment to buckets will be done with CUDA kernels.

```
procedure PARALLEL_SAMPLE_SORT(id, p, data)
begin
    localData := sort(data)
    samples := select_samples(localData, p-1)
    allSamples := gather_samples(samples)
    
    if id = 0 then
        sortedSamples := sort(allSamples)
        splitters := select_splitters(sortedSamples, p-1)
    end if
    
    splitters := broadcast(splitters)
    bucketedData := assign_to_buckets(localData, splitters)
    
    exchangedData := exchange_data(bucketedData, id, p)
    sortedExchangedData := sort(exchangedData)
    
    return sortedExchangedData
end procedure
```

## Performance Analysis of Parallel Sample Sort

### Strong Scaling Analysis for SampleSort with MPI
![](./sample_sort/Sample%20Sort%20Strong%20Scaling%20MPI.png)
#### Graph Overview
- The strong scaling graph for MPI displays the average execution time as the number of processors increases, with a fixed input size (626144).

#### Trends
- Ideally, the execution time should decrease as the number of processors increases.
- The execution time for sorted, 1% perturbed, and random inputs decreases with more processors, indicating good scaling.
- The reverse sorted input does not show a significant decrease in execution time, suggesting challenges in parallelization.

#### Interpretation
- The algorithm scales well with an increasing number of processors up to a point, after which the scaling benefits reduce, likely due to communication overhead or synchronization issues.

### Strong Scaling Analysis for SampleSort with CUDA
![](./sample_sort/Sample%20Sort%20Strong%20Scaling%20CUDA.png)
#### Graph Overview
- This graph illustrates how the execution time changes with an increasing number of threads in a block, keeping the input size constant.

#### Trends
- A sharp decrease in execution time is observed as threads increase up to around 600.
- Beyond this point, the execution time stabilizes or slightly increases.

#### Interpretation
- Initially, increased thread count leads to better GPU resource utilization.
- After reaching a threshold, managing more threads adds overhead, and the execution time does not improve, indicating an optimal thread count range for efficiency.

### Weak Scaling Analysis for SampleSort with MPI
![](./sample_sort/Sample%20Sort%20Weak%20Scaling%20MPI.png)
#### Graph Overview
- Weak scaling is examined by proportionally increasing the problem size with the number of processors.

#### Trends
- Ideally, execution time should remain constant with increases in problem size and processor count.
- The execution time increases with more processors and larger problem sizes, which deviates from the ideal.

#### Interpretation
- The increase in execution time may indicate significant communication overhead or imbalanced work distribution among processors.

### Weak Scaling Analysis for SampleSort with CUDA
![](./sample_sort/Sample%20Sort%20Weak%20Scaling%20CUDA.png)
#### Graph Overview
- Similar to MPI, this graph shows execution time against increasing problem size and number of threads in a block.

#### Trends
- The execution time remains relatively constant as problem size and thread count increase, demonstrating good weak scaling.

#### Interpretation
- Consistent execution times across different thread counts suggest that the GPU implementation is efficiently parallelized, handling larger problem sizes effectively without a significant rise in execution time.

## Performance Analysis of Parallel Bitonic Sort

### Strong Scaling Analysis for Bitonic Sort with CUDA
![](./bitonic_sort/Bitonic%20Sort%20Strong%20Scaling%20CUDA.png)
#### Graph Overview
- The graph represents the average execution time for sorting a random input of size 65536 as the number of threads increases.

#### Trends
- Initial decrease in execution time as the number of threads increases from 75 to about 150.
- Beyond 150 threads, the execution time starts to increase.

#### Interpretation
- The decrease suggests that additional threads initially lead to better parallelization.
- The subsequent increase may be due to overhead or hardware limitations such as memory bandwidth or thread synchronization issues.

### Strong Scaling Analysis for Bitonic Sort with MPI
![](./bitonic_sort/Bitonic%20Sort%20Strong%20Scaling%20MPI.png)
#### Graph Overview
- Displays the average execution time for sorting a random input of size 65536 with an increasing number of processors.

#### Trends
- Significant reduction in execution time with an increase in processors up to 200.
- Slow increase in execution time with more processors beyond this point.

#### Interpretation
- Sharp decrease followed by a gradual increase suggests that there is an optimal number of processors for parallelization, beyond which overhead costs reduce efficiency.

### Weak Scaling Analysis for Bitonic Sort with CUDA
![](./bitonic_sort/Bitonic%20Sort%20Weak%20Scaling%20CUDA.png)
#### Graph Overview
- Shows execution time while increasing the number of threads in a block and the problem size proportionally.

#### Trends
- Execution times remain flat across the range of threads for different problem sizes.

#### Interpretation
- The stable execution times regardless of the increased problem size indicate good weak scaling, with the algorithm effectively handling larger datasets with more threads.

### Weak Scaling Analysis for Bitonic Sort with MPI
![](./bitonic_sort/Bitonic%20Sort%20Weak%20Scaling%20MPI.png)
#### Graph Overview
- Indicates execution time as both the problem size (1 million elements per processor) and the number of processors increase.

#### Trends
- Gradual increase in execution time as more processors are added.

#### Interpretation
- The algorithm's parallel efficiency seems to decrease with scaling, likely due to communication costs or load imbalances at larger scales.

Use this analysis to explore the Bitonic Sort algorithm's performance in both MPI and CUDA implementations, discussing potential reasons for the observed trends and considering improvements for efficiency.
