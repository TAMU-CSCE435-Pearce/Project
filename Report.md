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
  - MPI: Data partitioning and merging will involve `MPI_Scatterv` and `MPI_Gatherv`.
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

### 2c. Evaluation plan
We will measure and compare the following for each algorithm:
- **Input sizes**: Ranging from 2^10 to 2^24 elements.
- **Input types**: Sorted, random, reverse, and sorted with 1% perturbed.
- **Strong scaling**: We will evaluate how the performance improves as we increase the number of processors/nodes while keeping the problem size constant.
- **Weak scaling**: We will assess how the performance changes as we scale the problem size along with the number of processors.
- **Number of threads in a block on the GPU**: Ranging from 2^4 to 2^10.