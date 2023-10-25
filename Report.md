# CSCE 435 Group project

## 1. Group members:
1. Kyle Diano
2. Connor Bowling
3. Chris Anand
4. Connor McLean

---

## 2. _due 10/25_ Project topic
Choose 3+ parallel sorting algorithms, implement in MPI and CUDA.  Examine and compare performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

Note: Team will communicate via *Slack*

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Parallel Bitonic Sort on a Hypercube (MPI + CUDA)

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

pseudocode from slides

- Parallel Mergesort (MPI + CUDA)

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

pseudocode (slightly modified) from [tutorialspoint](https://www.tutorialspoint.com/parallel_algorithm/parallel_algorithm_sorting.htm)

- Parallel Odd-Even Transposition Sort (MPI + CUDA)

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

pseudocode from slides