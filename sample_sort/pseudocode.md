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