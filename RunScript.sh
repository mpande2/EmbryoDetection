#!/bin/bash

for ratio in $(seq 2 2 8); do # This is to change ratio of non-viable in the images. 
for viable in $(seq 50 10 150); do
    nonviable=$(( viable * ratio / 10 ))  # Calculate 30% as an integer
    viable=$(( viable - nonviable ))
		python3.11 GenerateSyntheticImages_MultipleTypes.py $viable $nonviable
done
done

