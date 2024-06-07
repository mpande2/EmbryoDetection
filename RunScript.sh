#!/bin/bash

for viable in $(seq 100 50 300); do
    nonviable=$(( viable * 3 / 10 ))  # Calculate 30% as an integer
    python3.11 GenerateSyntheticImages_MultipleTypes.py $viable $nonviable
done

