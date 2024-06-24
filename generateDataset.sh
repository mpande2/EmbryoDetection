#!/bin/bash
log_filename=generation_log.txt
output_path=generated_images_high_volume
min_eggs=300
max_eggs=500
number_images_per_ratio=50
child_pids=()

for i in $(seq 1 1 4); do
    viable_percent=$(( i*20 ))
    viable_fraction=$(bc -l <<< $viable_percent/100)
    nohup python SyntheticGenerator.py \
    --number_images=$number_images_per_ratio \
    --viable_percent=$viable_fraction \
    --min_eggs=$min_eggs \
    --max_eggs=$max_eggs \
    --save_dir=$output_path >> $log_filename 2>&1 &
    child_pids+=("$!")
done

printf "\n" >> $log_filename
printf "\nDataset generation has started. Output files will be available in:
$output_path\n
Logs from the generation process can be found in:
$log_filename\n\n"

for pid in "${child_pids[@]}"; do
    wait "$pid"
    return_code="$?"
    printf "Child process with PID = $pid finished with return_code = $return_code\n"
done

printf "\nDataset generation done.\n"