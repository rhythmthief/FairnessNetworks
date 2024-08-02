#!/bin/bash

# Function to run a single job
run_job() {
    local job_id=$1
    echo "Starting job $job_id"
    # Replace the following line with the command to run your Python script
    python main.py timing_algos low $job_id
    echo "Job $job_id completed"
}

# Total number of jobs
total_jobs=173

# Number of concurrent jobs
max_concurrent_jobs=7

# Current job index
job_index=0

# Array to keep track of background process IDs
pids=()

# Function to start a job and add its PID to the list
start_job() {
    run_job "$job_index" &
    pids+=($!)
    ((job_index++))
}

# Start initial batch of jobs
for ((i=0; i<max_concurrent_jobs; i++)); do
    start_job
done

# Monitor the jobs and start new ones as old ones finish
while ((job_index <= total_jobs)); do
    # Wait for any job to finish
    wait -n
    # Remove completed job PIDs
    pids=($(jobs -p))
    # Start a new job
    start_job
done

# Wait for all remaining jobs to finish
wait

echo "All jobs completed"
