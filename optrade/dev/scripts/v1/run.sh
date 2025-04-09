#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

# source /home/fam/xmootoo/torch2024_new/bin/activate # Venv activation (optional)

# Define list of ablations
ablations=(
   "forecasting/test"
)

# Set the number of parallel jobs (default to 2 if not specified)
parallel=1

# Function to run a single ablation
run_ablation() {
    local ablation=$1
    echo "Running ablation: $ablation"
    python time_series/tuning/tune.py "$ablation"
}

# Function to run jobs in parallel
run_parallel_jobs() {
    local max_jobs=$1
    local job_queue=("${ablations[@]}")
    local active_jobs=()

    while [ ${#job_queue[@]} -gt 0 ] || [ ${#active_jobs[@]} -gt 0 ]; do
        # Start new jobs if there's room
        while [ ${#active_jobs[@]} -lt $max_jobs ] && [ ${#job_queue[@]} -gt 0 ]; do
            local job=${job_queue[0]}
            job_queue=("${job_queue[@]:1}")  # Remove first element from queue
            run_ablation "$job" &
            local pid=$!
            active_jobs+=("$pid")
            echo "Started job $job with PID $pid"
        done

        # Wait for any job to finish
        wait -n 2>/dev/null

        # Remove finished jobs from active_jobs
        for pid in "${active_jobs[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                active_jobs=(${active_jobs[@]/$pid})
                echo "Job with PID $pid finished"
            fi
        done
    done
}

echo "Running ablation jobs with parallelism: $parallel"
run_parallel_jobs $parallel

echo "All ablations completed"
