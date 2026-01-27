import os
import argparse
import sys
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import torch
from utils import compute_dataset_size_bits

# Global flag for handling interrupts
shutdown_event = Event()


def primes_in_range(start, end):
    '''Sieve of Eratosthenes'''
    sieve = [True] * (end + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(end**0.5) + 1):
        if sieve[i]:
            sieve[i*i:end+1:i] = [False] * len(range(i*i, end+1, i))
    return [i for i in range(start, end + 1) if sieve[i]]


def kill_child_processes():
    """Kill all child python processes"""
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        if children:
            print(f"\nTerminating {len(children)} child process(es)...")
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait a bit for graceful termination
            _, alive = psutil.wait_procs(children, timeout=3)

            # Force kill any remaining processes
            for child in alive:
                try:
                    print(f"Force killing process {child.pid}...")
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
    except Exception as e:
        print(f"Error killing child processes: {e}")


def run_in_env(cmd):
    """Run command (assumes conda environment is already activated)"""
    # Just run the command directly - user should activate conda environment
    # before running this script (e.g., conda activate ml13 && python main.py ...)
    return os.system(cmd)


def get_cuda_devices():
    """Get number of available CUDA devices using torch"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_workers_per_gpu(device_id):
    """Determine number of workers a GPU can support based on compute capability

    Mapping based on GPU architecture:
    - Compute 9.0+ (H100, H200): 4 workers
    - Compute 8.0-8.9 (A100, A10): 3 workers
    - Compute 7.0-7.9 (V100, T4): 2 workers
    - Compute < 7.0: 1 worker
    """
    if not torch.cuda.is_available():
        return 1

    capability = torch.cuda.get_device_capability(device_id)
    major, minor = capability
    compute_capability = major + minor / 10

    if compute_capability >= 9.0:
        return 4  # H100, H200
    elif compute_capability >= 8.0:
        return 3  # A100, A10
    elif compute_capability >= 7.0:
        return 2  # V100, T4
    else:
        return 1  # Older GPUs


def get_total_workers():
    """Calculate total workers based on compute capability of all GPUs"""
    num_devices = get_cuda_devices()
    if num_devices == 0:
        return os.cpu_count()

    total_workers = 0
    gpu_info = []
    for device_id in range(num_devices):
        workers = get_workers_per_gpu(device_id)
        total_workers += workers

        # Get GPU name and capability for logging
        props = torch.cuda.get_device_properties(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        gpu_info.append({
            'id': device_id,
            'name': props.name,
            'capability': f"{capability[0]}.{capability[1]}",
            'workers': workers
        })

    return total_workers, gpu_info


def run_command_with_device(cmd, device_id, print_lock):
    """Run a command with specified CUDA device"""
    # Check if shutdown was requested
    if shutdown_event.is_set():
        return -1

    # Only add --device flag if a specific GPU is assigned
    # Otherwise let the script use its default (could be MPS, CPU, etc.)
    if device_id is not None:
        cmd_with_device = f"{cmd} --device cuda:{device_id}"
        device_str = f"cuda:{device_id}"
    else:
        cmd_with_device = cmd
        device_str = "default"

    with print_lock:
        print(f"[{device_str}] Running: {cmd}")

    exit_status = run_in_env(cmd_with_device)

    # Check if command was interrupted (exit code 130 = SIGINT, 256*130 = 33280 on some systems)
    # os.system returns the exit code in the wait() format, so SIGINT (2) is encoded as 2 or 130
    if exit_status in (2, 130, 33280) or (exit_status >> 8) == 130:
        with print_lock:
            print(f"[{device_str}] Interrupted by user")
        shutdown_event.set()
        return -1

    with print_lock:
        status = "✓" if exit_status == 0 else "✗"
        print(f"[{device_str}] {status} Completed")

    return exit_status


def run_experiment(commands, max_workers=None):
    """Run grokking experiments across a range of primes with multithreading

    Note: Activate your conda environment before running this script:
    conda activate ml13 && python main.py --p-start 97 --p-end 113
    """

    # Get available CUDA devices
    num_devices = get_cuda_devices()
    print(f"Found {num_devices} CUDA device(s)")

    # Determine number of workers based on GPU compute capability
    if max_workers is None:
        if num_devices > 0:
            max_workers, gpu_info = get_total_workers()
            print("\nGPU Worker Allocation:")
            for info in gpu_info:
                print(f"  GPU {info['id']}: {info['name']} "
                      f"(Compute {info['capability']}) -> {info['workers']} workers")
        else:
            max_workers = os.cpu_count()
            gpu_info = None
    else:
        # Manual worker count specified, but still get GPU info for device assignment
        if num_devices > 0:
            _, gpu_info = get_total_workers()
        else:
            gpu_info = None

    print(f"\nUsing {max_workers} worker thread(s)\n")
    print(f"Total commands to execute: {len(commands)}\n")

    # Create device assignment list based on workers per GPU
    device_assignment = []
    if num_devices > 0 and gpu_info is not None:
        for info in gpu_info:
            device_assignment.extend([info['id']] * info['workers'])

    # Execute commands in parallel with device assignment
    print_lock = Lock()
    failed_commands = []
    cancelled_commands = []

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_cmd = {}
            for idx, cmd in enumerate(commands):
                if shutdown_event.is_set():
                    break
                # Assign device in round-robin fashion from device_assignment list
                if device_assignment:
                    device_id = device_assignment[idx % len(device_assignment)]
                else:
                    device_id = None
                future = executor.submit(run_command_with_device, cmd, device_id,
                                        print_lock)
                future_to_cmd[future] = cmd

            # Wait for completion
            for future in as_completed(future_to_cmd):
                if shutdown_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_cmd:
                        if not f.done():
                            f.cancel()
                            cancelled_commands.append(future_to_cmd[f])
                    break

                cmd = future_to_cmd[future]
                try:
                    return_code = future.result()
                    if return_code == -1:
                        cancelled_commands.append(cmd)
                    elif return_code != 0:
                        failed_commands.append(cmd)
                except Exception as e:
                    with print_lock:
                        print(f"Error executing command: {cmd}")
                        print(f"Exception: {e}")
                    failed_commands.append(cmd)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt in main thread, shutting down...")
        shutdown_event.set()
        kill_child_processes()

    # Ensure all child processes are terminated
    kill_child_processes()

    # Report results
    print("\n" + "="*80)
    completed = len(commands) - len(failed_commands) - len(cancelled_commands)
    print(f"Execution summary: {completed}/{len(commands)} completed successfully")

    if cancelled_commands:
        print(f"\nCancelled commands ({len(cancelled_commands)}):")
        for cmd in cancelled_commands[:5]:  # Show first 5
            print(f"  - {cmd}")
        if len(cancelled_commands) > 5:
            print(f"  ... and {len(cancelled_commands) - 5} more")

    if failed_commands:
        print(f"\nFailed commands ({len(failed_commands)}):")
        for cmd in failed_commands:
            print(f"  - {cmd}")
        return 1

    if cancelled_commands:
        return 130  # Standard exit code for SIGINT

    return 0


def generate_commands(p_start, p_end, seeds, train_fraction=0.5, operation='/', split_type='random'):
    """Generate commands for the experiment"""
    primes = primes_in_range(p_start, p_end)
    p_mid = primes[len(primes) // 2]

    speed_dims = list(range(20, 128, 4)) + list(range(128, 264, 8))
    grok_starts = [20, 130]
    grok_ends = [128, 1000]
    grok_steps = [2, 10]

    # Collect all commands to run
    commands = []
    default_seed = seeds[0]
    
    # Capacity experiment (single run)
    commands.append(f"python capacity.py --p {p_mid} --no-show --seed {default_seed}")

    # Speed and grokking experiments for each prime
    for p in primes:
        for seed in seeds:
            for gs, ge, gt in zip(grok_starts, grok_ends, grok_steps):
                commands.append(
                    f"python groks.py --p {p} --dim-start {gs} --dim-end {ge} --dim-step {gt} "
                    f"--train-fraction {train_fraction} --seed {seed} --split-type {split_type} "
                    f"--op {operation} --ignore-memorisation --epochs 5000 --no-show"
                )
        
            n, size = compute_dataset_size_bits(p, operation, train_fraction)
            commands.append(
                f"python speed.py --p {p} --dims {' '.join(map(str, speed_dims))} "
                f"--samples-start {n} --samples-end {n} --samples-steps 1 --seed {seed} "
                f"--epochs 5000 --no-show"
            )

    return commands


def main():
    # Don't register custom SIGINT handler - let default KeyboardInterrupt work
    # signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description='Run grokking experiments across a range of prime numbers with multithreading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--p-start', type=int, default=40,
                       help='Starting prime number for the range')
    parser.add_argument('--p-end', type=int, default=160,
                       help='Ending prime number for the range')

    # Experiment parameters
    parser.add_argument('--train-fraction', type=float, default=0.5,
                       help='Fraction of data to use for training')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 42 + 10)),
                       help='Random seeds for reproducibility')
    parser.add_argument('--operation', type=str, default='/',
                       choices=['*', '/', '+', '-'],
                       help='Arithmetic operation to use')
    parser.add_argument('--split-type', type=str, default='random',
                       choices=['random', 'sequential', 'alternating'],
                       help='Type of train/val split')

    # Parallelization options
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (defaults to number of GPUs or CPUs)')

    args = parser.parse_args()

    # Run the experiment
    commands = generate_commands(
        p_start=args.p_start,
        p_end=args.p_end,
        train_fraction=args.train_fraction,
        seeds=args.seeds,
        operation=args.operation,
        split_type=args.split_type
    )
    return_code = run_experiment(commands, max_workers=args.max_workers)

    sys.exit(return_code)


if __name__ == '__main__':
    main()
