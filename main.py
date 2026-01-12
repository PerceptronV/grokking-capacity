import os
import argparse
import sys
import signal
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


def signal_handler(signum, frame):
    """Handle keyboard interrupt"""
    print("\n\nKeyboard interrupt received. Shutting down gracefully...")
    shutdown_event.set()


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

    with print_lock:
        status = "✓" if exit_status == 0 else "✗"
        print(f"[{device_str}] {status} Completed")

    return exit_status


def run_experiment(p_start, p_end, training_fraction=0.5, seed=42,
                   max_workers=None, operation='/'):
    """Run grokking experiments across a range of primes with multithreading

    Note: Activate your conda environment before running this script:
    conda activate ml13 && python main.py --p-start 97 --p-end 113
    """

    # Get available CUDA devices
    num_devices = get_cuda_devices()
    print(f"Found {num_devices} CUDA device(s)")

    # Determine number of workers
    if max_workers is None:
        max_workers = max(1, num_devices) if num_devices > 0 else os.cpu_count()
    print(f"Using {max_workers} worker thread(s)\n")

    primes = primes_in_range(p_start, p_end)
    p_mid = primes[len(primes) // 2]

    speed_dims = list(range(60, 128, 4)) + list(range(128, 264, 8))
    grok_starts = [20, 130]
    grok_ends = [128, 200]
    grok_steps = [2, 10]

    # Collect all commands to run
    commands = []

    # Capacity experiment (single run)
    commands.append(f"python capacity.py --p {p_mid} --no-show --seed {seed}")

    # Speed and grokking experiments for each prime
    for p in primes:
        for d in speed_dims:
            n, size = compute_dataset_size_bits(p, operation, training_fraction)
            commands.append(
                f"python speed.py --p {p} --no-show --dims {d} --samples-start {n} "
                f"--samples-end {n} --samples-steps 1 --seed {seed}"
            )

        for gs, ge, gt in zip(grok_starts, grok_ends, grok_steps):
            commands.append(
                f"python groks.py --p {p} --no-show --dim-start {gs} --dim-end {ge} "
                f"--dim-step {gt} --train-fraction {training_fraction} --seed {seed} --epochs 5000"
            )

    print(f"Total commands to execute: {len(commands)}\n")

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
                # Assign device in round-robin fashion if GPUs available
                device_id = idx % num_devices if num_devices > 0 else None
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


def main():
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description='Run grokking experiments across a range of prime numbers with multithreading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--p-start', type=int, default=90,
                       help='Starting prime number for the range')
    parser.add_argument('--p-end', type=int, default=160,
                       help='Ending prime number for the range')

    # Experiment parameters
    parser.add_argument('--training-fraction', type=float, default=0.5,
                       help='Fraction of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--operation', type=str, default='/',
                       choices=['*', '/', '+', '-'],
                       help='Arithmetic operation to use')

    # Parallelization options
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (defaults to number of GPUs or CPUs)')

    args = parser.parse_args()

    # Run the experiment
    return_code = run_experiment(
        p_start=args.p_start,
        p_end=args.p_end,
        training_fraction=args.training_fraction,
        seed=args.seed,
        max_workers=args.max_workers,
        operation=args.operation
    )

    sys.exit(return_code)


if __name__ == '__main__':
    main()
