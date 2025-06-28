# SPDX-License-Identifier: Apache-2.0
import time
from dataclasses import dataclass, field
from enum import StrEnum, auto

import numpy as np
import matplotlib.pyplot as plt
import nvtx
import torch

class Workflow(StrEnum):
    GPU_TO_CPU = auto()
    CPU_TO_GPU_PINNED = auto()
    CPU_TO_GPU_PAGED = auto()
    BIDIRECTIONAL = auto()


def to_gigabits(t: torch.Tensor):
    return t / 1e9


@dataclass
class WorkerConfig:
    id: str
    profile: bool
    rank: int
    num_warmups: int
    num_tries: int
    tensor_sizes: list[int]


@dataclass
class Estimation:
    mean: float = 0
    std: float = 0
    tensor_size: int = 0
    workflow: Workflow = None
    run_compute: bool = False
    times: list[float] = field(default_factory=list)

    def from_times(times: list[float], **kwargs):
        tensor = torch.tensor(times)

        return Estimation(mean=tensor.mean().item(),
                          std=tensor.std().item(),
                          times=times,
                          **kwargs)


class Worker:

    def run(self, worker_config: WorkerConfig):
        print(f'Run: {worker_config.id}')

        self.worker_config = worker_config

        self.cpu_to_gpu_stream = torch.cuda.Stream()
        self.gpu_to_cpu_stream = torch.cuda.Stream()
        self.exec_stream = torch.cuda.Stream()

        self._measure(int(1e4),
                      workflow=Workflow.BIDIRECTIONAL,
                      run_compute=True,
                      num_tries=worker_config.num_warmups)

        results = []

        for ts in worker_config.tensor_sizes:
            for workflow in Workflow:
                for run_compute in [True, False]:
                    print(f'Running: {ts} / {run_compute}')
                    est = self._measure(ts,
                                        workflow=workflow,
                                        run_compute=run_compute,
                                        num_tries=worker_config.num_tries)

                    results.append(est)

        return results

    def _measure(self, tensor_size: int, workflow: Workflow, run_compute: bool,
                 num_tries: int):
        tensor_cpu_paged = self._allocate_tensors(tensor_size,
                                                  pin_memory=False,
                                                  device='cpu')
        tensor_cpu_pinned = self._allocate_tensors(tensor_size,
                                                   pin_memory=True,
                                                   device='cpu')
        tensor_gpu = self._allocate_tensors(tensor_size, device='cuda')
        size = tensor_size
        times = []

        for _ in range(num_tries):
            start = time.time()

            if run_compute:
                self._exec()

            match workflow:
                case Workflow.GPU_TO_CPU:
                    self._copy_to_cpu(tensor_gpu)

                    self.gpu_to_cpu_stream.synchronize()
                case Workflow.CPU_TO_GPU_PAGED:
                    self._copy_to_gpu(tensor_cpu_paged)

                    self.cpu_to_gpu_stream.synchronize()
                case Workflow.CPU_TO_GPU_PINNED:
                    self._copy_to_gpu(tensor_cpu_pinned)

                    self.cpu_to_gpu_stream.synchronize()
                case Workflow.BIDIRECTIONAL:
                    self._copy_to_gpu(tensor_cpu_pinned)
                    self._copy_to_cpu(tensor_gpu)

                    self.cpu_to_gpu_stream.synchronize()
                    self.gpu_to_cpu_stream.synchronize()

            transfer_time = time.time() - start

            times.append(transfer_time)

            torch.cuda.synchronize()

        if workflow == Workflow.BIDIRECTIONAL:
            size *= 2

        times_t = torch.tensor(times)
        bandwidths = to_gigabits((size * 8) / times_t)

        return Estimation(mean=bandwidths.mean().item(),
                          std=bandwidths.std().item(),
                          tensor_size=tensor_size,
                          workflow=workflow,
                          run_compute=run_compute,
                          times=times)

    def _allocate_tensors(self, tensor_size: int, **kwargs):
        numel = tensor_size // 2

        return torch.empty(numel, dtype=torch.float16, **kwargs)

    def _exec(self):
        with torch.cuda.StreamContext(self.exec_stream), nvtx.annotate('Exec'):
            a = torch.empty((4096, 2048), device='cuda')
            b = torch.empty((2048, 4096), device='cuda')

            _ = a @ b

    def _copy_to_gpu(self, tensor: torch.Tensor):
        with torch.cuda.StreamContext(self.cpu_to_gpu_stream),\
             nvtx.annotate('CPU to GPU'):
            tensor.to('cuda', non_blocking=True)

    def _copy_to_cpu(self, tensor: torch.Tensor):
        with torch.cuda.StreamContext(self.gpu_to_cpu_stream),\
             nvtx.annotate('GPU to CPU'):
            tensor.to('cpu', non_blocking=True)

# Generate logarithmically spaced tensor sizes from 1KB (2^10) to 1GB (2^30)
def generate_tensor_sizes():
    start = 10  # 2^10 = 1KB
    end = 28    # 2^28 = 256MB
    exponents = np.linspace(start, end, num=20, dtype=int)
    return [1 << exp for exp in exponents]  # 1 << n is faster than 2**n

# Plotting function
def plot_times(results):
    # Filter results for CPU to GPU (Pinned Memory) and no compute
    workflow = Workflow.CPU_TO_GPU_PINNED
    filtered = [r for r in results if r.workflow == workflow and not r.run_compute]

    batch_sizes = [r.tensor_size for r in filtered]
    mean_times = [np.mean(r.times) for r in filtered]

    plt.figure(figsize=(12, 8))
    plt.plot(batch_sizes, mean_times, marker='o', linestyle='-', color='b')

    # Set x-axis to log scale (batch sizes span multiple orders of magnitude)
    plt.xscale('log', base=2)

    plt.xlabel('Batch Size (Bytes)')
    plt.ylabel('Mean Transfer Time (Seconds)')
    plt.title('CPU to GPU (Pinned) Transfer Time vs Batch Size')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Optional: Annotate points with sizes
    for i, (size, times) in enumerate(zip(batch_sizes, mean_times)):
        plt.annotate(f'{size / (1 << 20):.2f}MB', (size, times),
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig('times.png')

def plot_bw(results):
    # Filter results for CPU to GPU (Pinned Memory) and no compute
    workflow = Workflow.CPU_TO_GPU_PINNED
    filtered = [r for r in results if r.workflow == workflow and not r.run_compute]

    batch_sizes = [r.tensor_size for r in filtered]
    mean_bandwidths = [r.mean for r in filtered]

    plt.figure(figsize=(12, 6))

    # Plot with linear y-axis and log x-axis
    plt.plot(batch_sizes, mean_bandwidths, marker='o', linestyle='-', color='b')

    # Set x-axis to log scale (batch sizes span multiple orders of magnitude)
    plt.xscale('log', base=2)

    # Axis labels and title
    plt.xlabel('Batch Size (Bytes)', fontsize=12)
    plt.ylabel('Mean Bandwidth (Gbps)', fontsize=12)
    plt.title('CPU to GPU (Pinned) Transfer Bandwidth vs Batch Size', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Optional: Annotate each point with batch size in MB
    for i, (size, bw) in enumerate(zip(batch_sizes, mean_bandwidths)):
        plt.annotate(f'{size / (1 << 20):.2f}MB',
                     (batch_sizes[i], mean_bandwidths[i]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     fontsize=9)

    plt.savefig('bw.png')

# Example usage in your main script:
if __name__ == "__main__":
    tensor_sizes = generate_tensor_sizes()
    config = WorkerConfig(
        id="transfer_benchmark",
        profile=False,
        rank=0,
        num_warmups=5,
        num_tries=20,
        tensor_sizes=tensor_sizes
    )

    worker = Worker()
    results = worker.run(config)  # This will take time depending on hardware

    plot_times(results)
    plot_bw(results)