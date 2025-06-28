import torch
import time
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ----------------------- Config -----------------------
shape_fixed = (32, 1024)
dtype = torch.float16
device = 'cuda'
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
save_dir = '/mnt/kvcache'  # Change as needed
os.makedirs(save_dir, exist_ok=True)

# ------------------ Dataset Class ---------------------
class TensorFileDataset(Dataset):
    def __init__(self, batch_sizes, dir_path):
        self.files = [os.path.join(dir_path, f"tensor_{B}.pt") for B in batch_sizes]
        self.batch_sizes = batch_sizes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        start = time.perf_counter()
        tensor = torch.load(self.files[idx], map_location='cpu')
        end = time.perf_counter()
        return tensor, self.batch_sizes[idx], end - start  # Disk-to-CPU time

# ---------------------- Main --------------------------
def main():
    # Step 1: Write benchmark
    write_times = []
    for B in batch_sizes:
        shape = (B, *shape_fixed)
        tensor_gpu = torch.full(shape, 32, dtype=dtype, device=device)
        filename = os.path.join(save_dir, f"tensor_{B}.pt")
        tensor_cpu = tensor_gpu.cpu()

        start = time.perf_counter()
        torch.save(tensor_cpu, filename)
        end = time.perf_counter()
        write_times.append(end - start)

    # Step 2: Read benchmark
    dataset = TensorFileDataset(batch_sizes, save_dir)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    read_times = []
    transfer_times = []

    for i, (tensor_cpu, B, read_time) in enumerate(loader):
        start = time.perf_counter()
        tensor_gpu = tensor_cpu.squeeze(0).to(device, non_blocking=True)
        end = time.perf_counter()

        read_times.append(read_time.item())
        transfer_times.append(end - start)

    # Cleanup
    for B in batch_sizes:
        os.remove(os.path.join(save_dir, f"tensor_{B}.pt"))

    # Step 3: Bandwidth Calculation
    element_size = torch.tensor([], dtype=dtype).element_size()
    bandwidth_write = []
    bandwidth_read = []

    for i, B in enumerate(batch_sizes):
        num_elements = B * shape_fixed[0] * shape_fixed[1]
        total_bytes = num_elements * element_size
        total_gb = total_bytes / (1024 ** 3)

        bw_w = total_gb / write_times[i]
        bw_r = total_gb / (read_times[i] + transfer_times[i])

        bandwidth_write.append(bw_w)
        bandwidth_read.append(bw_r)

    # Step 4: Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, bandwidth_write, label='GPU → CPU + Write (torch.save)', marker='o')
    plt.plot(batch_sizes, bandwidth_read, label='Disk → CPU + GPU Load (torch.load + to)', marker='o')
    plt.xlabel('Batch Size (B)')
    plt.ylabel('Effective Bandwidth (GB/s)')
    plt.title('Torch Save/Load and Transfer Bandwidth vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ssd_speeds_torch_io.png")

# ------------------ Safe Entry ------------------------
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
