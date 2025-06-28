import numpy as np
import threading
import time
from tqdm import tqdm

def write_file(batch_size, filename):
    data = np.random.randn(batch_size, 32, 1024).astype(np.float16)
    start = time.time()
    data.tofile(filename)
    end = time.time()
    dur = end - start
    bw = data.nbytes / dur / 1e9
    return bw

def read_file(filename, shape):
    start = time.time()
    data = np.fromfile(filename, dtype=np.float16)
    #data = data.reshape(shape)
    end = time.time()
    dur = end - start
    bw = data.nbytes / dur / 1e9
    return bw

def worker(batch_size, filename, bw_writes, bw_reads, idx):
    bw_writes[idx] = write_file(batch_size, filename)
    bw_reads[idx] = read_file(filename, (batch_size, 32, 1024))

def main():
    batches = list(range(100, 200))
    bw_writes = [0] * len(batches)
    bw_reads = [0] * len(batches)

    threads = []
    max_threads = 1

    for i in tqdm(range(0, len(batches), max_threads)):
        current_threads = []
        for j in range(i, min(i + max_threads, len(batches))):
            t = threading.Thread(target=worker, args=(batches[j], f"/mnt/kvcache/test_{j}.bin", bw_writes, bw_reads, j))
            t.start()
            current_threads.append(t)

        for t in current_threads:
            t.join()

    print(f"Average write bandwidth: {np.mean(bw_writes):.2f} GB/s")
    print(f"Average read bandwidth: {np.mean(bw_reads):.2f} GB/s")

if __name__ == "__main__":
    main()