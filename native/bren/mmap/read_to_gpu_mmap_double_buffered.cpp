#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

size_t parse_size_arg(const char* arg) {
    char unit = arg[strlen(arg) - 1];
    size_t multiplier = 1;
    if (unit == 'G' || unit == 'g') multiplier = 1024L * 1024 * 1024;
    else if (unit == 'M' || unit == 'm') multiplier = 1024L * 1024;
    else if (unit == 'K' || unit == 'k') multiplier = 1024L;
    else return std::stoul(arg);
    return std::stoul(std::string(arg, strlen(arg) - 1)) * multiplier;
}

int main(int argc, char** argv) {
    size_t total_size = 1L * 1024 * 1024 * 1024;  // 1 GB
    int device_id = 0;
    const size_t chunk_size = 1024 * 1024;        // 1 MB
    const char* filepath = "/mnt/kvcache/huge_test_4tb.bin";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--total_size") == 0 && i + 1 < argc) {
            total_size = parse_size_arg(argv[++i]);
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        }
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        return 1;
    }

    size_t file_size = st.st_size;
    size_t max_offset = file_size - total_size;
    std::mt19937_64 rng(getpid() + time(nullptr));
    size_t offset = (rng() % (max_offset / chunk_size)) * chunk_size;

    void* src = mmap(nullptr, total_size, PROT_READ, MAP_PRIVATE, fd, offset);
    if (src == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    madvise(src, total_size, MADV_WILLNEED);

    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, total_size));

    // Allocate two pinned staging buffers for double buffering
    void* pinned_buffers[2];
    CUDA_CHECK(cudaHostAlloc(&pinned_buffers[0], chunk_size, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&pinned_buffers[1], chunk_size, cudaHostAllocDefault));

    // Create two streams
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    auto start = std::chrono::high_resolution_clock::now();

    size_t i = 0;
    for (; i + chunk_size <= total_size; i += chunk_size) {
        int buf_id = (i / chunk_size) % 2;
        size_t prev_i = i - 2 * chunk_size;

        // If buffer reuse is about to happen, sync previous stream
        if (i >= 2 * chunk_size)
            CUDA_CHECK(cudaStreamSynchronize(streams[buf_id]));

        std::memcpy(pinned_buffers[buf_id], static_cast<const char*>(src) + i, chunk_size);

        void* chunk_dst = static_cast<char*>(dst) + i;
        CUDA_CHECK(cudaMemcpyAsync(chunk_dst, pinned_buffers[buf_id], chunk_size,
                                   cudaMemcpyHostToDevice, streams[buf_id]));
    }

    // Handle final partial chunk
    size_t remaining = total_size - i;
    if (remaining > 0) {
        int buf_id = i / chunk_size % 2;
        CUDA_CHECK(cudaStreamSynchronize(streams[buf_id]));

        std::memcpy(pinned_buffers[buf_id], static_cast<const char*>(src) + i, remaining);
        void* chunk_dst = static_cast<char*>(dst) + i;
        CUDA_CHECK(cudaMemcpyAsync(chunk_dst, pinned_buffers[buf_id], remaining,
                                   cudaMemcpyHostToDevice, streams[buf_id]));
    }

    // Final sync
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double gbps = total_size / elapsed / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Elapsed time: " << elapsed << " s\n";
    std::cout << "Transfer bandwidth: " << gbps << " GB/s\n";

    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFreeHost(pinned_buffers[0]));
    CUDA_CHECK(cudaFreeHost(pinned_buffers[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

    munmap(src, total_size);
    close(fd);

    return 0;
}
