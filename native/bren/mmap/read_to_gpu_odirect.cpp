#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <errno.h>
#include <malloc.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            return 1; \
        } \
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
    size_t total_size = 1L * 1024 * 1024 * 1024;  // default: 1 GB
    int device_id = 0;
    const size_t chunk_size = 11 * 1024 * 1024;        // 1 MB
    const size_t alignment = 4096;                // required for O_DIRECT
    const char* filepath = "/mnt/kvcache/huge_test_4tb.bin";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--total_size") == 0 && i + 1 < argc)
            total_size = parse_size_arg(argv[++i]);
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device_id = std::atoi(argv[++i]);
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    // Open file with O_DIRECT to bypass kernel page cache
    int fd = open(filepath, O_RDONLY | O_DIRECT);
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

    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, total_size));

    // Allocate two aligned host buffers for double-buffered read
    void* buffer[2];
    if (posix_memalign(&buffer[0], alignment, chunk_size) != 0 ||
        posix_memalign(&buffer[1], alignment, chunk_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        return 1;
    }

    // Create two CUDA streams
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    auto start = std::chrono::high_resolution_clock::now();

    size_t i = 0;
    for (; i + chunk_size <= total_size; i += chunk_size) {
        int buf_id = (i / chunk_size) % 2;
        size_t file_offset = offset + i;

        // Sync previous stream before reusing buffer
        if (i >= 2 * chunk_size)
            CUDA_CHECK(cudaStreamSynchronize(streams[buf_id]));

        ssize_t bytes_read = pread(fd, buffer[buf_id], chunk_size, file_offset);
        if (bytes_read < 0) {
            perror("pread");
            return 1;
        }

        void* chunk_dst = static_cast<char*>(dst) + i;
        CUDA_CHECK(cudaMemcpyAsync(chunk_dst, buffer[buf_id], chunk_size,
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

    // Cleanup
    CUDA_CHECK(cudaFree(dst));
    free(buffer[0]);
    free(buffer[1]);
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    close(fd);

    return 0;
}
