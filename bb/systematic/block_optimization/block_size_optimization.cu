#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

// Your proven single-threaded approach with configurable block size
void test_block_size(const char* filename, void* gpu_dst, size_t total_size, 
                    size_t block_size, const char* label) {
    printf("%s test (%zu KB blocks)\n", label, block_size / 1024);
    
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(32, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    void* buffers[32];
    for (int i = 0; i < 32; ++i) {
        cudaHostAlloc(&buffers[i], block_size, cudaHostAllocDefault);
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    while (submitted < 32 && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + block_size > total_size) ? 
                          (total_size - offset) : block_size;
        io_uring_prep_read(sqe, fd, buffers[submitted], read_size, offset);
        sqe->user_data = submitted;
        offset += read_size;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    while (total_read < total_size) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) {
            perror("io_uring_wait_cqe");
            break;
        }

        int buf_idx = cqe->user_data;
        size_t this_size = cqe->res;

        if ((int)this_size <= 0) {
            fprintf(stderr, "Read failed: %d\n", cqe->res);
            break;
        }

        cudaMemcpyAsync((char*)gpu_dst + total_read,
                        buffers[buf_idx],
                        this_size,
                        cudaMemcpyHostToDevice,
                        stream);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = (offset + block_size > total_size) ? 
                              (total_size - offset) : block_size;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    cudaStreamSynchronize(stream);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[%s] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           label, total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Test various block sizes
extern "C" void test_512kb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 512 * 1024, "512KB-BLOCKS");
}

extern "C" void test_1mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 1024 * 1024, "1MB-BLOCKS");
}

extern "C" void test_2mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 2 * 1024 * 1024, "2MB-BLOCKS");
}

extern "C" void test_4mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 4 * 1024 * 1024, "4MB-BLOCKS");
}

extern "C" void test_8mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 8 * 1024 * 1024, "8MB-BLOCKS");
}

extern "C" void test_16mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 16 * 1024 * 1024, "16MB-BLOCKS");
}

// Your original for comparison
extern "C" void read_to_gpu_io_uring(const char* filename, void* gpu_dst, size_t total_size) {
    test_block_size(filename, gpu_dst, total_size, 4 * 1024 * 1024, "ORIGINAL-4MB");
}

// Advanced version with different queue depths
extern "C" void test_1mb_qd64(const char* filename, void* gpu_dst, size_t total_size) {
    printf("1MB blocks with QD=64 (vs QD=32)\n");
    
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(64, &ring, 0)) {  // Higher queue depth
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    const size_t block_size = 1024 * 1024;
    void* buffers[64];  // More buffers
    for (int i = 0; i < 64; ++i) {
        cudaHostAlloc(&buffers[i], block_size, cudaHostAllocDefault);
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    while (submitted < 64 && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + block_size > total_size) ? 
                          (total_size - offset) : block_size;
        io_uring_prep_read(sqe, fd, buffers[submitted], read_size, offset);
        sqe->user_data = submitted;
        offset += read_size;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    while (total_read < total_size) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) {
            perror("io_uring_wait_cqe");
            break;
        }

        int buf_idx = cqe->user_data;
        size_t this_size = cqe->res;

        if ((int)this_size <= 0) {
            fprintf(stderr, "Read failed: %d\n", cqe->res);
            break;
        }

        cudaMemcpyAsync((char*)gpu_dst + total_read,
                        buffers[buf_idx],
                        this_size,
                        cudaMemcpyHostToDevice,
                        stream);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = (offset + block_size > total_size) ? 
                              (total_size - offset) : block_size;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    cudaStreamSynchronize(stream);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[1MB-QD64] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 64; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}