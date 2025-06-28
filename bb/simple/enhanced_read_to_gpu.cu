// enhanced_read_to_gpu.cu - Simple enhancement of your working code
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

#define QUEUE_DEPTH 64        // Increased from 32
#define MY_BLOCK_SIZE (8 * 1024 * 1024)  // Increased to 8 MiB
#define NUM_STREAMS 4         // Multiple CUDA streams

extern "C" void enhanced_read_to_gpu(const char* filename, void* gpu_dst, size_t total_size) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    // Allocate pinned host memory buffers
    void* buffers[QUEUE_DEPTH];
    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        // Use write-combined memory for better GPU transfer performance
        cudaHostAlloc(&buffers[i], MY_BLOCK_SIZE, 
                     cudaHostAllocWriteCombined | cudaHostAllocDefault);
    }

    // Create multiple CUDA streams for parallel transfers
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Prime the ring with initial reads
    while (submitted < QUEUE_DEPTH && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? 
                          (total_size - offset) : MY_BLOCK_SIZE;
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

        // Use round-robin stream assignment for parallel GPU transfers
        int stream_idx = completed % NUM_STREAMS;
        
        // Copy to GPU using the appropriate stream
        cudaMemcpyAsync((char*)gpu_dst + completed * MY_BLOCK_SIZE,
                        buffers[buf_idx],
                        this_size,
                        cudaMemcpyHostToDevice,
                        streams[stream_idx]);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        // Re-submit if more data remains
        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? 
                              (total_size - offset) : MY_BLOCK_SIZE;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    // Ensure all GPU copies are complete
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[ENHANCED] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    io_uring_queue_exit(&ring);
    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        cudaFreeHost(buffers[i]);
    }
    close(fd);
}

// Your original function for comparison
extern "C" void read_to_gpu_io_uring(const char* filename, void* gpu_dst, size_t total_size) {
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

    // Allocate pinned host memory buffers
    void* buffers[32];
    for (int i = 0; i < 32; ++i) {
        cudaHostAlloc(&buffers[i], 4 * 1024 * 1024, cudaHostAllocDefault);
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Prime the ring with initial reads
    while (submitted < 32 && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], 4 * 1024 * 1024, offset);
        sqe->user_data = submitted;
        offset += 4 * 1024 * 1024;
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

        // Copy to GPU (offset = completed * 4MB)
        cudaMemcpyAsync((char*)gpu_dst + completed * 4 * 1024 * 1024,
                        buffers[buf_idx],
                        this_size,
                        cudaMemcpyHostToDevice,
                        stream);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        // Re-submit if more data remains
        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[buf_idx], 4 * 1024 * 1024, offset);
            sqe->user_data = buf_idx;
            offset += 4 * 1024 * 1024;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    // Ensure all GPU copies are complete
    cudaStreamSynchronize(stream);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[ORIGINAL] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    // Cleanup
    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}