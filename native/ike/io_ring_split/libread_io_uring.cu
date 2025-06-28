#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

#define QUEUE_DEPTH 32
#define MY_BLOCK_SIZE (16 * 1024 * 1024) // 16 MiB blocks
#define NUM_STREAMS 32

extern "C" {

// Global variables to hold resources
static struct io_uring ring;
static void* buffers[QUEUE_DEPTH];
static cudaStream_t streams[NUM_STREAMS];
static int fd = -1;
static int initialized = 0;

// Initialize io_uring, open file, allocate pinned memory and create CUDA streams
int io_uring_cuda_init(const char* filename) {
    if (initialized) return 0; // Already initialized, do nothing

    fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    // Initialize io_uring queue with defined depth
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return -2;
    }

    // Allocate pinned host memory buffers for efficient DMA transfer
    for (int i = 0; i < QUEUE_DEPTH; i++) {
        cudaHostAlloc(&buffers[i], MY_BLOCK_SIZE, cudaHostAllocDefault);
    }

    // Create multiple CUDA streams for concurrent data copy to GPU
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    initialized = 1; // Mark as initialized
    return 0;
}

// Perform a single read and copy operation from SSD to GPU memory
int io_uring_cuda_run(void* gpu_dst, size_t total_size) {
    if (!initialized) {
        fprintf(stderr, "Error: io_uring_cuda_run called before initialization.\n");
        return -1;
    }

    struct io_uring_cqe* cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

    // Submit initial batch of read requests
    while (submitted < QUEUE_DEPTH && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], MY_BLOCK_SIZE, offset);
        sqe->user_data = submitted;
        offset += MY_BLOCK_SIZE;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    // Process completed I/O events and launch asynchronous GPU copy
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

        int stream_idx = completed % NUM_STREAMS;
        cudaMemcpyAsync((char*)gpu_dst + completed * MY_BLOCK_SIZE,
                         buffers[buf_idx],
                         this_size,
                         cudaMemcpyHostToDevice,
                         streams[stream_idx]);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        // Submit next read request if remaining data
        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[buf_idx], MY_BLOCK_SIZE, offset);
            sqe->user_data = buf_idx;
            offset += MY_BLOCK_SIZE;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    // Synchronize all CUDA streams to ensure data copy completion
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    return 0;
}

// Clean up all allocated resources: CUDA streams, io_uring queue, pinned memory and close file
void io_uring_cuda_close() {
    if (!initialized) return;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    io_uring_queue_exit(&ring);

    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        cudaFreeHost(buffers[i]);
    }

    close(fd);

    initialized = 0; // Mark as uninitialized
}

} // extern "C"