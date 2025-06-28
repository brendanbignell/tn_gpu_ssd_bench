// read_to_gpu_io_uring.cu
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

#define QUEUE_DEPTH 8
#define MY_BLOCK_SIZE (1 * 1024 * 1024 )  // 1 MiB

extern "C" void read_to_gpu_io_uring(const char* filename, void* gpu_dst, size_t total_size) {
    int fd = open(filename, O_RDONLY | O_DIRECT | O_NONBLOCK) ;
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
        cudaHostAlloc(&buffers[i], MY_BLOCK_SIZE, cudaHostAllocDefault);
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
    while (submitted < QUEUE_DEPTH && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], MY_BLOCK_SIZE, offset);
        sqe->user_data = submitted;
        offset += MY_BLOCK_SIZE;
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

        // Copy to GPU (offset = completed * MY_BLOCK_SIZE)
        cudaMemcpyAsync((char*)gpu_dst + completed * MY_BLOCK_SIZE,
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
            io_uring_prep_read(sqe, fd, buffers[buf_idx], MY_BLOCK_SIZE, offset);
            sqe->user_data = buf_idx;
            offset += MY_BLOCK_SIZE;
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

    printf("[IO_URING + CUDA] Read %.2f MB in %.2f sec -> %.2f MB/s\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed);

    // Cleanup
    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < QUEUE_DEPTH; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}
