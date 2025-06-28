// optimized_read_to_gpu.cu - Fixed systematic test version
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

// Optimized parameters - avoid BLOCK_SIZE name conflict
#define MY_QUEUE_DEPTH 64
#define MY_BLOCK_SIZE (4 * 1024 * 1024)  // Keep 4MB - it works well
#define NUM_STREAMS 2  // Modest increase

// Your exact original working function
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

        cudaMemcpyAsync((char*)gpu_dst + completed * 4 * 1024 * 1024,
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
            io_uring_prep_read(sqe, fd, buffers[buf_idx], 4 * 1024 * 1024, offset);
            sqe->user_data = buf_idx;
            offset += 4 * 1024 * 1024;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    cudaStreamSynchronize(stream);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[ORIGINAL] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Conservative optimization - just increase queue depth
extern "C" void optimized_v1_read_to_gpu(const char* filename, void* gpu_dst, size_t total_size) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(MY_QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    void* buffers[MY_QUEUE_DEPTH];
    for (int i = 0; i < MY_QUEUE_DEPTH; ++i) {
        cudaHostAlloc(&buffers[i], MY_BLOCK_SIZE, cudaHostAllocDefault);
    }

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

    while (submitted < MY_QUEUE_DEPTH && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? (total_size - offset) : MY_BLOCK_SIZE;
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

        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? (total_size - offset) : MY_BLOCK_SIZE;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[OPTIMIZED-V1] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    io_uring_queue_exit(&ring);
    for (int i = 0; i < MY_QUEUE_DEPTH; ++i) {
        cudaFreeHost(buffers[i]);
    }
    close(fd);
}

// More aggressive optimization with larger blocks
extern "C" void optimized_v2_read_to_gpu(const char* filename, void* gpu_dst, size_t total_size) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(MY_QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    const size_t large_block = 8 * 1024 * 1024;  // 8MB blocks
    void* buffers[MY_QUEUE_DEPTH];
    for (int i = 0; i < MY_QUEUE_DEPTH; ++i) {
        cudaHostAlloc(&buffers[i], large_block, cudaHostAllocWriteCombined);
    }

    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    while (submitted < MY_QUEUE_DEPTH && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + large_block > total_size) ? (total_size - offset) : large_block;
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

        int stream_idx = completed % 4;
        cudaMemcpyAsync((char*)gpu_dst + total_read,  // Use actual offset instead of completed * block
                        buffers[buf_idx],
                        this_size,
                        cudaMemcpyHostToDevice,
                        streams[stream_idx]);

        total_read += this_size;
        completed++;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        if (offset < total_size) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = (offset + large_block > total_size) ? (total_size - offset) : large_block;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    for (int i = 0; i < 4; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[OPTIMIZED-V2] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    for (int i = 0; i < 4; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    io_uring_queue_exit(&ring);
    for (int i = 0; i < MY_QUEUE_DEPTH; ++i) {
        cudaFreeHost(buffers[i]);
    }
    close(fd);
}

// Experimental version with different memory allocation
extern "C" void optimized_v3_read_to_gpu(const char* filename, void* gpu_dst, size_t total_size) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(96, &ring, 0)) {  // Even higher queue depth
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    void* buffers[96];
    for (int i = 0; i < 96; ++i) {
        // Try portable allocation
        cudaHostAlloc(&buffers[i], MY_BLOCK_SIZE, cudaHostAllocPortable);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);  // Back to single stream

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    while (submitted < 96 && offset < total_size) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? (total_size - offset) : MY_BLOCK_SIZE;
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

        cudaMemcpyAsync((char*)gpu_dst + completed * MY_BLOCK_SIZE,
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
            size_t read_size = (offset + MY_BLOCK_SIZE > total_size) ? (total_size - offset) : MY_BLOCK_SIZE;
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

    printf("[OPTIMIZED-V3] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 96; ++i) {
        cudaFreeHost(buffers[i]);
    }
    close(fd);
}