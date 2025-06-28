#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>

// Your exact original (baseline)
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

// Micro-optimization 1: High priority stream
extern "C" void micro_opt_priority_stream(const char* filename, void* gpu_dst, size_t total_size) {
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

    // Only change: Create high-priority stream
    cudaStream_t stream;
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority_high);

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

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

    printf("[MICRO-PRIORITY] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Micro-optimization 2: Batch submit (submit every few ops vs every op)
extern "C" void micro_opt_batch_submit(const char* filename, void* gpu_dst, size_t total_size) {
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

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

    int pending_submits = 0;  // Track pending submissions
    
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
            pending_submits++;
            
            // Only change: Batch submit every 4 operations instead of every 1
            if (pending_submits >= 4) {
                io_uring_submit(&ring);
                pending_submits = 0;
            }
        }
    }
    
    // Submit any remaining
    if (pending_submits > 0) {
        io_uring_submit(&ring);
    }

    cudaStreamSynchronize(stream);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[MICRO-BATCH] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Micro-optimization 3: Slightly larger blocks (6MB instead of 4MB)
extern "C" void micro_opt_6mb_blocks(const char* filename, void* gpu_dst, size_t total_size) {
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

    const size_t block_size = 6 * 1024 * 1024;  // Only change: 4MB -> 6MB
    void* buffers[32];
    for (int i = 0; i < 32; ++i) {
        cudaHostAlloc(&buffers[i], block_size, cudaHostAllocDefault);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

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

        cudaMemcpyAsync((char*)gpu_dst + total_read,  // Use actual read offset
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

    printf("[MICRO-6MB] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Micro-optimization 4: Non-blocking stream + process overlapping
extern "C" void micro_opt_nonblocking(const char* filename, void* gpu_dst, size_t total_size) {
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

    // Only change: Non-blocking stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    size_t submitted = 0, inflight = 0, completed = 0;

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

    printf("[MICRO-NONBLOCK] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}