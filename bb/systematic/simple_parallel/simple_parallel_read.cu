// fixed_simple_parallel_read.cu - Fixed OpenMP structured block issues
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>
#include <cuda_runtime.h>
#include <omp.h>

// Your exact original working function (baseline)
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

// Fixed dual-reader version with proper OpenMP structure
extern "C" void dual_reader_simple(const char* filename, void* gpu_dst, size_t total_size) {
    printf("Dual reader test (2 parallel file readers)\n");
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    size_t total_read_combined = 0;
    
    #pragma omp parallel num_threads(2) reduction(+:total_read_combined)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = total_size / 2;
        size_t start_offset = thread_id * chunk_size;
        size_t bytes_to_read = (thread_id == 1) ? (total_size - start_offset) : chunk_size;
        
        size_t local_read = 0;
        
        // Each thread opens its own file descriptor
        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            struct io_uring ring;
            if (io_uring_queue_init(32, &ring, 0) == 0) {
                
                void* buffers[32];
                bool buffers_allocated = true;
                for (int i = 0; i < 32; ++i) {
                    if (cudaHostAlloc(&buffers[i], 4 * 1024 * 1024, cudaHostAllocDefault) != cudaSuccess) {
                        buffers_allocated = false;
                        break;
                    }
                }

                if (buffers_allocated) {
                    cudaStream_t stream;
                    if (cudaStreamCreate(&stream) == cudaSuccess) {
                        
                        struct io_uring_cqe *cqe;
                        off_t offset = start_offset;
                        size_t submitted = 0, inflight = 0, completed = 0;

                        // Prime the ring
                        while (submitted < 32 && offset < start_offset + bytes_to_read) {
                            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                            if (sqe) {
                                size_t read_size = (offset + 4 * 1024 * 1024 > start_offset + bytes_to_read) ? 
                                                  (start_offset + bytes_to_read - offset) : 4 * 1024 * 1024;
                                io_uring_prep_read(sqe, fd, buffers[submitted], read_size, offset);
                                sqe->user_data = submitted;
                                offset += read_size;
                                submitted++;
                                inflight++;
                            }
                        }
                        io_uring_submit(&ring);

                        while (local_read < bytes_to_read) {
                            if (io_uring_wait_cqe(&ring, &cqe) < 0) break;

                            int buf_idx = cqe->user_data;
                            size_t this_size = cqe->res;

                            if ((int)this_size > 0) {
                                cudaMemcpyAsync((char*)gpu_dst + start_offset + local_read,
                                               buffers[buf_idx],
                                               this_size,
                                               cudaMemcpyHostToDevice,
                                               stream);

                                local_read += this_size;
                            }
                            
                            completed++;
                            inflight--;
                            io_uring_cqe_seen(&ring, cqe);

                            if (offset < start_offset + bytes_to_read) {
                                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                                if (sqe) {
                                    size_t read_size = (offset + 4 * 1024 * 1024 > start_offset + bytes_to_read) ? 
                                                      (start_offset + bytes_to_read - offset) : 4 * 1024 * 1024;
                                    io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
                                    sqe->user_data = buf_idx;
                                    offset += read_size;
                                    submitted++;
                                    inflight++;
                                    io_uring_submit(&ring);
                                }
                            }
                        }

                        cudaStreamSynchronize(stream);
                        cudaStreamDestroy(stream);
                    }
                    
                    for (int i = 0; i < 32; ++i) {
                        cudaFreeHost(buffers[i]);
                    }
                }
                
                io_uring_queue_exit(&ring);
            }
            close(fd);
        }

        total_read_combined += local_read;
        printf("Reader %d completed: %.2f MB\n", thread_id, local_read / 1.0e6);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[DUAL-READER] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read_combined / 1.0e6, elapsed, (total_read_combined / 1.0e6) / elapsed,
           (total_read_combined / 1.0e9) / elapsed);
}

// Fixed quad-reader version
extern "C" void quad_reader_simple(const char* filename, void* gpu_dst, size_t total_size) {
    printf("Quad reader test (4 parallel file readers)\n");
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    size_t total_read_combined = 0;
    
    #pragma omp parallel num_threads(4) reduction(+:total_read_combined)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = total_size / 4;
        size_t start_offset = thread_id * chunk_size;
        size_t bytes_to_read = (thread_id == 3) ? (total_size - start_offset) : chunk_size;
        
        size_t local_read = 0;
        
        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            struct io_uring ring;
            if (io_uring_queue_init(32, &ring, 0) == 0) {
                
                void* buffers[32];
                bool buffers_allocated = true;
                for (int i = 0; i < 32; ++i) {
                    if (cudaHostAlloc(&buffers[i], 4 * 1024 * 1024, cudaHostAllocDefault) != cudaSuccess) {
                        buffers_allocated = false;
                        break;
                    }
                }

                if (buffers_allocated) {
                    cudaStream_t stream;
                    if (cudaStreamCreate(&stream) == cudaSuccess) {
                        
                        struct io_uring_cqe *cqe;
                        off_t offset = start_offset;
                        size_t submitted = 0, inflight = 0, completed = 0;

                        while (submitted < 32 && offset < start_offset + bytes_to_read) {
                            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                            if (sqe) {
                                size_t read_size = (offset + 4 * 1024 * 1024 > start_offset + bytes_to_read) ? 
                                                  (start_offset + bytes_to_read - offset) : 4 * 1024 * 1024;
                                io_uring_prep_read(sqe, fd, buffers[submitted], read_size, offset);
                                sqe->user_data = submitted;
                                offset += read_size;
                                submitted++;
                                inflight++;
                            }
                        }
                        io_uring_submit(&ring);

                        while (local_read < bytes_to_read) {
                            if (io_uring_wait_cqe(&ring, &cqe) < 0) break;

                            int buf_idx = cqe->user_data;
                            size_t this_size = cqe->res;
                            
                            if ((int)this_size > 0) {
                                cudaMemcpyAsync((char*)gpu_dst + start_offset + local_read,
                                               buffers[buf_idx],
                                               this_size,
                                               cudaMemcpyHostToDevice,
                                               stream);
                                local_read += this_size;
                            }

                            completed++;
                            inflight--;
                            io_uring_cqe_seen(&ring, cqe);

                            if (offset < start_offset + bytes_to_read) {
                                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                                if (sqe) {
                                    size_t read_size = (offset + 4 * 1024 * 1024 > start_offset + bytes_to_read) ? 
                                                      (start_offset + bytes_to_read - offset) : 4 * 1024 * 1024;
                                    io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
                                    sqe->user_data = buf_idx;
                                    offset += read_size;
                                    submitted++;
                                    inflight++;
                                    io_uring_submit(&ring);
                                }
                            }
                        }

                        cudaStreamSynchronize(stream);
                        cudaStreamDestroy(stream);
                    }
                    
                    for (int i = 0; i < 32; ++i) {
                        cudaFreeHost(buffers[i]);
                    }
                }
                
                io_uring_queue_exit(&ring);
            }
            close(fd);
        }

        total_read_combined += local_read;
        printf("Reader %d completed: %.2f MB\n", thread_id, local_read / 1.0e6);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[QUAD-READER] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read_combined / 1.0e6, elapsed, (total_read_combined / 1.0e6) / elapsed,
           (total_read_combined / 1.0e9) / elapsed);
}

// Version with 1MB blocks aligned to your array chunk size
extern "C" void array_aligned_1mb(const char* filename, void* gpu_dst, size_t total_size) {
    printf("Array-aligned 1MB blocks (matching your array chunk size)\n");
    
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

    const size_t block_size = 1024 * 1024;  // 1MB to match array chunk size
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

    printf("[1MB-ALIGNED] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);

    cudaStreamDestroy(stream);
    io_uring_queue_exit(&ring);
    for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
    close(fd);
}

// Simple multi-file descriptor version without OpenMP
extern "C" void multi_fd_simple(const char* filename, void* gpu_dst, size_t total_size) {
    printf("Multi-FD simple test (2 file descriptors, sequential)\n");
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    size_t total_read = 0;
    size_t half_size = total_size / 2;
    
    // First half
    int fd1 = open(filename, O_RDONLY | O_DIRECT);
    if (fd1 >= 0) {
        struct io_uring ring1;
        if (io_uring_queue_init(32, &ring1, 0) == 0) {
            void* buffers1[32];
            for (int i = 0; i < 32; ++i) {
                cudaHostAlloc(&buffers1[i], 4 * 1024 * 1024, cudaHostAllocDefault);
            }

            cudaStream_t stream1;
            cudaStreamCreate(&stream1);

            struct io_uring_cqe *cqe;
            size_t local_read = 0;
            off_t offset = 0;
            size_t submitted = 0, inflight = 0, completed = 0;

            // Read first half
            while (submitted < 32 && offset < half_size) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring1);
                size_t read_size = (offset + 4 * 1024 * 1024 > half_size) ? 
                                  (half_size - offset) : 4 * 1024 * 1024;
                io_uring_prep_read(sqe, fd1, buffers1[submitted], read_size, offset);
                sqe->user_data = submitted;
                offset += read_size;
                submitted++;
                inflight++;
            }
            io_uring_submit(&ring1);

            while (local_read < half_size) {
                if (io_uring_wait_cqe(&ring1, &cqe) < 0) break;
                int buf_idx = cqe->user_data;
                size_t this_size = cqe->res;
                if ((int)this_size <= 0) break;

                cudaMemcpyAsync((char*)gpu_dst + local_read,
                               buffers1[buf_idx], this_size,
                               cudaMemcpyHostToDevice, stream1);

                local_read += this_size;
                completed++;
                inflight--;
                io_uring_cqe_seen(&ring1, cqe);

                if (offset < half_size) {
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring1);
                    size_t read_size = (offset + 4 * 1024 * 1024 > half_size) ? 
                                      (half_size - offset) : 4 * 1024 * 1024;
                    io_uring_prep_read(sqe, fd1, buffers1[buf_idx], read_size, offset);
                    sqe->user_data = buf_idx;
                    offset += read_size;
                    submitted++;
                    inflight++;
                    io_uring_submit(&ring1);
                }
            }

            cudaStreamSynchronize(stream1);
            total_read += local_read;
            
            cudaStreamDestroy(stream1);
            for (int i = 0; i < 32; ++i) cudaFreeHost(buffers1[i]);
            io_uring_queue_exit(&ring1);
        }
        close(fd1);
    }

    // Second half  
    int fd2 = open(filename, O_RDONLY | O_DIRECT);
    if (fd2 >= 0) {
        struct io_uring ring2;
        if (io_uring_queue_init(32, &ring2, 0) == 0) {
            void* buffers2[32];
            for (int i = 0; i < 32; ++i) {
                cudaHostAlloc(&buffers2[i], 4 * 1024 * 1024, cudaHostAllocDefault);
            }

            cudaStream_t stream2;
            cudaStreamCreate(&stream2);

            struct io_uring_cqe *cqe;
            size_t local_read = 0;
            off_t offset = half_size;
            size_t submitted = 0, inflight = 0, completed = 0;
            size_t remaining = total_size - half_size;

            while (submitted < 32 && offset < total_size) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring2);
                size_t read_size = (offset + 4 * 1024 * 1024 > total_size) ? 
                                  (total_size - offset) : 4 * 1024 * 1024;
                io_uring_prep_read(sqe, fd2, buffers2[submitted], read_size, offset);
                sqe->user_data = submitted;
                offset += read_size;
                submitted++;
                inflight++;
            }
            io_uring_submit(&ring2);

            while (local_read < remaining) {
                if (io_uring_wait_cqe(&ring2, &cqe) < 0) break;
                int buf_idx = cqe->user_data;
                size_t this_size = cqe->res;
                if ((int)this_size <= 0) break;

                cudaMemcpyAsync((char*)gpu_dst + half_size + local_read,
                               buffers2[buf_idx], this_size,
                               cudaMemcpyHostToDevice, stream2);

                local_read += this_size;
                completed++;
                inflight--;
                io_uring_cqe_seen(&ring2, cqe);

                if (offset < total_size) {
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring2);
                    size_t read_size = (offset + 4 * 1024 * 1024 > total_size) ? 
                                      (total_size - offset) : 4 * 1024 * 1024;
                    io_uring_prep_read(sqe, fd2, buffers2[buf_idx], read_size, offset);
                    sqe->user_data = buf_idx;
                    offset += read_size;
                    submitted++;
                    inflight++;
                    io_uring_submit(&ring2);
                }
            }

            cudaStreamSynchronize(stream2);
            total_read += local_read;
            
            cudaStreamDestroy(stream2);
            for (int i = 0; i < 32; ++i) cudaFreeHost(buffers2[i]);
            io_uring_queue_exit(&ring2);
        }
        close(fd2);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[MULTI-FD] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed,
           (total_read / 1.0e9) / elapsed);
}