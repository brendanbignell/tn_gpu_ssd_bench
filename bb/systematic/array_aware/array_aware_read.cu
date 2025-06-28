// array_aware_read.cu - Optimized for 11x NVMe array with 1MB chunks
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <liburing.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

// Array-aware parameters
#define ARRAY_CHUNK_SIZE (1024 * 1024)     // 1MB to match your array chunk size
#define NUM_PARALLEL_READERS 8             // Multiple readers for array parallelism
#define QUEUE_DEPTH_PER_READER 32          // io_uring depth per reader
#define GPU_TRANSFER_THREADS 2              // Dedicated GPU transfer threads

// Your original single-threaded version (baseline)
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

// Data structure for coordinating between reader threads and GPU threads
struct DataChunk {
    void* host_buffer;
    size_t size;
    size_t file_offset;
    bool ready_for_gpu;
    int reader_id;
    
    DataChunk() : host_buffer(nullptr), size(0), file_offset(0), ready_for_gpu(false), reader_id(0) {}
};

// Global coordination
std::atomic<size_t> total_bytes_read{0};
std::atomic<size_t> total_bytes_gpu_copied{0};
std::atomic<bool> all_readers_finished{false};

// Thread-safe queue for GPU transfers
class ThreadSafeQueue {
private:
    std::queue<std::shared_ptr<DataChunk>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    
public:
    void push(std::shared_ptr<DataChunk> chunk) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(chunk);
        cv_.notify_one();
    }
    
    std::shared_ptr<DataChunk> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return !queue_.empty() || all_readers_finished.load(); });
        
        if (queue_.empty()) return nullptr;
        
        auto chunk = queue_.front();
        queue_.pop();
        return chunk;
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

ThreadSafeQueue gpu_transfer_queue;

// Array-aware reader thread - uses 1MB aligned reads to match array chunk size
void array_aware_reader(int reader_id, const char* filename, size_t start_offset, 
                       size_t bytes_to_read, cudaStream_t stream) {
    
    // Open with optimizations for your high-end setup
    int fd = open(filename, O_RDONLY | O_DIRECT | O_NOATIME);
    if (fd < 0) {
        fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            perror("open");
            return;
        }
    }

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH_PER_READER, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    // Allocate buffers aligned to 1MB for array optimization
    void* buffers[QUEUE_DEPTH_PER_READER];
    for (int i = 0; i < QUEUE_DEPTH_PER_READER; ++i) {
        if (posix_memalign(&buffers[i], ARRAY_CHUNK_SIZE, ARRAY_CHUNK_SIZE)) {
            perror("posix_memalign");
            return;
        }
        // Register as pinned memory
        cudaHostRegister(buffers[i], ARRAY_CHUNK_SIZE, cudaHostRegisterDefault);
    }

    struct io_uring_cqe *cqe;
    size_t local_read = 0;
    size_t offset = start_offset;
    size_t submitted = 0, inflight = 0;

    // Align start to 1MB boundary for array efficiency
    size_t aligned_start = (start_offset / ARRAY_CHUNK_SIZE) * ARRAY_CHUNK_SIZE;
    size_t aligned_end = ((start_offset + bytes_to_read + ARRAY_CHUNK_SIZE - 1) / ARRAY_CHUNK_SIZE) * ARRAY_CHUNK_SIZE;
    
    offset = aligned_start;

    // Prime the ring with 1MB aligned reads
    while (submitted < QUEUE_DEPTH_PER_READER && offset < aligned_end) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], ARRAY_CHUNK_SIZE, offset);
        sqe->user_data = submitted;
        offset += ARRAY_CHUNK_SIZE;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    while (local_read < bytes_to_read) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) {
            perror("io_uring_wait_cqe");
            break;
        }

        int buf_idx = cqe->user_data;
        size_t this_size = cqe->res;

        if ((int)this_size <= 0) {
            fprintf(stderr, "Reader %d: Read failed: %d\n", reader_id, cqe->res);
            break;
        }

        // Create chunk for GPU transfer
        auto chunk = std::make_shared<DataChunk>();
        
        // Allocate new pinned memory for this chunk
        cudaHostAlloc(&chunk->host_buffer, this_size, cudaHostAllocDefault);
        memcpy(chunk->host_buffer, buffers[buf_idx], this_size);
        
        chunk->size = this_size;
        chunk->file_offset = start_offset + local_read;
        chunk->ready_for_gpu = true;
        chunk->reader_id = reader_id;

        // Queue for GPU transfer
        gpu_transfer_queue.push(chunk);

        local_read += this_size;
        inflight--;
        io_uring_cqe_seen(&ring, cqe);

        // Submit next read if more data needed
        if (offset < aligned_end && local_read < bytes_to_read) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[buf_idx], ARRAY_CHUNK_SIZE, offset);
            sqe->user_data = buf_idx;
            offset += ARRAY_CHUNK_SIZE;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    // Wait for remaining operations
    while (inflight > 0) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        
        int buf_idx = cqe->user_data;
        size_t this_size = cqe->res;
        
        if (this_size > 0 && local_read < bytes_to_read) {
            auto chunk = std::make_shared<DataChunk>();
            cudaHostAlloc(&chunk->host_buffer, this_size, cudaHostAllocDefault);
            memcpy(chunk->host_buffer, buffers[buf_idx], this_size);
            chunk->size = this_size;
            chunk->file_offset = start_offset + local_read;
            chunk->ready_for_gpu = true;
            chunk->reader_id = reader_id;
            
            gpu_transfer_queue.push(chunk);
            local_read += this_size;
        }
        
        inflight--;
        io_uring_cqe_seen(&ring, cqe);
    }

    // Cleanup
    for (int i = 0; i < QUEUE_DEPTH_PER_READER; ++i) {
        cudaHostUnregister(buffers[i]);
        free(buffers[i]);
    }
    io_uring_queue_exit(&ring);
    close(fd);

    total_bytes_read.fetch_add(local_read);
    printf("Reader %d completed: %.2f MB\n", reader_id, local_read / 1.0e6);
}

// GPU transfer worker
void gpu_transfer_worker(void* gpu_dst, cudaStream_t stream) {
    size_t gpu_offset = 0;
    
    while (!all_readers_finished.load() || !gpu_transfer_queue.empty()) {
        auto chunk = gpu_transfer_queue.pop();
        if (!chunk) break;
        
        // Transfer to GPU at the correct offset
        cudaMemcpyAsync(
            static_cast<char*>(gpu_dst) + chunk->file_offset,
            chunk->host_buffer,
            chunk->size,
            cudaMemcpyHostToDevice,
            stream
        );
        
        total_bytes_gpu_copied.fetch_add(chunk->size);
        
        // Free the chunk memory
        cudaFreeHost(chunk->host_buffer);
    }
}

// Multi-threaded array-aware version targeting your 11x NVMe setup
extern "C" void array_aware_parallel_read(const char* filename, void* gpu_dst, size_t total_size) {
    printf("Array-aware parallel read for 11x NVMe array (1MB chunks, %d readers)\n", NUM_PARALLEL_READERS);
    
    // Reset global counters
    total_bytes_read.store(0);
    total_bytes_gpu_copied.store(0);
    all_readers_finished.store(false);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Create CUDA streams for GPU transfer workers
    std::vector<cudaStream_t> gpu_streams(GPU_TRANSFER_THREADS);
    for (int i = 0; i < GPU_TRANSFER_THREADS; ++i) {
        cudaStreamCreate(&gpu_streams[i]);
    }

    // Start GPU transfer workers
    std::vector<std::thread> gpu_workers;
    for (int i = 0; i < GPU_TRANSFER_THREADS; ++i) {
        gpu_workers.emplace_back(gpu_transfer_worker, gpu_dst, gpu_streams[i]);
    }

    // Start reader threads - each reads a chunk of the file
    std::vector<std::thread> readers;
    size_t bytes_per_reader = total_size / NUM_PARALLEL_READERS;
    
    for (int i = 0; i < NUM_PARALLEL_READERS; ++i) {
        size_t start_offset = i * bytes_per_reader;
        size_t bytes_to_read = (i == NUM_PARALLEL_READERS - 1) ? 
                              (total_size - start_offset) : bytes_per_reader;
        
        readers.emplace_back(array_aware_reader, i, filename, start_offset, 
                           bytes_to_read, gpu_streams[i % GPU_TRANSFER_THREADS]);
    }

    // Wait for all readers to complete
    for (auto& reader : readers) {
        reader.join();
    }
    
    all_readers_finished.store(true);

    // Wait for GPU transfers to complete
    for (auto& worker : gpu_workers) {
        worker.join();
    }

    // Synchronize all streams
    for (auto& stream : gpu_streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    size_t final_read = total_bytes_read.load();
    size_t final_gpu = total_bytes_gpu_copied.load();

    printf("[ARRAY-AWARE] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           final_read / 1.0e6, elapsed, (final_read / 1.0e6) / elapsed,
           (final_read / 1.0e9) / elapsed);
    printf("GPU transferred: %.2f MB\n", final_gpu / 1.0e6);
}

// Simpler multi-reader version using your proven single-reader approach
extern "C" void multi_reader_simple(const char* filename, void* gpu_dst, size_t total_size) {
    const int num_readers = 4;  // Conservative start
    printf("Simple multi-reader version (%d readers)\n", num_readers);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    std::vector<std::thread> readers;
    std::atomic<size_t> total_read{0};
    
    // Each reader gets its own file descriptor and GPU stream
    auto reader_worker = [&](int reader_id, size_t start_offset, size_t bytes_to_read) {
        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) return;

        struct io_uring ring;
        if (io_uring_queue_init(32, &ring, 0)) {
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
        size_t local_read = 0;
        off_t offset = start_offset;
        size_t submitted = 0, inflight = 0, completed = 0;

        while (submitted < 32 && offset < start_offset + bytes_to_read) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = std::min(4 * 1024 * 1024UL, bytes_to_read - (offset - start_offset));
            io_uring_prep_read(sqe, fd, buffers[submitted], read_size, offset);
            sqe->user_data = submitted;
            offset += read_size;
            submitted++;
            inflight++;
        }
        io_uring_submit(&ring);

        while (local_read < bytes_to_read) {
            if (io_uring_wait_cqe(&ring, &cqe) < 0) break;

            int buf_idx = cqe->user_data;
            size_t this_size = cqe->res;
            if ((int)this_size <= 0) break;

            cudaMemcpyAsync((char*)gpu_dst + start_offset + local_read,
                           buffers[buf_idx],
                           this_size,
                           cudaMemcpyHostToDevice,
                           stream);

            local_read += this_size;
            completed++;
            inflight--;
            io_uring_cqe_seen(&ring, cqe);

            if (offset < start_offset + bytes_to_read) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                size_t read_size = std::min(4 * 1024 * 1024UL, bytes_to_read - (offset - start_offset));
                io_uring_prep_read(sqe, fd, buffers[buf_idx], read_size, offset);
                sqe->user_data = buf_idx;
                offset += read_size;
                submitted++;
                inflight++;
                io_uring_submit(&ring);
            }
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        io_uring_queue_exit(&ring);
        for (int i = 0; i < 32; ++i) cudaFreeHost(buffers[i]);
        close(fd);

        total_read.fetch_add(local_read);
        printf("Simple reader %d: %.2f MB\n", reader_id, local_read / 1.0e6);
    };

    size_t bytes_per_reader = total_size / num_readers;
    for (int i = 0; i < num_readers; ++i) {
        size_t start_offset = i * bytes_per_reader;
        size_t bytes_to_read = (i == num_readers - 1) ? 
                              (total_size - start_offset) : bytes_per_reader;
        readers.emplace_back(reader_worker, i, start_offset, bytes_to_read);
    }

    for (auto& reader : readers) {
        reader.join();
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    size_t final_read = total_read.load();
    printf("[MULTI-SIMPLE] Read %.2f MB in %.2f sec -> %.2f MB/s (%.2f GB/s)\n",
           final_read / 1.0e6, elapsed, (final_read / 1.0e6) / elapsed,
           (final_read / 1.0e9) / elapsed);
}