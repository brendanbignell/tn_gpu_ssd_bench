#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <atomic>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>

constexpr size_t DEFAULT_DISK_BLOCK_SIZE = 1 << 20;      // 1 MiB
constexpr size_t DEFAULT_GPU_CHUNK_SIZE = 16 << 20;      // 16 MiB  
constexpr size_t DEFAULT_TOTAL_SIZE = 10L * 1024 * 1024 * 1024; // 10 GiB
constexpr int DEFAULT_QUEUE_DEPTH = 32;
constexpr int PINNED_BUFFER_COUNT = 64;  // Number of pinned buffers for GPU transfers

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Pinned memory buffer for GPU transfers
struct PinnedBuffer {
    void* host_ptr;
    size_t capacity;
    size_t used;
    bool in_use;
    
    PinnedBuffer(size_t size) : capacity(size), used(0), in_use(false) {
        CUDA_CHECK(cudaHostAlloc(&host_ptr, size, cudaHostAllocDefault));
    }
    
    ~PinnedBuffer() {
        if (host_ptr) {
            cudaFreeHost(host_ptr);
        }
    }
    
    void reset() {
        used = 0;
        in_use = false;
    }
    
    bool can_fit(size_t size) const {
        return used + size <= capacity;
    }
    
    void* append(const void* data, size_t size) {
        if (!can_fit(size)) return nullptr;
        void* dest = static_cast<char*>(host_ptr) + used;
        memcpy(dest, data, size);
        used += size;
        return dest;
    }
};

// Global state
std::vector<std::unique_ptr<PinnedBuffer>> pinned_buffers;
std::mutex pinned_mutex;
std::condition_variable pinned_cv;
std::queue<PinnedBuffer*> ready_buffers;
std::queue<PinnedBuffer*> available_buffers;

std::atomic<size_t> global_total_read{0};
std::atomic<size_t> global_total_gpu_copied{0};
std::atomic<bool> io_finished{false};

// Get an available pinned buffer
PinnedBuffer* get_available_buffer() {
    std::unique_lock<std::mutex> lock(pinned_mutex);
    pinned_cv.wait(lock, []{ return !available_buffers.empty(); });
    
    PinnedBuffer* buffer = available_buffers.front();
    available_buffers.pop();
    buffer->reset();
    buffer->in_use = true;
    return buffer;
}

// Mark buffer as ready for GPU transfer
void mark_buffer_ready(PinnedBuffer* buffer) {
    std::lock_guard<std::mutex> lock(pinned_mutex);
    ready_buffers.push(buffer);
    pinned_cv.notify_all();
}

// Return buffer to available pool
void return_buffer(PinnedBuffer* buffer) {
    std::lock_guard<std::mutex> lock(pinned_mutex);
    buffer->reset();
    available_buffers.push(buffer);
    pinned_cv.notify_all();
}

void gpu_transfer_worker(void* d_buffer, size_t total_gpu_size, cudaStream_t stream) {
    size_t gpu_offset = 0;
    
    while (true) {
        PinnedBuffer* buffer = nullptr;
        
        {
            std::unique_lock<std::mutex> lock(pinned_mutex);
            pinned_cv.wait(lock, []{ return !ready_buffers.empty() || io_finished.load(); });
            
            if (ready_buffers.empty() && io_finished.load()) {
                break;
            }
            
            if (!ready_buffers.empty()) {
                buffer = ready_buffers.front();
                ready_buffers.pop();
            }
        }
        
        if (buffer && buffer->used > 0) {
            // Copy buffer to GPU using pinned memory
            if (gpu_offset + buffer->used <= total_gpu_size) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(d_buffer) + gpu_offset,
                    buffer->host_ptr,
                    buffer->used,
                    cudaMemcpyHostToDevice,
                    stream
                ));
                
                gpu_offset += buffer->used;
                global_total_gpu_copied.fetch_add(buffer->used);
            }
            
            // Return buffer to available pool
            return_buffer(buffer);
        }
    }
    
    // Synchronize the stream to ensure all transfers complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void io_uring_read_worker(const char* filename, size_t start_offset, size_t size_to_read, 
                         int queue_depth, size_t disk_block_size, size_t gpu_chunk_size, int worker_id) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    struct io_uring ring;
    if (io_uring_queue_init(queue_depth, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    // Allocate aligned buffers for direct I/O
    std::vector<void*> io_buffers(queue_depth);
    for (int i = 0; i < queue_depth; ++i) {
        if (posix_memalign(&io_buffers[i], 4096, disk_block_size)) {
            perror("posix_memalign");
            return;
        }
    }

    size_t offset = start_offset;
    size_t total_read = 0;
    int inflight = 0;
    int submitted = 0;
    
    // Current pinned buffer for accumulating data
    PinnedBuffer* current_buffer = get_available_buffer();

    // Fill submission queue
    while (submitted < queue_depth && offset < start_offset + size_to_read) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        size_t read_size = std::min(disk_block_size, size_to_read - (offset - start_offset));
        io_uring_prep_read(sqe, fd, io_buffers[submitted], read_size, offset);
        sqe->user_data = submitted; // Store buffer index
        offset += read_size;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    struct io_uring_cqe* cqe;
    while (total_read < size_to_read) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        if (cqe->res < 0) {
            std::cerr << "Async read failed: " << strerror(-cqe->res) << "\n";
            break;
        }

        int buf_idx = static_cast<int>(cqe->user_data);
        size_t bytes_read = cqe->res;
        
        // Try to append to current pinned buffer
        if (!current_buffer->can_fit(bytes_read)) {
            // Current buffer is full, send to GPU and get new one
            if (current_buffer->used > 0) {
                mark_buffer_ready(current_buffer);
            } else {
                return_buffer(current_buffer);
            }
            current_buffer = get_available_buffer();
        }
        
        // Append data to pinned buffer
        current_buffer->append(io_buffers[buf_idx], bytes_read);
        
        // If buffer is now at target chunk size, send to GPU
        if (current_buffer->used >= gpu_chunk_size) {
            mark_buffer_ready(current_buffer);
            current_buffer = get_available_buffer();
        }

        total_read += bytes_read;
        io_uring_cqe_seen(&ring, cqe);
        inflight--;

        // Submit next read if more data to read
        if (offset < start_offset + size_to_read) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            size_t read_size = std::min(disk_block_size, size_to_read - (offset - start_offset));
            io_uring_prep_read(sqe, fd, io_buffers[buf_idx], read_size, offset);
            sqe->user_data = buf_idx;
            offset += read_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    // Process remaining completions
    while (inflight > 0) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        
        if (cqe->res > 0) {
            int buf_idx = static_cast<int>(cqe->user_data);
            size_t bytes_read = cqe->res;
            
            if (!current_buffer->can_fit(bytes_read)) {
                if (current_buffer->used > 0) {
                    mark_buffer_ready(current_buffer);
                } else {
                    return_buffer(current_buffer);
                }
                current_buffer = get_available_buffer();
            }
            
            current_buffer->append(io_buffers[buf_idx], bytes_read);
            total_read += bytes_read;
        }
        
        io_uring_cqe_seen(&ring, cqe);
        inflight--;
    }

    // Send final buffer if it has data
    if (current_buffer->used > 0) {
        mark_buffer_ready(current_buffer);
    } else {
        return_buffer(current_buffer);
    }

    // Cleanup
    io_uring_queue_exit(&ring);
    for (void* buf : io_buffers) free(buf);
    close(fd);

    global_total_read.fetch_add(total_read);
    std::cout << "Worker " << worker_id << " completed: " << total_read / (1024*1024) << " MB\n";
}

void print_cuda_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    std::cout << "Using CUDA Device " << device_id << ": " << prop.name << "\n";
    std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "  Peak Memory Bandwidth: " 
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
              << " GB/s\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file> <cuda_device_id> [threads] [total_mb] [disk_block_kb] [gpu_chunk_mb]\n";
        std::cerr << "Example: " << argv[0] << " /dev/nvme0n1 0 8 1024 64 16\n";
        std::cerr << "  disk_block_kb: Size of each disk read (default: 1024 KB)\n";
        std::cerr << "  gpu_chunk_mb: Size of each GPU transfer (default: 16 MB)\n";
        return 1;
    }

    const char* filename = argv[1];
    int cuda_device_id = atoi(argv[2]);
    int num_threads = argc > 3 ? atoi(argv[3]) : 4;
    size_t total_size = argc > 4 ? atol(argv[4]) * 1024 * 1024 : DEFAULT_TOTAL_SIZE;
    size_t disk_block_size = argc > 5 ? atol(argv[5]) * 1024 : DEFAULT_DISK_BLOCK_SIZE;
    size_t gpu_chunk_size = argc > 6 ? atol(argv[6]) * 1024 * 1024 : DEFAULT_GPU_CHUNK_SIZE;

    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (cuda_device_id >= device_count) {
        std::cerr << "Error: CUDA device " << cuda_device_id << " not found. Available devices: 0-" << (device_count-1) << "\n";
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(cuda_device_id));
    print_cuda_device_info(cuda_device_id);

    // Initialize pinned memory buffers
    for (int i = 0; i < PINNED_BUFFER_COUNT; ++i) {
        auto buffer = std::make_unique<PinnedBuffer>(gpu_chunk_size);
        available_buffers.push(buffer.get());
        pinned_buffers.push_back(std::move(buffer));
    }

    // Allocate GPU memory
    void* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    
    // Create CUDA stream for async transfers
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout << "Reading " << total_size / (1024 * 1024) << " MB using " << num_threads
              << " threads\n";
    std::cout << "Disk block size: " << disk_block_size / 1024 << " KB\n";
    std::cout << "GPU chunk size: " << gpu_chunk_size / (1024 * 1024) << " MB\n";
    std::cout << "Pinned buffers: " << PINNED_BUFFER_COUNT << "\n";
    std::cout << "Target GPU device: " << cuda_device_id << "\n\n";

    size_t chunk_size = total_size / num_threads;

    struct timeval start, end, gpu_start;
    gettimeofday(&start, NULL);

    // Start GPU transfer worker
    std::thread gpu_worker(gpu_transfer_worker, d_buffer, total_size, stream);

    // Start IO workers
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        size_t offset = i * chunk_size;
        size_t worker_size = (i == num_threads - 1) ? 
                           total_size - offset : chunk_size;
        threads.emplace_back(io_uring_read_worker, filename, offset, worker_size, 
                           DEFAULT_QUEUE_DEPTH, disk_block_size, gpu_chunk_size, i);
    }

    // Wait for all IO threads to complete
    for (auto& t : threads) t.join();

    gettimeofday(&gpu_start, NULL);
    
    // Signal GPU worker that IO is done
    io_finished.store(true);
    pinned_cv.notify_all();
    
    // Wait for GPU transfers to complete
    gpu_worker.join();

    gettimeofday(&end, NULL);
    
    // Calculate timings
    double io_elapsed = (gpu_start.tv_sec - start.tv_sec) + (gpu_start.tv_usec - start.tv_usec) / 1e6;
    double total_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    double gpu_elapsed = (end.tv_sec - gpu_start.tv_sec) + (end.tv_usec - gpu_start.tv_usec) / 1e6;
    
    double mb_read = global_total_read.load() / 1.0e6;
    double mb_gpu = global_total_gpu_copied.load() / 1.0e6;

    std::cout << "\nPerformance Results:\n";
    std::cout << "==================\n";
    std::cout << "Disk Read: " << mb_read << " MB in " << io_elapsed << " sec → "
              << (mb_read / io_elapsed) << " MB/s\n";
    std::cout << "GPU Transfer: " << mb_gpu << " MB in " << gpu_elapsed << " sec → "
              << (mb_gpu / gpu_elapsed) << " MB/s\n";
    std::cout << "Total Time: " << total_elapsed << " sec → "
              << (mb_read / total_elapsed) << " MB/s overall\n";
    std::cout << "GPU Memory Used: " << total_size / (1024*1024) << " MB\n";
    std::cout << "Efficiency: " << (mb_gpu / mb_read) * 100 << "% data transferred to GPU\n";

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buffer));
    
    // Clear pinned buffers
    pinned_buffers.clear();
    
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}