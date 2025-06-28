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

constexpr size_t DEFAULT_BLOCK_SIZE = 1 << 20;      // 1 MiB
constexpr size_t DEFAULT_TOTAL_SIZE = 10L * 1024 * 1024 * 1024; // 10 GiB
constexpr int DEFAULT_QUEUE_DEPTH = 32;

std::mutex io_mutex;
size_t global_total_read = 0;

void io_uring_read_worker(const char* filename, size_t start_offset, size_t size_to_read, int queue_depth, size_t block_size) {
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

    std::vector<void*> buffers(queue_depth);
    for (int i = 0; i < queue_depth; ++i) {
        if (posix_memalign(&buffers[i], 4096, block_size)) {
            perror("posix_memalign");
            return;
        }
    }

    size_t offset = start_offset;
    size_t total_read = 0;
    int inflight = 0;
    int submitted = 0;

    // Fill submission queuep
    while (submitted < queue_depth && total_read + submitted * block_size < size_to_read) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], block_size, offset);
        offset += block_size;
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

        total_read += cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        inflight--;

        if (offset < start_offset + size_to_read) {
            int buf_idx = submitted % queue_depth;
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[buf_idx], block_size, offset);
            offset += block_size;
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    while (inflight > 0) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        total_read += cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        inflight--;
    }

    io_uring_queue_exit(&ring);
    for (void* buf : buffers) free(buf);
    close(fd);

    std::lock_guard<std::mutex> lock(io_mutex);
    global_total_read += total_read;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file> [threads] [total_mb] [block_kb]\n";
        return 1;
    }

    const char* filename = argv[1];
    int num_threads = argc > 2 ? atoi(argv[2]) : 4;
    size_t total_size = argc > 3 ? atol(argv[3]) * 1024 * 1024 : DEFAULT_TOTAL_SIZE;
    size_t block_size = argc > 4 ? atol(argv[4]) * 1024 : DEFAULT_BLOCK_SIZE;

    std::cout << "Reading " << total_size / (1024 * 1024) << " MB using " << num_threads
              << " threads with " << block_size / 1024 << " KiB blocks.\n";

    size_t chunk_size = total_size / num_threads;

    std::vector<std::thread> threads;
    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; ++i) {
        size_t offset = i * chunk_size;
        threads.emplace_back(io_uring_read_worker, filename, offset, chunk_size, DEFAULT_QUEUE_DEPTH, block_size);
    }

    for (auto& t : threads) t.join();

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    double mb = global_total_read / 1.0e6;

    std::cout << "Total read: " << mb << " MB in " << elapsed << " sec → "
              << (mb / elapsed) << " MB/s\n";

    return 0;
}
