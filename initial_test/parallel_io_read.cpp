#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <liburing.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <fstream>
#include <random>
#include <sstream>

size_t BLOCKSIZE = 1 << 20; // default 1 MiB
size_t TOTAL_TRANSFER_SIZE = 1024L * 1024 * 1024; // default 1 GiB
constexpr int QUEUE_DEPTH = 32;

std::mutex io_mutex;
size_t global_total_read = 0;

enum class Mode { IO_URING, MMAP, PREAD };

std::string mode_to_string(Mode mode) {
    switch (mode) {
        case Mode::IO_URING: return "io_uring";
        case Mode::MMAP: return "mmap";
        case Mode::PREAD: return "pread";
    }
    return "unknown";
}

std::vector<size_t> generate_random_offsets(size_t start, size_t end, size_t count) {
    std::vector<size_t> offsets;
    auto seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(start / BLOCKSIZE, (end - BLOCKSIZE) / BLOCKSIZE);
    for (size_t i = 0; i < count; ++i) {
        offsets.push_back(dist(rng) * BLOCKSIZE);
    }
    return offsets;
}

void io_uring_worker(const char* filename, std::vector<size_t> offsets) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return;
    }

    io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        close(fd);
        return;
    }

    std::vector<void*> buffers(QUEUE_DEPTH);
    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        if (posix_memalign(&buffers[i], 4096, BLOCKSIZE)) {
            perror("posix_memalign");
            return;
        }
    }

    size_t inflight = 0, submitted = 0, total_read = 0;
    while (submitted < QUEUE_DEPTH && submitted < offsets.size()) {
        auto sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], BLOCKSIZE, offsets[submitted]);
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    io_uring_cqe* cqe;
    while (total_read < offsets.size() * BLOCKSIZE) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        if (cqe->res < 0) break;
        total_read += cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        inflight--;

        if (submitted < offsets.size()) {
            int buf_idx = submitted % QUEUE_DEPTH;
            auto sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[buf_idx], BLOCKSIZE, offsets[submitted]);
            submitted++;
            inflight++;
            io_uring_submit(&ring);
        }
    }

    io_uring_queue_exit(&ring);
    for (void* buf : buffers) free(buf);
    close(fd);

    std::lock_guard<std::mutex> lock(io_mutex);
    global_total_read += total_read;
}

void mmap_worker(const char* filename, std::vector<size_t> offsets) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return;

    void* map = mmap(NULL, TOTAL_TRANSFER_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) return;

    madvise(map, TOTAL_TRANSFER_SIZE, MADV_RANDOM);

    // Use volatile pointer to ensure actual memory read
    volatile char* data = static_cast<volatile char*>(map);
    size_t total_read = 0;
    for (size_t off : offsets) {
        for (size_t i = 0; i < BLOCKSIZE; i += 4096) {
            volatile char c = data[off + i];
            (void)c;
        }
        total_read += BLOCKSIZE;
    }

    munmap((void*)map, TOTAL_TRANSFER_SIZE);
    close(fd);

    std::lock_guard<std::mutex> lock(io_mutex);
    global_total_read += total_read;
}

void pread_worker(const char* filename, std::vector<size_t> offsets) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return;

    std::vector<char> buffer(BLOCKSIZE);
    size_t total_read = 0;
    for (size_t off : offsets) {
        ssize_t n = pread(fd, buffer.data(), BLOCKSIZE, off);
        if (n > 0) total_read += n;
    }

    close(fd);
    std::lock_guard<std::mutex> lock(io_mutex);
    global_total_read += total_read;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file> <mode: io_uring|mmap|pread> [threads] [total_mb] [block_kb]\n";
        return 1;
    }

    const char* filename = argv[1];
    std::string mode_str = argv[2];
    int num_threads = argc > 3 ? atoi(argv[3]) : 4;
    if (argc > 4) TOTAL_TRANSFER_SIZE = static_cast<size_t>(atol(argv[4])) * 1024 * 1024;
    if (argc > 5) BLOCKSIZE = static_cast<size_t>(atol(argv[5])) * 1024;

    Mode mode = mode_str == "io_uring" ? Mode::IO_URING : mode_str == "mmap" ? Mode::MMAP : Mode::PREAD;

    std::vector<size_t> all_offsets = generate_random_offsets(0, TOTAL_TRANSFER_SIZE, TOTAL_TRANSFER_SIZE / BLOCKSIZE);
    size_t chunk = all_offsets.size() / num_threads;

    std::vector<std::thread> threads;
    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < num_threads; ++i) {
        auto begin = all_offsets.begin() + i * chunk;
        auto end_it = (i == num_threads - 1) ? all_offsets.end() : begin + chunk;
        std::vector<size_t> chunk_offsets(begin, end_it);
        if (mode == Mode::IO_URING)
            threads.emplace_back(io_uring_worker, filename, chunk_offsets);
        else if (mode == Mode::MMAP)
            threads.emplace_back(mmap_worker, filename, chunk_offsets);
        else
            threads.emplace_back(pread_worker, filename, chunk_offsets);
    }

    for (auto& t : threads) t.join();

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    double mb = global_total_read / 1e6;

    std::ofstream csv("read_benchmark_results.csv", std::ios::app);
    csv << mode_to_string(mode) << "," << num_threads << "," << mb / elapsed << "\n";
    csv.close();

    std::cout << mode_to_string(mode) << " with " << num_threads << " threads: "
              << mb << " MB read in " << elapsed << " sec -> " << (mb / elapsed) << " MB/s\n";

    return 0;
}
