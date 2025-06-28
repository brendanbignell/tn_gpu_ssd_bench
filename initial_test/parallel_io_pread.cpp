#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <atomic>

constexpr size_t MIN_BLOCKSIZE = 1024 * 1024;  // 1MB
constexpr size_t MAX_BLOCKSIZE = 256 * 1024 * 1024;  // 256MB

std::atomic<size_t> global_total_read = 0;

size_t get_file_size(const char* filename) {
    int fd = open(filename, O_RDONLY | O_NONBLOCK);
    if (fd < 0) throw std::runtime_error("Unable to open file to get size");
    off_t size = lseek(fd, 0, SEEK_END);
    close(fd);
    return size;
}

std::vector<size_t> generate_random_offsets(size_t file_size, size_t total_read_size, size_t blocksize) {
    size_t count = total_read_size / blocksize;
    if (file_size < blocksize) throw std::runtime_error("File size smaller than block size");

    std::vector<size_t> offsets;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, (file_size - blocksize) / blocksize);

    for (size_t i = 0; i < count; ++i)
        offsets.push_back(dist(rng) * blocksize);

    return offsets;
}

std::vector<std::vector<size_t>> split_offsets(const std::vector<size_t>& offsets, size_t threads) {
    std::vector<std::vector<size_t>> result(threads);
    for (size_t i = 0; i < offsets.size(); ++i)
        result[i % threads].push_back(offsets[i]);
    return result;
}

void pread_worker(const char* filename, std::vector<size_t> offsets, size_t blocksize) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return;

    std::unique_ptr<char[]> buffer(new (std::nothrow) char[blocksize]);
    if (!buffer) return;

    size_t total_read = 0;
    for (size_t offset : offsets) {
        ssize_t n = pread(fd, buffer.get(), blocksize, offset);
        if (n < 0) break;
        total_read += n;
    }

    close(fd);
    global_total_read += total_read;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <file> <threads> <max_total_read_size_in_GB>\n";
        return 1;
    }

    const char* filename = argv[1];
    size_t threads = atoi(argv[2]);
    size_t total_read_gb = atoi(argv[3]);
    size_t total_read_size = total_read_gb * 1024UL * 1024UL * 1024UL;

    size_t file_size = get_file_size(filename);

    for (size_t blocksize = MIN_BLOCKSIZE; blocksize <= MAX_BLOCKSIZE; blocksize *= 2) {
        global_total_read = 0;

        auto offsets = generate_random_offsets(file_size, total_read_size, blocksize);
        auto thread_offsets = split_offsets(offsets, threads);

        std::vector<std::thread> workers;
        struct timeval start, end;
        gettimeofday(&start, nullptr);

        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back(pread_worker, filename, thread_offsets[i], blocksize);

        for (auto& worker : workers) worker.join();

        gettimeofday(&end, nullptr);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        double throughput = global_total_read / (1024.0 * 1024.0) / elapsed;

        std::cout << "Threads: " << threads << ", Blocksize: " << blocksize / 1024
                  << " KB, Throughput: " << throughput << " MB/s\n";
    }

    return 0;
}
