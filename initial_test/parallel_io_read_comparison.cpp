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
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <atomic>

size_t REPEAT = 1;
constexpr int QUEUE_DEPTH = 32;

std::mutex io_mutex;
std::atomic<size_t> global_total_read = 0;

enum class Mode { IO_URING, MMAP, PREAD };

std::string mode_to_string(Mode mode) {
    switch (mode) {
        case Mode::IO_URING: return "io_uring";
        case Mode::MMAP: return "mmap";
        case Mode::PREAD: return "pread";
    }
    return "unknown";
}

std::vector<size_t> generate_random_offsets(size_t start, size_t end, size_t count, size_t blocksize) {
    if (end < start + blocksize) {
        throw std::runtime_error("Invalid range: end is too small for given blocksize");
    }
    std::vector<size_t> offsets;
    auto seed = std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(start / blocksize, (end - blocksize) / blocksize);
    for (size_t i = 0; i < count; ++i) {
        offsets.push_back(dist(rng) * blocksize);
    }
    return offsets;
}

std::vector<std::vector<size_t>> split_offsets(const std::vector<size_t>& offsets, size_t num_threads) {
    std::vector<std::vector<size_t>> split(num_threads);
    for (size_t i = 0; i < offsets.size(); ++i) {
        split[i % num_threads].push_back(offsets[i]);
    }
    return split;
}

void io_uring_worker(const char* filename, std::vector<size_t> offsets, size_t blocksize) {
    try {
        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed");

        io_uring ring;
        if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0)) throw std::runtime_error("io_uring init failed");

        std::vector<void*> buffers(QUEUE_DEPTH);
        for (int i = 0; i < QUEUE_DEPTH; ++i) {
            if (posix_memalign(&buffers[i], 4096, blocksize) || !buffers[i]) {
                io_uring_queue_exit(&ring);
                close(fd);
                throw std::runtime_error("posix_memalign failed or returned nullptr");
            }
        }

        size_t inflight = 0, submitted = 0, total_read = 0;
        while (submitted < QUEUE_DEPTH && submitted < offsets.size()) {
            auto sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, fd, buffers[submitted], blocksize, offsets[submitted]);
            submitted++;
            inflight++;
        }
        io_uring_submit(&ring);

        io_uring_cqe* cqe;
        while (total_read < offsets.size() * blocksize && inflight > 0) {
            if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
            if (cqe->res < 0) {
                std::cerr << "Async read failed: " << strerror(-cqe->res) << "\n";
                break;
            }
            total_read += cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            inflight--;

            if (submitted < offsets.size()) {
                int buf_idx = submitted % QUEUE_DEPTH;
                auto sqe = io_uring_get_sqe(&ring);
                if (!sqe) break;
                io_uring_prep_read(sqe, fd, buffers[buf_idx], blocksize, offsets[submitted]);
                submitted++;
                inflight++;
                io_uring_submit(&ring);
            }
        }

        io_uring_queue_exit(&ring);
        for (void* buf : buffers) if (buf) free(buf);
        close(fd);

        std::lock_guard<std::mutex> lock(io_mutex);
        global_total_read += total_read;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Exception in io_uring_worker: " << e.what() << std::endl;
    }
}

void mmap_worker(const char* filename, std::vector<size_t> offsets, size_t blocksize) {
    try {
        int fd = open(filename, O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed");

        off_t file_size = lseek(fd, 0, SEEK_END);
        size_t map_size = *std::max_element(offsets.begin(), offsets.end()) + blocksize;
        if (map_size > (size_t)file_size) throw std::runtime_error("Requested mmap range exceeds file size");

        void* map = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) throw std::runtime_error("mmap failed");

        madvise(map, map_size, MADV_SEQUENTIAL);

        size_t total_read = 0;
        for (size_t offset : offsets) {
            char* src = static_cast<char*>((char*)map + offset);
            for (size_t i = 0; i < blocksize; i += 4096) {
                total_read += src[i];
            }
        }

        munmap(map, map_size);
        close(fd);

        std::lock_guard<std::mutex> lock(io_mutex);
        global_total_read += total_read;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Exception in mmap_worker: " << e.what() << std::endl;
    }
}

void pread_worker(const char* filename, std::vector<size_t> offsets, size_t blocksize) {
    try {
        int fd = open(filename, O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed");

        std::unique_ptr<char[]> buffer(new (std::nothrow) char[blocksize]);
        if (!buffer) throw std::runtime_error("Failed to allocate read buffer");

        size_t total_read = 0;
        for (size_t offset : offsets) {
            ssize_t bytes = pread(fd, buffer.get(), blocksize, offset);
            if (bytes < 0) throw std::runtime_error("pread failed");
            total_read += bytes;
        }

        close(fd);
        std::lock_guard<std::mutex> lock(io_mutex);
        global_total_read += total_read;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Exception in pread_worker: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file> <repeat>\n";
        return 1;
    }

    const char* filename = argv[1];
    REPEAT = atoi(argv[2]);

    std::vector<Mode> modes = { Mode::IO_URING, Mode::MMAP, Mode::PREAD };

    //std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};
    std::vector<int> thread_counts = {2};

    
    std::vector<size_t> blocksizes = {
        1L << 20,  // 1 MiB
        2L << 20,  // 2 MiB
        4L << 20,  // 4 MiB
        8L << 20,  // 8 MiB
        16L << 20, // 16 MiB
        32L << 20,
        64L << 20,
        128L << 20,
        256L << 20,
        512L << 20,
        1024L << 20  // 1 GiB
    };

    size_t total_transfer = 1024L * 1024 * 1024; // 1 GiB default

    std::ofstream csv("read_benchmark_results.csv", std::ios::app);
    csv << "mode,threads,block_kb,total_mb,run,elapsed_sec,throughput_MBps\n";

    for (size_t blocksize : blocksizes) {
        for (int num_threads : thread_counts) {
            for (Mode mode : modes) {
                double total_throughput = 0.0, min_throughput = 1e12, max_throughput = 0.0;
                std::vector<double> throughputs;

                for (size_t r = 1; r <= REPEAT; ++r) {
                    global_total_read = 0;
                    auto all_offsets = generate_random_offsets(0, total_transfer, total_transfer / blocksize, blocksize);
                    size_t chunk = all_offsets.size() / num_threads;

                    std::vector<std::thread> threads;
                    struct timeval start, end;
                    gettimeofday(&start, NULL);

                    for (int i = 0; i < num_threads; ++i) {
                        auto begin = all_offsets.begin() + i * chunk;
                        auto end_it = (i == num_threads - 1) ? all_offsets.end() : begin + chunk;
                        std::vector<size_t> chunk_offsets(begin, end_it);

                        if (mode == Mode::IO_URING)
                            threads.emplace_back([=] { io_uring_worker(filename, chunk_offsets, blocksize); });
                        else if (mode == Mode::MMAP)
                            threads.emplace_back([=] { mmap_worker(filename, chunk_offsets, blocksize); });
                        else
                            threads.emplace_back([=] { pread_worker(filename, chunk_offsets, blocksize); });
                    }

                    for (auto& t : threads) t.join();

                    gettimeofday(&end, NULL);
                    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
                    double mb = global_total_read / 1e6;
                    double throughput = mb / elapsed;

                    csv << mode_to_string(mode) << "," << num_threads << "," << (blocksize / 1024) << ","
                        << (total_transfer / (1024 * 1024)) << "," << r << ","
                        << std::fixed << std::setprecision(3) << elapsed << "," << throughput << "\n";

                    total_throughput += throughput;
                    min_throughput = std::min(min_throughput, throughput);
                    max_throughput = std::max(max_throughput, throughput);
                    throughputs.push_back(throughput);
                }

                double avg = total_throughput / REPEAT;
                double sum_sq_diff = 0.0;
                for (double v : throughputs) sum_sq_diff += (v - avg) * (v - avg);
                double stddev = std::sqrt(sum_sq_diff / REPEAT);

                std::cout << mode_to_string(mode) << " summary (" << REPEAT << " runs):\n"
                          << "  Threads: " << num_threads << ", Block Size: " << blocksize / 1024 << " KB\n"
                          << "  Transfer: " << total_transfer / (1024 * 1024) << " MB\n"
                          << "  Min: " << min_throughput << " MB/s, Max: " << max_throughput
                          << " MB/s, Avg: " << avg << " MB/s, Stddev: " << stddev << " MB/s\n\n";
            }
        }
        std::cout << "Completed benchmarks for block size: " << (blocksize / (1024 * 1024)) << " MiB\n";
        std::flush(std::cout);
    }

    std::cout << "Results will be saved to read_benchmark_results.csv\n";
    std::flush(std::cout);

    csv.close();
    return 0;
}
