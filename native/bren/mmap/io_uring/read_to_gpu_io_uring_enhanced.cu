#include <liburing.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>

#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(1); }} while (0)

#define READ_BLOCK_SIZE (1024 * 1024)  // 1 MB
#define IO_DEPTH 8
#define ALIGNMENT 1024 * 1024  // 1 MB alignment

struct TransferSlot {
    void* host_buf;
    cudaStream_t stream;
    off_t file_offset;
};

extern "C" void read_to_gpu_io_uring(const char* filepath, void* gpu_ptr, size_t total_size) {
    int fd = open(filepath, O_RDONLY | O_DIRECT| O_NONBLOCK);
    if (fd < 0) {
        perror("open");
        return;
    }

    io_uring ring;
    if (io_uring_queue_init(IO_DEPTH, &ring, 0) < 0) {
        std::cerr << "io_uring init failed" << std::endl;
        close(fd);
        return;
    }

    TransferSlot ctx[IO_DEPTH];
    for (int i = 0; i < IO_DEPTH; ++i) {
        if (posix_memalign(&ctx[i].host_buf, ALIGNMENT, READ_BLOCK_SIZE) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            return;
        }
        CUDA_CHECK(cudaStreamCreate(&ctx[i].stream));
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    size_t submitted = 0, completed = 0;
    const size_t num_blocks = total_size / READ_BLOCK_SIZE;

    auto enqueue = [&](size_t block_idx) {
        int slot = block_idx % IO_DEPTH;
        ctx[slot].file_offset = block_idx * READ_BLOCK_SIZE;
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, ctx[slot].host_buf, READ_BLOCK_SIZE, ctx[slot].file_offset);
        io_uring_sqe_set_data(sqe, &ctx[slot]);
    };

    for (; submitted < std::min(num_blocks, (size_t)IO_DEPTH); ++submitted)
        enqueue(submitted);
    io_uring_submit(&ring);

    while (completed < num_blocks) {
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;

        TransferSlot* entry = (TransferSlot*)io_uring_cqe_get_data(cqe);
        if (cqe->res < 0) {
            std::cerr << "Read failed: " << strerror(-cqe->res) << std::endl;
            break;
        }

        void* dst = static_cast<char*>(gpu_ptr) + entry->file_offset;
        CUDA_CHECK(cudaMemcpyAsync(dst, entry->host_buf, READ_BLOCK_SIZE,
                                   cudaMemcpyHostToDevice, entry->stream));

        io_uring_cqe_seen(&ring, cqe);
        ++completed;

        if (submitted < num_blocks) {
            enqueue(submitted++);
            io_uring_submit(&ring);
        }
    }

    for (int i = 0; i < IO_DEPTH; ++i) {
        cudaStreamSynchronize(ctx[i].stream);
        cudaStreamDestroy(ctx[i].stream);
        free(ctx[i].host_buf);
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("[IO_URING + CUDA] Read %.2f MB in %.2f sec -> %.2f MB/s\n",
           total_size / 1.0e6, elapsed, (total_size / 1.0e6) / elapsed);

    io_uring_queue_exit(&ring);
    close(fd);
}
