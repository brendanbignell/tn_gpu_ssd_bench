#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <liburing.h>

#define QUEUE_DEPTH 256
#define MY_BLOCK_SIZE (1 *1024 * 1024)            // 1 MiB
#define TOTAL_SIZE (10 * 1024L * 1024L * 1024L)      // 1 GiB

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file_path>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0)) {
        perror("io_uring_queue_init");
        return 1;
    }

    void *buffers[QUEUE_DEPTH];
    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        if (posix_memalign(&buffers[i], 4096, MY_BLOCK_SIZE)) {
            perror("posix_memalign");
            return 1;
        }
    }

    struct io_uring_cqe *cqe;
    size_t total_read = 0;
    off_t offset = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int inflight = 0;
    int submitted = 0;

    while (submitted < QUEUE_DEPTH && total_read + submitted * MY_BLOCK_SIZE < TOTAL_SIZE) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, buffers[submitted], MY_BLOCK_SIZE, offset);
        offset += MY_BLOCK_SIZE;
        submitted++;
        inflight++;
    }
    io_uring_submit(&ring);

    while (total_read < TOTAL_SIZE) {
        if (io_uring_wait_cqe(&ring, &cqe) < 0) {
            perror("io_uring_wait_cqe");
            break;
        }

        if (cqe->res < 0) {
            fprintf(stderr, "Async read failed: %s\n", strerror(-cqe->res));
            break;
        }

        total_read += cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        inflight--;

        if (offset < TOTAL_SIZE) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            int buffer_index = submitted % QUEUE_DEPTH;
            io_uring_prep_read(sqe, fd, buffers[buffer_index], MY_BLOCK_SIZE, offset);
            offset += MY_BLOCK_SIZE;
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

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("Read %.2f MB in %.2f seconds → %.2f MB/s\n",
           total_read / 1.0e6, elapsed, (total_read / 1.0e6) / elapsed);

    io_uring_queue_exit(&ring);
    for (int i = 0; i < QUEUE_DEPTH; ++i) {
        free(buffers[i]);
    }
    close(fd);
    return 0;
}