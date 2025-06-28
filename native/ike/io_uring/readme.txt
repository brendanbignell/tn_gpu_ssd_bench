nvcc -Xcompiler -fPIC -shared read_to_gpu_io_uring.cu -o libread_io_uring.so -luring

dd if=/dev/urandom of=largefile.bin bs=1M count=1024

