# Makefile for simple parallel versions using OpenMP
CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -Xcompiler -fopenmp
LDFLAGS = -shared -luring -lcudart -lgomp

TARGET = libsimple_parallel.so
SOURCE = simple_parallel_read.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 simple_parallel_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1

quick: $(TARGET)
	python3 simple_parallel_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 2.0 1

.PHONY: all clean test quick