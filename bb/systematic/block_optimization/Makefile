# Makefile for block size optimization
CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
LDFLAGS = -shared -luring -lcudart

TARGET = libblock_size_opt.so
SOURCE = block_size_optimization.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 block_size_optimization_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1

quick: $(TARGET)
	python3 block_size_optimization_test.py /mnt/kvcache/huge_test_4tb.bin 2.0 1

.PHONY: all clean test quick