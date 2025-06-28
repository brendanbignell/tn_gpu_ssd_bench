CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
LDFLAGS = -shared -luring -lcudart

TARGET = liboptimized_io.so
SOURCE = optimized_read_to_gpu.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

quick: $(TARGET)
	python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 2.0 1 --quick

test: $(TARGET)
	python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1

.PHONY: all clean quick test

