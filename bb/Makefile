CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
LDFLAGS = -shared -luring -lcudart

TARGET = libultra_fast_io.so
SOURCE = ultra_fast_read_to_gpu.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 ultra_fast_test.py /mnt/kvcache/huge_test_4tb.bin --size 5.0

sweep: $(TARGET)
	python3 ultra_fast_test.py /mnt/kvcache/huge_test_4tb.bin --size 5.0 --sweep

.PHONY: all clean test sweep