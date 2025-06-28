CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -Xcompiler -pthread
LDFLAGS = -shared -luring -lcudart -lpthread

TARGET = libarray_aware.so
SOURCE = array_aware_read.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 array_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1

quick: $(TARGET)
	python3 array_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 2.0 1

.PHONY: all clean test quick