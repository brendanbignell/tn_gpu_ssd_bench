CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -arch compute_86 -code compute_86
LDFLAGS = -shared -luring -lcudart

TARGET = libenhanced_io.so
SOURCE = enhanced_read_to_gpu.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 simple_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1

.PHONY: all clean test