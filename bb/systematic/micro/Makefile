# Makefile for micro-optimizations
CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
LDFLAGS = -shared -luring -lcudart

TARGET = libmicro_opt.so
SOURCE = micro_optimizations.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

quick: $(TARGET)
	python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1 --quick

test: $(TARGET)
	python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1

.PHONY: all clean quick test