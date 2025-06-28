#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <iomanip>
#include <fstream>
#include <memory>
#include <algorithm>
#include <cmath>

// Kvikio headers
#include <kvikio/file_handle.hpp>
#include <kvikio/parallel_io.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/cufile.hpp>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>

// For plotting results
//#include <matplotlibcpp.h>
//namespace plt = matplotlibcpp;

struct GPUInfo {
    int device_id;
    std::string name;
    size_t memory_total;
    size_t memory_free;
    int pcie_gen;
    int pcie_width;
};

struct BenchmarkResult {
    int gpu_id;
    size_t transfer_size;
    double bandwidth_gbps;
    double latency_ms;
    double cpu_utilization;
};

class KVCacheBandwidthTester {
private:
    std::string test_file_path;
    std::vector<GPUInfo> gpu_info;
    std::vector<BenchmarkResult> results;
    size_t file_size;
    
    // Transfer sizes from 2^24 to 2^30 bytes
    std::vector<size_t> transfer_sizes = {
        1UL << 24,  // 16 MB
        1UL << 25,  // 32 MB
        1UL << 26,  // 64 MB
        1UL << 27,  // 128 MB
        1UL << 28,  // 256 MB
        1UL << 29,  // 512 MB
        1UL << 30   // 1 GB
    };

public:
    KVCacheBandwidthTester(const std::string& file_path) 
        : test_file_path(file_path) {
        initialize_cuda();
        detect_gpus();
        check_file_size();
        setup_kvikio();
    }

    void initialize_cuda() {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize CUDA: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Initialize cuFile for GPUDirect Storage
        if (cuFileDriverOpen() != CU_FILE_SUCCESS) {
            std::cerr << "Warning: cuFile driver failed to open. "
                      << "GPUDirect Storage may not be available." << std::endl;
        }
    }

    void detect_gpus() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        std::cout << "Detected " << device_count << " GPU(s):" << std::endl;
        
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            cudaMemGetInfo(&free_mem, &total_mem);
            
            GPUInfo info;
            info.device_id = i;
            info.name = prop.name;
            info.memory_total = total_mem;
            info.memory_free = free_mem;
            info.pcie_gen = prop.pciBusID;
            info.pcie_width = prop.pciDeviceID;
            
            gpu_info.push_back(info);
            
            std::cout << "  GPU " << i << ": " << info.name 
                      << " (" << total_mem / (1024*1024*1024) << " GB)" << std::endl;
        }
    }

    void check_file_size() {
        std::ifstream file(test_file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open test file: " + test_file_path);
        }
        
        file_size = file.tellg();
        file.close();
        
        std::cout << "Test file size: " << file_size / (1024*1024*1024) 
                  << " GB" << std::endl;
        
        if (file_size < (1UL << 30)) {
            throw std::runtime_error("Test file too small for largest transfer size");
        }
    }

    void setup_kvikio() {
        // Set optimal Kvikio settings for COMAT mode
        setenv("KVIKIO_BOUNCE_BUFFER_SIZE", "16777216", 1);  // 16MB bounce buffer
        setenv("KVIKIO_GDS_THRESHOLD", "1048576", 1);        // 1MB GDS threshold
        setenv("KVIKIO_THREAD_POOL_NTHREADS", "32", 1);      // 32 threads for Threadripper
        setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7", 1); // Support up to 8 GPUs
        
        std::cout << "Kvikio configured for COMAT mode optimization" << std::endl;
    }

    BenchmarkResult benchmark_transfer(int gpu_id, size_t transfer_size, 
                                     size_t num_iterations = 10) {
        cudaSetDevice(gpu_id);
        
        // Allocate GPU memory
        void* gpu_buffer;
        cudaError_t err = cudaMalloc(&gpu_buffer, transfer_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory: " + 
                                   std::string(cudaGetErrorString(err)));
        }

        // Create CUDA stream for async operations
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Open file with Kvikio
        auto file_handle = kvikio::FileHandle(test_file_path, "r");
        
        std::vector<double> iteration_times;
        iteration_times.reserve(num_iterations);

        // Warm-up iteration
        kvikio::pread(file_handle, gpu_buffer, transfer_size, 0);
        cudaStreamSynchronize(stream);

        for (size_t iter = 0; iter < num_iterations; ++iter) {
            // Calculate random offset to avoid cache effects
            size_t max_offset = file_size - transfer_size;
            size_t offset = (iter * 1024 * 1024) % max_offset;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Perform the transfer using Kvikio with GPUDirect Storage
            auto future = kvikio::pread_async(file_handle, gpu_buffer, 
                                            transfer_size, offset, stream);
            
            // Wait for completion
            cudaStreamSynchronize(stream);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                           (end_time - start_time);
            iteration_times.push_back(duration.count() / 1000.0); // Convert to ms
        }

        // Calculate statistics
        std::sort(iteration_times.begin(), iteration_times.end());
        double median_time_ms = iteration_times[num_iterations / 2];
        double bandwidth_gbps = (transfer_size / (1024.0 * 1024.0 * 1024.0)) / 
                               (median_time_ms / 1000.0);

        // Cleanup
        cudaFree(gpu_buffer);
        cudaStreamDestroy(stream);

        BenchmarkResult result;
        result.gpu_id = gpu_id;
        result.transfer_size = transfer_size;
        result.bandwidth_gbps = bandwidth_gbps;
        result.latency_ms = median_time_ms;
        result.cpu_utilization = 0.0; // TODO: Implement CPU utilization measurement

        return result;
    }

    void run_all_benchmarks() {
        std::cout << "\nRunning bandwidth benchmarks..." << std::endl;
        std::cout << std::setw(8) << "GPU ID" 
                  << std::setw(12) << "Size (MB)" 
                  << std::setw(15) << "Bandwidth (GB/s)"
                  << std::setw(15) << "Latency (ms)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        // Test each GPU with each transfer size
        for (const auto& gpu : gpu_info) {
            for (size_t transfer_size : transfer_sizes) {
                try {
                    auto result = benchmark_transfer(gpu.device_id, transfer_size);
                    results.push_back(result);
                    
                    std::cout << std::setw(8) << result.gpu_id
                              << std::setw(12) << std::fixed << std::setprecision(1)
                              << (transfer_size / (1024.0 * 1024.0))
                              << std::setw(15) << std::setprecision(2)
                              << result.bandwidth_gbps
                              << std::setw(15) << std::setprecision(3)
                              << result.latency_ms << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error testing GPU " << gpu.device_id 
                              << " with size " << transfer_size << ": " 
                              << e.what() << std::endl;
                }
            }
        }
    }

    void parallel_benchmark() {
        std::cout << "\nRunning parallel benchmark across all GPUs..." << std::endl;
        
        // Test all GPUs simultaneously with largest transfer size
        size_t parallel_transfer_size = 1UL << 29; // 512 MB per GPU
        
        std::vector<std::future<BenchmarkResult>> futures;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& gpu : gpu_info) {
            futures.push_back(std::async(std::launch::async, 
                [this, gpu, parallel_transfer_size]() {
                    return benchmark_transfer(gpu.device_id, parallel_transfer_size, 5);
                }));
        }
        
        double total_bandwidth = 0.0;
        for (auto& future : futures) {
            auto result = future.get();
            total_bandwidth += result.bandwidth_gbps;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                       (end_time - start_time);
        
        std::cout << "Parallel aggregate bandwidth: " << std::fixed 
                  << std::setprecision(2) << total_bandwidth 
                  << " GB/s across " << gpu_info.size() << " GPUs" << std::endl;
        std::cout << "Total test time: " << duration.count() << " ms" << std::endl;
    }
/*
    void generate_graph() {
        std::cout << "\nGenerating performance graphs..." << std::endl;
        
        // Prepare data for plotting
        std::map<int, std::vector<double>> gpu_transfer_sizes;
        std::map<int, std::vector<double>> gpu_bandwidths;
        
        for (const auto& result : results) {
            gpu_transfer_sizes[result.gpu_id].push_back(
                result.transfer_size / (1024.0 * 1024.0)); // Convert to MB
            gpu_bandwidths[result.gpu_id].push_back(result.bandwidth_gbps);
        }
        
        // Create bandwidth vs transfer size plot
        plt::figure_size(1200, 800);
        
        std::vector<std::string> colors = {"b-", "r-", "g-", "c-", "m-", "y-", "k-", "orange"};
        int color_idx = 0;
        
        for (const auto& gpu_data : gpu_bandwidths) {
            int gpu_id = gpu_data.first;
            std::string label = "GPU " + std::to_string(gpu_id);
            std::string color = colors[color_idx % colors.size()];
            
            plt::plot(gpu_transfer_sizes[gpu_id], gpu_data.second, color);
            plt::named_plot(label, gpu_transfer_sizes[gpu_id], gpu_data.second, color);
            color_idx++;
        }
        
        plt::xlabel("Transfer Size (MB)");
        plt::ylabel("Bandwidth (GB/s)");
        plt::title("KV Cache NVME to GPU Transfer Bandwidth");
        plt::legend();
        plt::grid(true);
        plt::xscale("log");
        
        // Save the plot
        plt::save("kv_cache_bandwidth_results.png");
        std::cout << "Graph saved as kv_cache_bandwidth_results.png" << std::endl;
        
        // Show the plot
        plt::show();
    }
*/
    void save_results_csv() {
        std::ofstream csv_file("kv_cache_benchmark_results.csv");
        csv_file << "GPU_ID,Transfer_Size_MB,Bandwidth_GBps,Latency_ms" << std::endl;
        
        for (const auto& result : results) {
            csv_file << result.gpu_id << ","
                     << (result.transfer_size / (1024.0 * 1024.0)) << ","
                     << result.bandwidth_gbps << ","
                     << result.latency_ms << std::endl;
        }
        
        csv_file.close();
        std::cout << "Results saved to kv_cache_benchmark_results.csv" << std::endl;
    }

    void print_system_info() {
        std::cout << "\n=== System Information ===" << std::endl;
        std::cout << "Target System: AMD Threadripper Pro WX3995" << std::endl;
        std::cout << "Memory: 512 GB RAM" << std::endl;
        std::cout << "Storage: 11-NVME PCIe4 RAID0 (1MB chunk size)" << std::endl;
        std::cout << "Test File: " << test_file_path << std::endl;
        std::cout << "Kvikio COMAT Mode: Enabled" << std::endl;
        std::cout << "GPUDirect Storage: " << 
                     (cuFileDriverIsOpen() ? "Available" : "Not Available") << std::endl;
    }

    void print_recommendations() {
        std::cout << "\n=== Performance Recommendations ===" << std::endl;
        
        // Find best performing configuration
        auto best_result = *std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.bandwidth_gbps < b.bandwidth_gbps;
            });
        
        std::cout << "Best single GPU performance: " << std::fixed << std::setprecision(2)
                  << best_result.bandwidth_gbps << " GB/s (GPU " << best_result.gpu_id
                  << ", " << (best_result.transfer_size / (1024*1024)) << " MB transfers)" << std::endl;
        
        // Calculate aggregate bandwidth
        std::map<size_t, double> size_to_total_bandwidth;
        for (const auto& result : results) {
            size_to_total_bandwidth[result.transfer_size] += result.bandwidth_gbps;
        }
        
        auto best_aggregate = *std::max_element(size_to_total_bandwidth.begin(),
                                              size_to_total_bandwidth.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
        
        std::cout << "Best aggregate bandwidth: " << std::fixed << std::setprecision(2)
                  << best_aggregate.second << " GB/s with "
                  << (best_aggregate.first / (1024*1024)) << " MB transfers" << std::endl;
        
        std::cout << "\nOptimization suggestions:" << std::endl;
        std::cout << "- Use " << (best_aggregate.first / (1024*1024)) 
                  << " MB transfer sizes for maximum aggregate throughput" << std::endl;
        std::cout << "- Distribute workload across all available GPUs" << std::endl;
        std::cout << "- Ensure RAID0 stripe size matches transfer patterns" << std::endl;
        std::cout << "- Consider using multiple concurrent streams per GPU" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        std::string test_file = "/mnt/kvcache/huge_test_4tb.bin";
        
        // Allow override of test file path
        if (argc > 1) {
            test_file = argv[1];
        }
        
        std::cout << "KV Cache NVME to GPU Bandwidth Tester" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        KVCacheBandwidthTester tester(test_file);
        
        tester.print_system_info();
        tester.run_all_benchmarks();
        tester.parallel_benchmark();
        //tester.generate_graph();
        tester.save_results_csv();
        tester.print_recommendations();
        
        std::cout << "\nBenchmark completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}