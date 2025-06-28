def kvikio_multi_gpu_benchmark(filespecs, batch_sizes, total_bytes, repeats):
    import cupy as cp
    from kvikio import CuFile
    np_dtype = np.float16
    cp_dtype = cp.float16
    dtype_size = np.dtype(np_dtype).itemsize
    results = []

    for gpu_id, (file_path, _) in enumerate(filespecs):
        cp.cuda.Device(gpu_id).use()
        batch_results = []
        for bidx, batch_size in enumerate(batch_sizes):
            n_batches = total_bytes // batch_size
            gds_bandwidths = []
            pipeline_times = []
            for run in range(repeats):
                print(f"GPU {gpu_id} {batch_size//1024} KB Run {run+1}")
                cp.cuda.Stream.null.synchronize()
                pipeline_start = time.time()
                with CuFile(file_path, "rb") as f:
                    for batch_idx in range(n_batches):
                        n_elements = batch_size // dtype_size
                        gpu_array = cp.empty((n_elements,), dtype=cp_dtype)
                        actual_bytes = gpu_array.nbytes
                        if batch_idx == 0 and run == 0 and bidx == 0:
                            print(f"[DEBUG] batch_size={batch_size}, n_elements={n_elements}, array_bytes={actual_bytes}, offset=0")
                        offset = batch_idx * batch_size
                        f.read(gpu_array, offset, batch_size)
                        cp.cuda.Stream.null.synchronize()
                        del gpu_array
                pipeline_end = time.time()
                pipeline_time = pipeline_end - pipeline_start
                gds_bandwidths.append(total_bytes / pipeline_time / 1e9)
                pipeline_times.append(pipeline_time)
            mean, minv, maxv, std = np.mean(gds_bandwidths), np.min(gds_bandwidths), np.max(gds_bandwidths), np.std(gds_bandwidths)
            batch_results.append({
                "batch_size": batch_size, "gds_bw_mean": mean, "gds_bw_min": minv, "gds_bw_max": maxv, "gds_bw_std": std,
                "pipeline_time_mean": np.mean(pipeline_times), "pipeline_time_min": np.min(pipeline_times),
                "pipeline_time_max": np.max(pipeline_times), "pipeline_time_std": np.std(pipeline_times),
                "n_batches": n_batches
            })
        results.append((gpu_id, batch_results))
    return results
