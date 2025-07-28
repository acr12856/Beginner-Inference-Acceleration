# PyTorch vs Torch-TensorRT Inference Profiling

## Overview

This project explores deep learning inference on NVIDIA GPUs (I used a Nvidia T4), comparing:

* **PyTorch FP32** (native PyTorch execution)
* **Torch-TensorRT FP32** (TensorRT-compiled PyTorch model, FP32 precision)
* **Torch-TensorRT FP16** (TensorRT-compiled PyTorch model, FP16 precision)

For each configuration we measure inference latency across various batch sizes and visualize hardware utilization and kernel execution breakdown using TensorBoard's PyTorch Profiler plugin.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (e.g. Tesla T4, V100, A100)
2. **CUDA Toolkit** (e.g. 11.7) and **cuDNN** installed and on your `PATH`
3. **Python 3.8+** (we use Python 3.10)
4. `virtualenv` or `conda` for creating isolated environments

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv ttrt-env
source ttrt-env/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install core dependencies
pip install \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 \
    torch-tensorrt \
    tensorboard tensorboard-plugin-profile \
    nvidia-pyindex tensorrt \
    numpy

# 4. Verify installations
python -c "import torch; print('PyTorch:', torch.__version__); import torch_tensorrt; print('Torch-TensorRT:', torch_tensorrt.__version__)"

```

> **Note**: You might need to adjust CUDA/cuDNN versions and `--extra-index-url` to match your system as they can throw issues with mismatched versioning.

---

## Project Workflow

1. **Model Preparation**: Load a pretrained model (e.g., ResNet50) in FP32 and (optionally) convert to FP16.
2. **Warm‑up Runs**: Run a few iterations to initialize CUDA contexts and compile kernels.
3. **Profiling Block**: Wrap inference calls in `torch.profiler.profile` with a TensorBoard trace handler.
4. **Batch Sizes**: Evaluate batch sizes `[1, 8, 16, 32]` for each of the three configurations.
5. **Trace Collection**: JSON traces are automatically written under `tb_logs/<run_name>/plugins/profile/`.
6. **TensorBoard Visualization**: Launch TensorBoard to inspect GPU utilization, kernel stats, and performance advice.

---

## Profiling Configuration

In your inference loop, use:

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule
torch.profiler.tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir),
    with_stack=True,
) as prof:
    for _ in range(total_steps):
        with record_function("inference_step"):  # custom span name
            _ = model(input_tensor)
        prof.step()
```

* **`wait=1`**: this skips the first step (eliminate overhead)
* **`warmup=1`**: one unrecorded step to warm caches (eliminate overhead)
* **`active=3`**: record three steps
* **`repeat=1`**: do one cycle of wait -> warmup -> active

---

## Running the Benchmarks

```bash
# Kill any existing TensorBoard
pkill -f tensorboard || true

# Launch TensorBoard, binding to all interfaces:
tensorboard \
  --logdir $(pwd)/tb_logs \
  --port 6006 \
  --bind_all \
  --load_fast=false
```
You can then run the jupyter notebook that is in the repo or what you have created!

After it finishes, traces will be in:

```
tb_logs/
├── pytorch_bs1/plugins/profile/...
├── pytorch_bs8/plugins/profile/...
└── ...
```
> **Note**: Traces created on my machine are currently in tb_logs to allow people to compare the results without needing to run the notebook on their machine. 
---

## Inspecting Results in TensorBoard

1. Open a browser at `http://localhost:6006`
2. Select the **PYTORCH\_PROFILER** tab
3. In the **Runs** dropdown, pick `pytorch_bs1`, `ttrt_fp32_bs1`, or `ttrt_fp16_bs1`
4. Under **Views** choose:

   * **Overview**: high‑level GPU Summary & Step Time Breakdown
   * **GPU Kernel**: per‑kernel durations, tensor core usage, occupancy
   * **DiffView**: compare two runs (baseline vs experiment)
5. In **Spans**, choose `1`, `2`, or `3` to inspect individual recorded steps, or pick **Aggregated** to see the average.

---

## Interpreting the Profiling Dashboards

### Overview

* **GPU Summary**: Utilization (%), SM efficiency, achieved occupancy.
* **Execution Summary**: breakdown of time in Kernel, CPU exec, Memcpy, etc.
* **Step Time Breakdown**: stacked bar of Kernel vs CPU vs Other per step.
* **Performance Recommendation**: hints (e.g., “GPU has low utilization → increase batch size”).

### GPU Kernel View

Shows all CUDA kernels sorted by total time:

* **Tensor Cores Used**: whether the kernel ran on tensor cores.
* **Calls**: number of launches.
* **Mean Duration**: per launch.
* **Estimated Achieved Occupancy**: how well warps filled the SM.

### DiffView

Compare two runs (e.g., FP32 vs FP16):

* **Execution Comparison**: side‑by‑side bar chart of average step time.
* **Diff/Accumulated Diff**: how much time was saved/spent on each node.
* **Trend Lines**: visualize scaling across recorded steps.

---
