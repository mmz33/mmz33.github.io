---
layout: post
title: CUDA Programming Notes
excerpt: GPU can be much faster than CPU for some computations such as matrix multiplications. They are intended to be fast for such computations. I find it very important to learn more how to utilize such a hardware to do advanced stuff and more optimizations. Especially when it comes to deep learning! Thus, this blog contains my notes about CUDA programming which I collected during my learning process.
---

GPU can be much faster than CPU for some computations
such as matrix multiplications.
They are intended to be fast for such computations.
I find it very important to learn more how to utilize such a hardware to
do advanced stuff and more optimizations.
Especially when it comes to deep learning! Thus, this blog
contains my notes about CUDA programming which I collected while learning.
For reference, you can check this [book](https://developer.nvidia.com/cuda-example).
You can also find practice code with comments [here](https://github.com/mmz33/practice-cuda).

{% include mathjax.html %}

### Functions

- `__global__`: called from host/device and run only on device.
- `__device__`: called from device and run only on device.
- `__host__`  : called from host and run only on host.

### Variables

- `__device__`: declares device variable in global memory for all threads and lifetime
of application.
- `__shared__`: declares device variable in shared memory between threads within a block
with lifetime of block.
- `__constant__`: declares device variable in constant memory for all threads and lifetime
of application.

### Types

- dim3: basically 3D vector of type `uint3` (3D unsigned integer vector). Dimension
which is not specified has 1 as default value.

### Memory Managemnet

- `cudaMalloc((void**)&pointer, size)`: allocates memory on device.
- `cudaFree`: frees allocated memory on device.
- `cudaMemcpy(dest_ptr, src_ptr, size, direction)`: synchronous copy between host and device.
- `cudaMemcpyHostToDevice`: used as copy direction from host to device (e.g in `cudaMemcpy`).
- `cudaMemcpyDeviceToHost`: used as copy direction from device to host (e.g in `cudaMemcpy`).
- `cudaMemcpyToSymbol(dest_ptr, src_ptr, size, direction)`: copy to a global or
constant memory.
- `cudaHostAlloc`: allocates page-locked host memory.
- `cudaFreeHost`: frees page-locked host memory.

### Thread Management

- `__syncthreads()`: wait until all threads reach this sync.

### Event Management

- `cudaEventCreate(cudaEvent_t *event)`: creates CUDA event object.
- `cudaEventRecord(cudaEvent_t *event, cudaStream_t stream)`: records an event
on given CUDA stream (stream used for concurrency management).
- `cudaEventSynchronize(cudaEvent_t *event)`: synchronizes between GPU and CPU on
the given event.

### Blocks and threads

- Kernel signature: `kernel_name<<<NUM_BLOCKS, NUM_THREADS>>>(params)`
- A device kernel can run N number of blocks each having M number of threads.
This can be imagined as a 2D grid where the number of rows is the number
of blocks and the number of columns is the number of threads per block. Then,
you can access threads linearly similar to how it is done for flattened arrays.
So each block would take a copy of the code and run it in parallel.
- There are many reasons why we have this combination between blocks and threads.
One of them is due to hardware limitations. There is a limited number of blocks
that can be created. In addition, one of the important benefits from that
is to do thread collaboration with shared memory and synchronization. For example,
this would allows us to do efficient reduction functions.
- Related keywords:
  - `threadIdx.x`: thread index
  - `blockDim.x`: number of threads
  - `blockIdx.x`: block index
  - `gridDim.x`: number of blocks

### Constant Memory

- Kinds of memories: global, shared, *constant*, etc
- Constant memory can be used to store data that won't change during the
kernel execution. This will improve memory bandwidth
- No need to allocated or free constant memory explicitly
- It reduces memory bandwidth by caching consecutive reads of the same address.
Thus, this will save read access for nearby threads or threads in the same warp.
E.g if many threads read from same address, one thread can read the data and
broadcast it to all other threads.
- Reading from constant memory could be slower than global memory in case many
threads request reads for different addresses
- Constant memory can be useful for an important application called "Ray Tracing."
Briefly, the idea is to output 3D objects in a 2D image taking into consideration
light, shades, objects material, etc. Constant memory can be used to cache
the objects in the environment which make if fast for threads to access them

### Atomics

- The execution of *atomic* operations can not be divided into smaller
parts by other threads.
- Some function names: atomicAdd, atomicMin, atomicSub, etc
(see more [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)).
- They are used to avoid race condition between multiple threads.
A simple case is incrementing a shared variable. This is also known as
*read-modify-write* operation. Each thread would need to read the variable,
modify it, and then write the new result back. If the threads are not
scheduled in a correct way, the last result might be wrong. Thus,
here atomic operations would be used to avoid such an issue.


### Page-locked host memory

- Typically, we use the C library `malloc` function to allocate host
memory. This will allocate pageable memory, i.e it is possible to be paged
out to the disk (or swapped to the disk).
- However, in some cases, it is preferable to allocate host memory
that is guaranteed by the OS that it will not be paged out to the disk.
In this way, we can always access the physical address of the memory.
- This will speed up the copy between host and device.
- Note that allocating too much page-locked memories can lead to
memory overflow issues not only for the current running application
but also for other applications.
- Allocating such kind of memory can be done via `cudaHostAlloc`.
