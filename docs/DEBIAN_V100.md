# Building for a V100 on Debian

Eclair uses the proprietary NVIDIA driver for Volta. The recommended starting
point is the R580 branch and CUDA Toolkit 12.9.2. Confirm support for your card
in the driver release notes and verify that it appears in `nvidia-smi`.
CUDA 13 removed offline compilation for Volta.

Install the driver and toolkit separately. Follow
[NVIDIA's Debian driver guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/debian.html)
for your distribution and keep the headers for the running kernel installed
when using DKMS. If Secure Boot is enabled, follow the distribution's module
signing procedure.

On the configuration used for validation, CUDA 12.9 headers conflicted with
glibc 2.41 on Debian 13. Build with a supported compiler and userspace,
such as Debian 12 with GCC 12. The resulting application can run against a
compatible driver on the host.

```sh
nvidia-smi
nvcc --version
nvcc --list-gpu-arch | grep -x compute_70

cmake -S . -B build-v100 -DECLAIR_ENABLE_CUDA=ON \
  -DECLAIR_CUDA_ARCHITECTURES=70 -DECLAIR_BUILD_TESTS=ON
cmake --build build-v100
ctest --test-dir build-v100 --output-on-failure
build-v100/eclair_device_info
```

Record the driver, toolkit, compiler, flags and device when reporting results.
See the [CUDA 12.9.2 release notes](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html).
