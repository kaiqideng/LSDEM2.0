# FunDEM Build Guide (Ubuntu / Windows + WSL + CUDA)

This README shows a minimal workflow to install CUDA, build FunDEM with CMake + Ninja, run an example, and visualize the result.

⸻

## 1. Install the NVIDIA driver

### Ubuntu

```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Windows + WSL

Install the NVIDIA driver on Windows.

---

Check after installation (in Ubuntu or WSL):

```bash
nvidia-smi
```

Make sure `nvidia-smi` works before installing CUDA.

⸻

## 2. Install CUDA Toolkit

### Ubuntu

First, choose the correct CUDA repository name based on your Ubuntu version:

* Ubuntu 20.04 → ubuntu2004
* Ubuntu 22.04 → ubuntu2204
* Ubuntu 24.04 → ubuntu2404

Then run:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntuXXXX/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-4
```

Replace `ubuntuXXXX` with your actual Ubuntu version, and replace `12-4` with your desired CUDA version if needed.

---

### Windows + WSL

Use the WSL-specific CUDA repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-4
```

---

### Configure environment variables

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

Verify the installation:

```bash
nvcc --version
nvidia-smi
```

⸻

## 3. Install build tools

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build
```

Check the installed tools:

```bash
cmake --version
ninja --version
```

⸻

## 4. Check your GPU architecture

You should set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.

Check your GPU model:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Common examples:

* RTX 20 series → 75
* RTX 30 series → 86
* RTX 40 series → 89
* A100 → 80
* H100 → 90

If you are not sure, use:

```cmake
CMAKE_CUDA_ARCHITECTURES=native
```

CMake will detect the correct architecture automatically.

⸻

## 5. Configure and build FunDEM

Run from the project root (where `FunDEM/` is located):

```bash
cmake -S FunDEM -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Then build:

```bash
cmake --build build -j
```

Notes:

* Adjust `CMAKE_CUDA_ARCHITECTURES` to match your GPU
* Do not mix apt installation with `.run` installation for CUDA

⸻

## 6. Run an example

```bash
./build/apps/<your_app_name>
```

For example:

```bash
./build/apps/tutorial__tutorial2-LSParticleRolling
```

⸻

## 7. Visualize the result

### Install ParaView

```bash
sudo apt update
sudo apt install -y paraview
```

Or download the latest version directly from the official website for better compatibility

---

### Open the output in ParaView

1. Launch ParaView
2. Go to **File → Open**
3. Navigate to the output directory and select a `.vtu` file (or `.pvd` if available)
4. Click **Apply** in the Properties panel to load the data
5. Use the toolbar to select display variables (e.g. velocity, particleID) and adjust the colormap

⸻

## 8. Create your own application

Create your own code in:

```
FunDEM/apps
```

You can use the existing tutorials as templates for your own applications.