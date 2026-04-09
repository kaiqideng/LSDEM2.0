FunDEM Build Guide (Ubuntu + CUDA)

This README shows a minimal workflow to install CUDA, build FunDEM with CMake + Ninja, run an example, and visualize the result.

⸻

1. Install the NVIDIA driver

sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot

Check after reboot:

nvidia-smi

Make sure nvidia-smi works before installing CUDA.

⸻

2. Install CUDA Toolkit

First, choose the correct CUDA repository name based on your Ubuntu version:
	•	Ubuntu 20.04 -> ubuntu2004
	•	Ubuntu 22.04 -> ubuntu2204
	•	Ubuntu 24.04 -> ubuntu2404

Then run:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntuXXXX/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit

Replace ubuntuXXXX with your actual Ubuntu version.

Configure environment variables:

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

Verify the installation:

nvcc --version
nvidia-smi


⸻

3. Install build tools

sudo apt update
sudo apt install -y build-essential cmake ninja-build

Check the installed tools:

cmake --version
ninja --version
nvcc --version


⸻

4. Check your GPU architecture

You should set CMAKE_CUDA_ARCHITECTURES to match your GPU.

Check your GPU model:

nvidia-smi

Or query it directly:

nvidia-smi --query-gpu=name,driver_version --format=csv

Common examples:
	•	RTX 20 series -> 75
	•	RTX 30 series -> 86
	•	RTX 40 series -> 89
	•	A100 -> 80
	•	H100 -> 90

If you are not sure, check the GPU model first and then choose the matching architecture value.

⸻

5. Configure and build FunDEM

cmake -S FunDEM -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build -j

Notes:
	•	Adjust CMAKE_CUDA_ARCHITECTURES=75 to match your GPU.
	•	Do not mix apt installation with .run installation for CUDA.

⸻

6. Run an example

./build/apps/tutorial__tutorial2-LSParticleRolling


⸻

7. Visualize the result

Use ParaView for visualization.

⸻

8. Create your own application

Create your own code in:

FunDEM/apps

You can use the existing tutorials as templates for your own applications.