run FunDEM

env:
sudo apt-get install -y build-essential cmake ninja-build git
sudo apt-get install -y pkg-config
(no CUDA? sudo apt install -y nvidia-cuda-toolkit)

create your own code in FunDEM/apps

build:
cmake -S FunDEM -B build -G Ninja \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j

run:
./build/apps/...