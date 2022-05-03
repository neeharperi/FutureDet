#!/bin/bash
source ~/.bashrc

pip install -r requirements.txt

conda install -c anaconda cmake
conda install mkl-service
#conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

export CUDA_HOME=/usr/local/cuda-10.1
export CUDA_ROOT=/usr/local/cuda-10.1

cd ~/Workspace/FutureDet/det3d/ops/dcn
rm -rf build
rm deform_conv_cuda.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace

cd ~/Workspace/FutureDet/det3d/ops/iou3d_nms
rm -rf build
rm iou3d_nms_cuda.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace

export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda

cd ~/Workspace/Core/spconv
rm -rf build
rm -rf dist
rm -rf spconv.egg-info
python setup.py bdist_wheel

cd ~/Workspace/Core/spconv/dist
pip install * --force-reinstall

export CUDA_HOME=/usr/local/cuda-10.1
export CUDA_ROOT=/usr/local/cuda-10.1

cd ~/Workspace/Core/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

