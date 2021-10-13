#!/bin/bash
source ~/.bashrc

cd ~/Workspace/CenterForecast/det3d/ops/dcn
rm -rf build
rm deform_conv_cuda.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace

cd ~/Workspace/CenterForecast/det3d/ops/iou3d_nms
rm -rf build
rm iou3d_nms_cuda.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace

cd ~/Workspace/Core/spconv
rm -rf build
rm -rf dist
rm -rf spconv.egg-info
python setup.py bdist_wheel

cd ~/Workspace/Core/spconv/dist
pip install * --force-reinstall

#cd ~/Workspace/Core/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
