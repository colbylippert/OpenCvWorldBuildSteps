Build OpenCV World:
===================
Perform CMake configured high speed build of OpenCV_World DLL with performance and contrib options on Windows platform:

Install Prerequisites:
======================

Git - Version Control
[https://git-scm.com/download/win](https://git-scm.com/download/win)

	- Default install

CMake - Build System
[https://cmake.org/download/](https://cmake.org/download/)

	- Default install

Ninja - High Performance Compiler
[https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases)

	- Extract ninja.exe to c:\src\ninja\ninja.exe

OpenCv - Computer Vision Library

```
- cd \src
- git clone https://github.com/opencv/opencv.git
```

OpenCv Contrib - Computer Vision Contrib Modules Library 
git clone https://github.com/opencv/opencv_contrib.git

```
- cd \src
- git clone https://github.com/opencv/opencv_contrib.git
```

nVidia Windows Driver - GeForce Game Ready Driver
[https://www.nvidia.com/en-us/drivers/](https://www.nvidia.com/en-us/drivers/)

	- Default install

nVidia CUDA - nVidia CUDA Library
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

	- Default install

cuDNN - nVidia Deep Neural Network Library
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

	- Default install

Intel Math Kernel Library - High Performance Math Library
[https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

	- Online installer
	- Default install

Eigen - High Performance Linear Algebra, Matrix and Vectory Library

```
- cd \src
- git clone https://gitlab.com/libeigen/eigen.git
```

GStreamer - Multimedia Framework Library
[https://gstreamer.freedesktop.org/download/#windows](https://gstreamer.freedesktop.org/download/#windows)

	- Install MSVC Runtime Installer
		. Select Complete install option

	- Install MSVC Development Installer
		. Select Complete install option

Get CPU and GPU Architecture Configurations:
============================================

nVidia Computer Capability Tables:
[https://developer.nvidia.com/cuda-gpus#compute](https://developer.nvidia.com/cuda-gpus#compute)

	- Lookup GPU capability version in the table
	- set GPU_CAPABILITY below to this version

Processor Instructions:

	- Query an LLM for OpenCV CMake CPU_BASELINE and CPU_DISPATCH supported for the target processor model
	- set CPU_BASELINE below to supported instruction sets
	- set CPU_DISPATCH below to supported instruction sets

Build OpenCv (Debug):
=====================

 - Open command shell
 - Set Visual Studio env variables:

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

 - Set build specific environment variables:

```
set "CUDNN_LIBRARY_DIR=C:\Program Files\NVIDIA\CUDNN\v9.2\lib\12.5\x64\cudnn.lib"
set "CUDNN_INCLUDE_DIR=C:\Program Files\NVIDIA\CUDNN\v9.2\include\12.5"
set "CUDA_TOOLKIT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
set "GPU_CAPABILITY=8.6"
set "CPU_BASELINE=AVX2"
set "CPU_DISPATCH=AVX,FMA3,AVX2"
set "MKL_DIR=C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2"
set "EIGEN_DIR=C:/src/eigen"
set "NINJA_DIR=C:\src\ninja"
set path=%NINJA_DIR%;%path%
set "OPENCV_DIR=C:\src\opencv"
set "OPENCV_MODULES_DIR=C:\src\opencv_contrib\modules"
set "GENERATOR=Ninja"
set "GSTREAMER_ROOT=C:\gstreamer\1.0\msvc_x86_64"
set path=%GSTREAMER_ROOT%\bin;%GSTREAMER_ROOT%\lib;%path%
set "OPENCV_BUILD_DIR=C:\src\OpenCvBuild\Debug"
set "BUILD_TYPE=Debug"
```

 - Execute CMake:

```
"C:\Program Files\CMake\bin\cmake.exe" ^
-B"%OPENCV_BUILD_DIR%/" ^
-H"%OPENCV_DIR%/" ^
-G"%GENERATOR%" ^
-DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
-DOPENCV_EXTRA_MODULES_PATH="%OPENCV_MODULES_DIR%/" ^
-DBUILD_opencv_world=ON ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DCPU_BASELINE="%CPU_BASELINE%" ^
-DCPU_DISPATCH="%CPU_DISPATCH%" ^
-DENABLE_FAST_MATH=ON ^
-DWITH_OPENMP=ON ^
-DWITH_TBB=ON ^
-DWITH_EIGEN=ON ^
-DEIGEN_INCLUDE_PATH="%EIGEN_DIR%" ^
-DWITH_LAPACK=ON ^
-DLAPACK_IMPL=MKL ^
-DMKL_ROOT_DIR="%MKL_DIR%" ^
-DWITH_CUDA=ON ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_DIR%" ^
-DCUDA_FAST_MATH=ON ^
-DCUDA_ARCH_BIN="%GPU_CAPABILITY%" ^
-DWITH_CUBLAS=ON ^
-DWITH_CUVID=ON ^
-DWITH_NVCUVID=ON ^
-DWITH_NVENC=ON ^
-DHAVE_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DCUDNN_LIBRARY="%CUDNN_LIBRARY_DIR%" ^
-DCUDNN_INCLUDE_DIR="%CUDNN_INCLUDE_DIR%" ^
-DBUILD_opencv_cudacodec=ON ^
-DWITH_FFMPEG=ON ^
-DWITH_GSTREAMER=ON ^
-DGSTREAMER_ROOT_DIR="%GSTREAMER_ROOT%" ^
-DGSTREAMER_LIB_DIR="%GSTREAMER_ROOT%\lib" ^
-DGSTREAMER_INCLUDE_DIR="%GSTREAMER_ROOT%\include" ^
-DGSTREAMER_VERSION=1.0 ^
-DWITH_OPENGL=ON ^
-DWITH_MFX=ON ^
-DWITH_GTK=ON
```

 - Execute buld

```
"C:\Program Files\CMake\bin\cmake.exe" --build %OPENCV_BUILD_DIR% --target install
```

 - Build artifacts are located at:

	C:\src\OpenCvBuild\Debug

 Build OpenCv (Release):
=====================

 - Open command shell
 - Set Visual Studio env variables:

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

 - Set build specific environment variables:

```
set "CUDNN_LIBRARY_DIR=C:\Program Files\NVIDIA\CUDNN\v9.2\lib\12.5\x64\cudnn.lib"
set "CUDNN_INCLUDE_DIR=C:\Program Files\NVIDIA\CUDNN\v9.2\include\12.5"
set "CUDA_TOOLKIT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
set "GPU_CAPABILITY=8.6"
set "CPU_BASELINE=AVX2"
set "CPU_DISPATCH=AVX,FMA3,AVX2"
set "MKL_DIR=C:\Program Files (x86)\Intel\oneAPI\mkl\2024.2"
set "EIGEN_DIR=C:/src/eigen"
set "NINJA_DIR=C:\src\ninja"
set path=%NINJA_DIR%;%path%
set "OPENCV_DIR=C:\src\opencv"
set "OPENCV_MODULES_DIR=C:\src\opencv_contrib\modules"
set "GENERATOR=Ninja"
set "GSTREAMER_ROOT=C:\gstreamer\1.0\msvc_x86_64"
set path=%GSTREAMER_ROOT%\bin;%GSTREAMER_ROOT%\lib;%path%
set "OPENCV_BUILD_DIR=C:\src\OpenCvBuild\Release"
set "BUILD_TYPE=Release"
```

 - Execute CMake:

```
"C:\Program Files\CMake\bin\cmake.exe" ^
-B"%OPENCV_BUILD_DIR%/" ^
-H"%OPENCV_DIR%/" ^
-G"%GENERATOR%" ^
-DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
-DOPENCV_EXTRA_MODULES_PATH="%OPENCV_MODULES_DIR%/" ^
-DBUILD_opencv_world=ON ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DCPU_BASELINE="%CPU_BASELINE%" ^
-DCPU_DISPATCH="%CPU_DISPATCH%" ^
-DENABLE_FAST_MATH=ON ^
-DWITH_OPENMP=ON ^
-DWITH_TBB=ON ^
-DWITH_EIGEN=ON ^
-DEIGEN_INCLUDE_PATH="%EIGEN_DIR%" ^
-DWITH_LAPACK=ON ^
-DLAPACK_IMPL=MKL ^
-DMKL_ROOT_DIR="%MKL_DIR%" ^
-DWITH_CUDA=ON ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_DIR%" ^
-DCUDA_FAST_MATH=ON ^
-DCUDA_ARCH_BIN="%GPU_CAPABILITY%" ^
-DWITH_CUBLAS=ON ^
-DWITH_CUVID=ON ^
-DWITH_NVCUVID=ON ^
-DWITH_NVENC=ON ^
-DHAVE_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DCUDNN_LIBRARY="%CUDNN_LIBRARY_DIR%" ^
-DCUDNN_INCLUDE_DIR="%CUDNN_INCLUDE_DIR%" ^
-DBUILD_opencv_cudacodec=ON ^
-DWITH_FFMPEG=ON ^
-DWITH_GSTREAMER=ON ^
-DGSTREAMER_ROOT_DIR="%GSTREAMER_ROOT%" ^
-DGSTREAMER_LIB_DIR="%GSTREAMER_ROOT%\lib" ^
-DGSTREAMER_INCLUDE_DIR="%GSTREAMER_ROOT%\include" ^
-DGSTREAMER_VERSION=1.0 ^
-DWITH_OPENGL=ON ^
-DWITH_MFX=ON ^
-DWITH_GTK=ON
```

 - Execute buld

```
"C:\Program Files\CMake\bin\cmake.exe" --build %OPENCV_BUILD_DIR% --target install
```

 - Build artifacts are located at:

	C:\src\OpenCvBuild\Release
