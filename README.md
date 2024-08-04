HowTo: Build OpenCV World
=========================
Perform CMake configured high speed build of OpenCV_World DLL with performance and contrib options on Windows platform.

System configuration:
=====================

Example given for the following configuration:

	OS: Windows 11
	Source Directory: c:\src
	CPU: Intel i9 12900K
	GPU: nVidiaa RTX 3090
 	Visual Studio 2022 Community

CMake configuration:
====================
```
General configuration for OpenCV 4.10.0-dev

  CPU/HW features:
    Baseline:                    SSE SSE2 SSE3 SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
      requested:                 AVX2
    Dispatched code generation:
      requested:                 AVX FMA3 AVX2

    Extra dependencies:          cudart_static.lib nppc.lib nppial.lib nppicc.lib nppidei.lib nppif.lib nppig.lib
								nppim.lib nppist.lib nppisu.lib nppitc.lib npps.lib cublas.lib cudnn.lib cufft.lib

  OpenCV modules:
    To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudacodec
								cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo
								cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann
								fuzzy gapi hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor
								mcc ml objdetect optflow phase_unwrapping photo plot quality rapid reg rgbd saliency
								shape signal stereo stitching structured_light superres surface_matching text tracking
								ts video videoio videostab wechat_qrcode world xfeatures2d ximgproc xobjdetect xphoto

    Unavailable:                 cannops cvv freetype hdf java julia matlab ovis python2 python3 sfm viz
    Applications:                tests perf_tests apps
    Documentation:               NO
    Non-free algorithms:         YES

  GUI:
    Win32 UI:                    YES
    GTK+:                        NO
    OpenGL support:              YES (opengl32 glu32)
    VTK support:                 NO

  Media I/O:
    ZLib:                        build (ver 1.3.1)
    JPEG:                        build-libjpeg-turbo (ver 3.0.3-70)
      SIMD Support Request:      YES
      SIMD Support:              NO
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         build (ver 1.6.43)
      SIMD Support Request:      YES
      SIMD Support:              YES (Intel SSE)
    TIFF:                        build (ver 42 - 4.6.0)
    JPEG 2000:                   build (ver 2.5.0)
    OpenEXR:                     build (ver 2.3.0)
    HDR:                         YES
    SUNRASTER:                   YES
    PXM:                         YES
    PFM:                         YES

  Video I/O:
    DC1394:                      NO
    FFMPEG:                      YES (prebuilt binaries)
      avcodec:                   YES (58.134.100)
      avformat:                  YES (58.76.100)
      avutil:                    YES (56.70.100)
      swscale:                   YES (5.9.100)
      avresample:                YES (4.0.0)
    GStreamer:                   YES (1.24.5)
    DirectShow:                  YES
    Media Foundation:            YES
      DXVA:                      YES
    Intel Media SDK:             NO

  Parallel framework:            OpenMP

  Trace:                         YES (with Intel ITT)

  Other third-party libraries:
    Intel IPP:                   2021.12.0 [2021.12.0]
           at:                   C:/src/OpenCvBuild/Debug/3rdparty/ippicv/ippicv_win/icv
    Intel IPP IW:                sources (2021.12.0)
              at:                C:/src/OpenCvBuild/Debug/3rdparty/ippicv/ippicv_win/iw
    Lapack:                      YES (C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/lib/mkl_intel_lp64.lib C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/lib/mkl_sequential.lib C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2/lib/mkl_core.lib)
    Eigen:                       YES (ver 3.4.90)
    Custom HAL:                  NO
    Protobuf:                    build (3.19.1)
    Flatbuffers:                 builtin/3rdparty (23.5.9)

  NVIDIA CUDA:                   YES (ver 12.5, CUFFT CUBLAS FAST_MATH)
    NVIDIA GPU arch:             86
    NVIDIA PTX archs:

  cuDNN:                         YES (ver 9.2.1)

  OpenCL:                        YES (NVD3D11)
    Include path:                C:/src/opencv/3rdparty/include/opencl/1.2
    Link libraries:              Dynamic load

  Python (for build):            NO

  Java:
    ant:                         NO
    Java:                        NO
    JNI:                         NO
    Java wrappers:               NO
    Java tests:                  NO
```

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

```
- cd \src
- git clone https://github.com/opencv/opencv_contrib.git
```

nVidia Windows Driver - GeForce Game Ready Driver
[https://www.nvidia.com/en-us/drivers/](https://www.nvidia.com/en-us/drivers/)

	- Default install

nVidia CUDA - nVidia CUDA Library
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

	- Download version 12.5
 	- Default install

cuDNN - nVidia Deep Neural Network Library
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

	- Download version 9.3.0
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

	- Add to System Path:

		C:\gstreamer\1.0\msvc_x86_64\bin

	- Reopen command shell and execute the following (Possible long startup delay on first invocation during plugin scan)

		gst-launch-1.0 videotestsrc ! videoconvert ! autovideosink
  
Get CPU and GPU Architecture Configurations:
============================================

nVidia Computer Capability Tables:
[https://developer.nvidia.com/cuda-gpus#compute](https://developer.nvidia.com/cuda-gpus#compute)

	- Lookup GPU capability version in the table
	- set GPU_CAPABILITY below to this version

Processor Instructions:

	- Query a LLM for OpenCV CMake CPU_BASELINE and CPU_DISPATCH supported for the target processor model
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
