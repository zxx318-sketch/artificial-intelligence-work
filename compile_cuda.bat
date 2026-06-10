@echo off
REM Windows CUDA compilation script for minitorch
REM Requirements: NVIDIA GPU + CUDA Toolkit (nvcc in PATH)

echo [Compile] Creating output directory...
if not exist minitorch\cuda_kernels mkdir minitorch\cuda_kernels

echo [Compile] Building CUDA kernels...
nvcc -shared -o minitorch\cuda_kernels\combine.dll src\combine.cu -Xlinker /DEF:src\combine.def

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed!
    echo [HINT] Make sure 'nvcc' is in your PATH. 
    echo   If you installed CUDA Toolkit, run this script from "x64 Native Tools Command Prompt"
    echo   or activate the CUDA environment first.
    exit /b 1
)

echo [SUCCESS] Built: minitorch\cuda_kernels\combine.dll
echo [NEXT] You can now run training with --device cuda
