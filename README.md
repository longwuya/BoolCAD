[English](README.md) | [简体中文](README.zh-CN.md)

# BoolCAD

**BoolCAD: Boolean-Aware Particle Reconstruction and Mesh Optimization**

BoolCAD reconstructs the result of Boolean operations on two triangle meshes by sampling surface particles, extracting intersection constraints, projecting particles onto the resulting surface, and minimizing a combined distribution and normal-consistency energy with L-BFGS.

## Project Links

- Project homepage:
- Code repository: https://github.com/longwuya/BoolCAD
- Paper:
- Video:

## Requirements and Dependencies

- Windows x64 and a compiler with C++17 support.
- CMake 3.20 or later.
- vcpkg.
- CGAL, Eigen3, and OpenMP.

The repository includes ANN, L-BFGS, and PQP components. The current build configuration uses a precompiled ANN library for Windows.

## Build

Run the following commands in PowerShell. Install CMake, vcpkg, and the Visual Studio C++ build tools first, and set the `VCPKG_ROOT` environment variable to the vcpkg installation directory.

```powershell
git clone https://github.com/longwuya/BoolCAD.git
cd BoolCAD

& "$env:VCPKG_ROOT\vcpkg.exe" install --triplet x64-windows

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -A x64
cmake --build build --config Release
```

The CMake target is named `BoolCAD`. The executable is written to `build/bin/Release/BoolCAD.exe`, with `ANN.dll` copied beside it.

## Run

```text
BoolCAD.exe <MODEL_A> <MODEL_B> [SAMPLES] [OPERATION] [FACTORY_DECAY] [FACTOR_FOR_PARTICLE_SYSTEM]
```

| Parameter | Description |
| --- | --- |
| `MODEL_A` | First input path, without the `.obj` or `.off` extension. |
| `MODEL_B` | Second input path, without the `.obj` or `.off` extension. |
| `SAMPLES` | Target number of sample points. Default: `2000`. |
| `OPERATION` | `union`, `intersection`, or `difference`; `1`, `2`, and `3` are also accepted. Default: `union`. |
| `FACTORY_DECAY` | Particle-energy decay coefficient. Default: `0.98`. |
| `FACTOR_FOR_PARTICLE_SYSTEM` | Particle kernel width factor. Default: `0.30`. |

Each input model must have matching `.obj` and `.off` files.

Run this example from the repository root:

```powershell
.\build\bin\Release\BoolCAD.exe `
  .\data\sample_A `
  .\data\sample_B `
  2000 `
  union `
  0.98 `
  0.30
```

Example input meshes are available in `data/`. Only the final optimized `.xyz` point cloud is saved, in a timestamped output directory. The program produces no intermediate files or console logs during execution; when finished, it prints the output path, actual point count, and total runtime.
