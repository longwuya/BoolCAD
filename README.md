# Reconstruction_Ena

C++17 implementation of Boolean-aware particle reconstruction. The optimizer
samples the input surfaces, computes intersection constraints, projects
particles onto the resulting surfaces, and minimizes a combined particle
spacing and normal-consistency energy with L-BFGS.

## Repository layout

```text
src/          Core reconstruction and Boolean-operation code
include/      Project headers
utils/ann/    Vendored ANN headers and Windows runtime
utils/lbfgs/  Vendored L-BFGS implementation
utils/pqp/    Vendored PQP proximity-query library
data/         Small input pair for a smoke test
```

## Requirements

- Windows x64
- Visual Studio 2022 with the C++ desktop workload
- CMake 3.20 or newer
- vcpkg
- CGAL, Eigen3 and OpenMP

The source distribution includes a precompiled ANN library for Windows x64.

## Build

Set `VCPKG_ROOT` to the directory containing `vcpkg.exe`:

```powershell
$env:VCPKG_ROOT = "D:\vcpkg\vcpkg"

cmake -B build -S . -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release
```

The executable is written to `build/bin/Release/Reconstruction_Ena.exe`, with
`ANN.dll` copied beside it.

## Run

The two model arguments are paths without an extension. Each model must have
matching `.obj` and `.off` files.

```text
Reconstruction_Ena.exe <MODEL_A> <MODEL_B> [SAMPLES] [RATE] [SIGMA] [OPERATION]
```

- `MODEL_A`, `MODEL_B`: input paths without `.obj` or `.off`.
- `SAMPLES`: target particle count; default `1000`.
- `RATE`: intersection sampling rate; default `0.0`.
- `SIGMA`: intersection-region weight factor; default `6.0`.
- `OPERATION`: `union`, `intersection`, or `difference` (also accepted as
  `1`, `2`, or `3`); default `difference`.

Run the bundled example from the repository root:

```powershell
.\build\bin\Release\Reconstruction_Ena.exe `
  .\data\sample_ball `
  .\data\sample_block `
  1000 `
  0.5 `
  6 `
  union
```

The program writes intersection diagnostics in the working directory and the
optimized point cloud in a timestamped result directory.

Set `RECONSTRUCTION_PROFILE=1` to print phase-level timing information. Set
`RECONSTRUCTION_SILENT=1` to suppress per-iteration console output.

## Notes

- The current implementation targets Windows x64 because it uses the Windows
  API and a precompiled ANN library.
- `ANN.dll` must remain next to the executable at runtime.
- Large experimental data, generated meshes, logs, and benchmark snapshots are
  intentionally excluded from this repository.
