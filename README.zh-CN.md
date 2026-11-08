[English](README.md) | [简体中文](README.zh-CN.md)

# BoolCAD

**BoolCAD：面向布尔运算的粒子重建与网格优化**

BoolCAD 通过对两个三角网格表面进行粒子采样、提取交线约束、将粒子投影到布尔结果表面，并使用 L-BFGS 优化粒子分布与法向一致性，重建两个网格之间的布尔运算结果。

## 项目地址

- 项目主页：
- 代码仓库：https://github.com/longwuya/BoolCAD
- 论文：
- 视频：

## 环境与依赖

- Windows x64 与支持 C++17 的编译器。
- CMake 3.20 或更新版本。
- vcpkg。
- CGAL、Eigen3、OpenMP。

仓库内包含 ANN、L-BFGS 和 PQP 等组件。当前构建配置使用 Windows 版 ANN 预编译库。

## 编译

以下命令在 PowerShell 中执行。请先安装 CMake、vcpkg 和 Visual Studio C++ 构建工具，并将 `VCPKG_ROOT` 环境变量设置为 vcpkg 所在目录。

```powershell
git clone https://github.com/longwuya/BoolCAD.git
cd BoolCAD

& "$env:VCPKG_ROOT\vcpkg.exe" install --triplet x64-windows

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -A x64
cmake --build build --config Release
```

当前 CMake 目标名称为 `BoolCAD`，生成的可执行文件位于 `build/bin/Release/BoolCAD.exe`，`ANN.dll` 会同时复制到该目录。

## 运行

```text
BoolCAD.exe <模型A> <模型B> [采样数] [运算类型] [FACTORY_DECAY] [FACTOR_FOR_PARTICLE_SYSTEM]
```

| 参数 | 含义 |
| --- | --- |
| `模型A` | 第一个输入模型路径，不带 `.obj` 或 `.off` 扩展名。 |
| `模型B` | 第二个输入模型路径，不带 `.obj` 或 `.off` 扩展名。 |
| `采样数` | 目标采样点数量，默认值为 `2000`。 |
| `运算类型` | 支持 `union`、`intersection`、`difference`，也支持 `1`、`2`、`3`。默认值为 `union`。 |
| `FACTORY_DECAY` | 粒子能量衰减系数，默认值为 `0.98`。 |
| `FACTOR_FOR_PARTICLE_SYSTEM` | 粒子核宽度系数，默认值为 `0.30`。 |

每个输入模型都需要同名的 `.obj` 和 `.off` 文件。

在仓库根目录运行示例：

```powershell
.\build\bin\Release\BoolCAD.exe `
  .\data\sample_ball `
  .\data\sample_block `
  2000 `
  union `
  0.98 `
  0.30
```

示例输入网格位于 `data/` 目录。程序只保存最终优化后的 `.xyz` 点云，输出到带时间戳的目录；运行过程中不生成中间文件，也不输出控制台日志。运行结束后会显示最终文件路径、实际点数和总运行时间。
