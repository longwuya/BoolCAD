[English](README.md) | [简体中文](README.zh-CN.md)

# Reconstruction_Ena

这是当前最终优化代码的整理版本，面向两个三角网格的布尔运算重建。程序先在输入表面上采样，提取网格交线并施加交线约束，再把粒子投影到布尔结果表面，最后通过 L-BFGS 优化粒子分布与法向一致性。

## 目录

```text
src/          核心重建和布尔运算代码
include/      项目头文件
utils/ann/    内置 ANN 头文件和 Windows 运行库
utils/lbfgs/  内置 L-BFGS 实现
utils/pqp/    内置 PQP 距离查询库
data/         快速验证用的小规模模型
```

## 环境

- Windows x64
- Visual Studio 2022，安装“使用 C++ 的桌面开发”
- CMake 3.20 或更新版本
- vcpkg
- CGAL、Eigen3、OpenMP

仓库内附带 Windows x64 版 ANN 预编译库。

## 编译

先将 `VCPKG_ROOT` 设置为包含 `vcpkg.exe` 的目录：

```powershell
$env:VCPKG_ROOT = "D:\vcpkg\vcpkg"

cmake -B build -S . -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release
```

生成文件位于 `build/bin/Release/Reconstruction_Ena.exe`，`ANN.dll` 会同时复制到该目录。

## 运行

两个模型参数都不带扩展名；每个模型需要同名的 `.obj` 和 `.off` 文件。

```text
Reconstruction_Ena.exe <模型A> <模型B> [采样数] [交线采样率] [交线权重] [运算类型]
```

在仓库根目录运行示例：

```powershell
.\build\bin\Release\Reconstruction_Ena.exe `
  .\data\sample_ball `
  .\data\sample_block `
  1000 `
  0.5 `
  6 `
  union
```

`OPERATION` 支持 `union`、`intersection`、`difference`，也支持 `1`、`2`、`3`。

程序会在工作目录写入交线诊断文件，并在带时间戳的目录中输出优化后的点云。

设置 `RECONSTRUCTION_PROFILE=1` 可以输出各阶段的耗时信息；设置 `RECONSTRUCTION_SILENT=1` 可以关闭每次迭代的控制台输出。

## 说明

- 当前代码依赖 Windows API 和预编译 ANN，仅支持 Windows x64。
- `ANN.dll` 必须与可执行文件位于同一目录。
- 大型实验数据、生成的网格、日志和性能测试快照不会上传到仓库。
