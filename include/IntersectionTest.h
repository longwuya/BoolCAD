#include "Distance_OBB.h"

#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>

#include <iostream>
#include <fstream>
#include <time.h>
#include<sstream>
#include <vector>
#include <map>

namespace PMP = CGAL::Polygon_mesh_processing;

class intersection_test
{
protected:
    // 使用精确断言但非精确构造的核，平衡精度和性能
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

    // 定义几何体类型
    typedef Kernel::Point_3 Point_3;
    typedef Kernel::Triangle_3 Triangle_3;
    typedef Kernel::Segment_3 Segment_3;

    // 定义多面体类型
    typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

    // 多面体迭代器类型
    typedef Polyhedron::Facet_iterator                   Facet_iterator;
    typedef Polyhedron::Halfedge_around_facet_circulator Halfedge_facet_circulator;
    typedef Polyhedron::Vertex_iterator                  Vertex_iterator;

    // 定义表面网格类型
    typedef CGAL::Surface_mesh<Kernel::Point_3>          Mesh;
    typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;
    typedef boost::graph_traits<Mesh>::edge_descriptor     edge_descriptor;
    typedef boost::graph_traits<Mesh>::face_descriptor     face_descriptor;

    // 存储交线线段和孤立交点
    std::vector<std::vector<Point_3>> intersection_segments;
    std::vector<Point_3> intersection_points;

    // 输入模型和交集结果
    Polyhedron p1, p2, intersection;

public:
    /*****************************************************************
    * 计算三维空间两点间欧氏距离
    * 参数：两个点的坐标(x1,y1,z1)和(x2,y2,z2)
    * 返回：两点间的直线距离
    *****************************************************************/
    double dis(double x1, double y1, double z1, double x2, double y2, double z2)
    {
        return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    }

    /*****************************************************************
    * 计算交线总长度（从文件中读取交线点序列）
    * 流程：
    * 1. 读取intersection_point.xyz文件中的点坐标
    * 2. 按顺序计算相邻点距离并累加
    * 返回：交线总长度
    * 注意：文件需为每行三个浮点数的格式
    *****************************************************************/
    double length()
    {
        double len = 0;
        std::string line;
        double x_temp, y_temp, z_temp, x_temp2, y_temp2, z_temp2;
        std::ifstream f1;
        f1.open("intersection_point.xyz");
        if (!f1) { std::cerr << "deal intersection_point file read error1! test.cpp" << std::endl; exit(0); }

        // 读取第一个点
        getline(f1, line);
        std::istringstream iss(line);
        iss >> x_temp >> y_temp >> z_temp;

        // 逐点计算距离
        while (getline(f1, line))
        {
            std::istringstream iss(line);
            if (iss >> x_temp2 >> y_temp2 >> z_temp2) {
                len += dis(x_temp, y_temp, z_temp, x_temp2, y_temp2, z_temp2);
                x_temp = x_temp2; y_temp = y_temp2; z_temp = z_temp2;
            }
            else {
                std::cerr << "deal intersection_point file read error2! " << std::endl;
            }
        }
        f1.close();
        return len;
    }

    /*****************************************************************
    * 计算交线相邻点间最小距离
    * 流程：类似length()，但记录最小值
    * 返回：相邻点间最小距离
    * 注意：初始min设为较大值，可能不适用于所有场景
    *****************************************************************/
    double length_min()
    {
        double min = 100000;
        double temp;
        std::string line;
        double x_temp, y_temp, z_temp, x_temp2, y_temp2, z_temp2;
        std::ifstream f1;
        f1.open("Intersection/intersection_point.xyz");
        if (!f1) { std::cerr << "deal intersection_point file read error1! test112" << std::endl; exit(0); }

        // 读取第一个点
        getline(f1, line);
        std::istringstream iss(line);
        iss >> x_temp >> y_temp >> z_temp;

        // 逐点计算并更新最小值
        while (getline(f1, line))
        {
            std::istringstream iss(line);
            if (iss >> x_temp2 >> y_temp2 >> z_temp2) {
                temp = dis(x_temp, y_temp, z_temp, x_temp2, y_temp2, z_temp2);    
                min = temp;
                x_temp = x_temp2; y_temp = y_temp2; z_temp = z_temp2;
            }
            else {
                std::cerr << "deal intersection_point file read error2! " << std::endl;
            }
        }
        f1.close();
        return min;
    }

    // 将所有交线折线输出到一个 .xyz 文件中
    void export_intersection_to_xyz(
        const std::vector<std::vector<Point_3>>& intersection_segments,
        const std::string& out_filename)
    {
        std::ofstream ofs(out_filename);
        if (!ofs) {
            std::cerr << "无法创建输出文件: " << out_filename << std::endl;
            return;
        }

        // 遍历每条折线
        for (size_t seg_idx = 0; seg_idx < intersection_segments.size(); ++seg_idx) {
            const auto& poly = intersection_segments[seg_idx];
            if (poly.empty()) continue;

            // 可选：在每段折线前加一条注释，标记折线序号
            ofs << "# segment " << seg_idx << "\n";

            // 输出该折线上的所有点
            for (const auto& P : poly) {
                ofs
                    << P.x() << " "
                    << P.y() << " "
                    << P.z() << "\n";
            }

            // 段与段之间空一行分隔
            ofs << "\n";
        }

        ofs.close();
        std::cout << "已将所有交线折线写入: " << out_filename << std::endl;
    }

    static void export_polylines_to_obj(const std::vector<std::vector<Point_3>>& polylines, const std::string& filename) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "无法创建OBJ文件: " << filename << std::endl;
            return;
        }

        int vertex_index = 1; // OBJ顶点索引从1开始
        std::vector<std::vector<int>> line_indices; // 存储每条折线的顶点索引序列

        // 1. 输出所有顶点 (v x y z)
        for (const auto& poly : polylines) {
            std::vector<int> indices;
            for (const auto& point : poly) {
                out << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
                indices.push_back(vertex_index++);
            }
            line_indices.push_back(indices);
        }

        // 2. 输出线段连接关系 (l i1 i2)
        out << "g IntersectionLines\n";
        for (const auto& indices : line_indices) {
            if (indices.size() < 2) continue; // 忽略单点折线
            // 将折线上的点依次连接成线段
            for (size_t i = 0; i < indices.size() - 1; ++i) {
                out << "l " << indices[i] << " " << indices[i + 1] << "\n";
            }
        }

        out.close();
        std::cout << "折线已输出至OBJ文件: " << filename << std::endl;
    }


    /*****************************************************************
    * 返回交线列表intersection_segments
    *****************************************************************/
    std::vector<std::vector<Point_3>> get_intersection_segments(){
        return intersection_segments;
    };


    /*****************************************************************
    * 辅助函数：将面描述符转换为字符串
    * 用途：处理CGAL面描述符的字符串格式（如"f12"）
    *****************************************************************/
    std::string face_descriptor_transto_string(face_descriptor t)
    {
        std::string temp;
        std::stringstream ss;
        ss << t;
        ss >> temp;
        return temp;
    }

    /*****************************************************************
    * 辅助函数：从面描述符字符串提取整数索引
    * 示例：输入"f123"返回123
    *****************************************************************/
    int trans(std::string s)
    {
        std::string trimmed_str = s.substr(1, s.length() - 1);
        return std::stoi(trimmed_str);
    }

    /*****************************************************************
    * 主流程函数：计算两模型的交线
    * 流程：
    * 1. 读取输入模型到Polyhedron和Surface_mesh
    * 2. 使用AABB树检测相交面对
    * 3. 对每个相交面对计算交线
    * 4. 写入交线数据并进行采样处理
    *****************************************************************/
    int intersection_main(string model1name, string model2name, double rate)
    {
        // 文件路径设置
        const std::string filename1 = model1name + ".off";
        const std::string filename2 = model2name + ".off";

        
        std::cout << "(intersection_main)打开文件1: " << filename1 << std::endl;
        std::cout << "(intersection_main)打开文件2: " << filename2 << std::endl;

        // 读取模型到Polyhedron
        std::ifstream in1(filename1);
        if (!in1) { std::cerr << "无法读取多面体p1数据" << std::endl; return 1; }
        std::ifstream in2(filename2);
        if (!in2) { std::cerr << "无法读取多面体p2数据" << std::endl; return 1; }
        in1 >> p1;
        in2 >> p2;
        in1.close();
        in2.close();

        // 读取模型到Surface_mesh（用于AABB树检测）
        Mesh mesh1, mesh2;
        if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(filename1, mesh1) ||
            !CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(filename2, mesh2))
        {
            std::cerr << "Invalid input." << std::endl;
            return 1;
        }
        std::cerr << "模型一面数:" << mesh1.number_of_faces() << std::endl
            << "模型二面数:" << mesh2.number_of_faces() << std::endl << std::endl;

        Mesh mesh_intersection;
        bool ok = PMP::corefine_and_compute_intersection(
            mesh1,
            mesh2,
            mesh_intersection,
            PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh1)),
            PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh2)),
            PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh_intersection))
        );
        if (!ok) {
            std::cerr << "[WARNING] Intersection result may be non-manifold." << std::endl;
        }


        std::cerr << "What Happened? 292" << std::endl;

        
        PMP::surface_intersection(
            mesh1,
            mesh2,
            std::back_inserter(intersection_segments),
            PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh1)),
            PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh2))
        );

        std::cout << "提取交线段数: " << intersection_segments.size() << std::endl;



        for (auto& poly : intersection_segments) {
            if (poly.size() < 2) continue;

            // 1. 计算各段长度与总长 L
            std::vector<double> seg_len(poly.size() - 1);
            double L = 0;
            for (size_t j = 1; j < poly.size(); ++j) {
                seg_len[j - 1] = std::sqrt(CGAL::squared_distance(poly[j - 1], poly[j]));
                L += seg_len[j - 1];
            }

            // 2. 确定采样点数 m 与间隔 delta
            int m = std::max(2, int(std::ceil(rate * poly.size() ) ));
            double delta = L / (m - 1);

            // 3. 新折线容器
            std::vector<Point_3> sampled;
            sampled.reserve(m);
            sampled.push_back(poly.front());

            double acc = 0;
            size_t idx = 0;
            double rem = seg_len[0];

            for (int i = 1; i < m - 1; ++i) {
                double target = i * delta;
                while (acc + rem < target && idx + 1 < seg_len.size()) {
                    acc += rem;
                    idx++;
                    rem = seg_len[idx];
                }
                double t = (target - acc) / rem;
                const auto& A = poly[idx];
                const auto& B = poly[idx + 1];
                Point_3 P(
                    (1 - t) * A.x() + t * B.x(),
                    (1 - t) * A.y() + t * B.y(),
                    (1 - t) * A.z() + t * B.z()
                );
                sampled.push_back(P);
            }

            sampled.push_back(poly.back());
            poly.swap(sampled); // 原地替换
        }

        return 0;
    }
    
};
