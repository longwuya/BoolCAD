#pragma once

#include "IntersectionTest.h"
#include "Distance_OBB.h"
#include <fstream>
#include <string>
#include <iostream>
#include "BooleanComputation.h"
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <streambuf>

using namespace std;

// Exact predicates with inexact constructions balance robustness and speed.
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

// Geometry type aliases.
typedef Kernel::Point_3 Point_3;
typedef Kernel::Triangle_3 Triangle_3;
typedef Kernel::Segment_3 Segment_3;

/* Run a Boolean operation between two 3D models and optimize the result. */

// Function declarations.
void PreLoad(CBaseModel* inputModel, int booleanType, string fileName1, string fileName2);

struct FinalResultInfo {
    std::string outputPath;
    int pointCount = 0;
};

// Utility function: return the filename component of a path.
std::string extractAfterLastSlash(const std::string& input) {
    // Find the last forward slash.
    size_t lastSlashPos = input.find_last_of("/\\");
    
    // Return the suffix after the last slash.
    if (lastSlashPos != std::string::npos) {
        return input.substr(lastSlashPos + 1);
    }
    
    // Return the original string when no slash exists.
    return input;
}

/*------------------------------------------------------------------
Generate a timestamp string with minute precision.
Format: YYYYMMDD_HHMM
------------------------------------------------------------------*/
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_s(&tm_buf, &in_time_t);  // Secure Windows variant.

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M");
    return oss.str();
}


/*------------------------------------------------------------------
Create a directory when it does not already exist.
Parameter: path - directory path.
Returns true on success or when the directory already exists.
------------------------------------------------------------------*/
bool createDirectory(const std::string& path) {

    cout << path << endl;
    // Create the directory through the Windows API.
    if (CreateDirectoryA(path.c_str(), NULL)) {
        return true;  // Directory created.
    }

    // Inspect the error code.
    DWORD error = GetLastError();
    if (error == ERROR_ALREADY_EXISTS) {
        return true;  // Directory already exists.
    }
    
    // Report the failure.
    std::cerr << "Error creating directory: " << error << std::endl;
    return false;
}


// Preload meshes for the Boolean operation A OP B.
void PreLoad(CBaseModel* inputModel, int booleanType, string fileName1, string fileName2)
{
    inputModel[1] = CBaseModel(fileName1);
    inputModel[2] = CBaseModel(fileName2);
    inputModel[1].LoadModel();
    inputModel[2].LoadModel();
    inputModel[0] = CBaseModel();
    inputModel[0].AppendModel(inputModel[1]);
    inputModel[0].AppendModel(inputModel[2]);
}

// Resolve the Boolean operation from the optional fourth CLI argument.
int resolveBooleanType(int argc, char** argv)
{
    if (argc <= 4) {
        return 1;
    }

    std::string operation = argv[4];
    std::transform(operation.begin(), operation.end(), operation.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (operation == "u" || operation == "union" || operation == "1") {
        return 1;
    }
    if (operation == "i" || operation == "intersection" || operation == "2") {
        return 2;
    }
    if (operation == "d" || operation == "s" || operation == "difference" ||
        operation == "subtraction" || operation == "3") {
        return 3;
    }

    throw std::invalid_argument(
        "operation must be union/1, intersection/2, or difference/3");
}

// Approximate Boolean reconstruction entry point.
FinalResultInfo main_approximate(int argc, char** argv)
{
    const int base_point_num = (argc > 3) ? std::stoi(argv[3]) : 2000;
    const double factor_decay = (argc > 5) ? std::stod(argv[5]) : 0.98;
    const double factor_for_particle_system =
        (argc > 6) ? std::stod(argv[6]) : 0.30;
    constexpr double intersection_rate = 0.01;
    constexpr double sigma_rate = 6.0;
    constexpr double lambda_penalty = 0.0;
    constexpr double lambda_normal = 5.0;
    constexpr double sample_radius = 1.0;
    constexpr double e_param_bottom = 0.0;
    const int booleanType = resolveBooleanType(argc, argv);

    const std::string model1_name = (argc > 1) ? argv[1] :
        CGAL::data_file_path(R"(Intersection/Banana_1.off)").substr(
            0, CGAL::data_file_path(R"(Intersection/plane_dense.off)").size() - 4);
    const std::string model2_name = (argc > 2) ? argv[2] :
        CGAL::data_file_path(R"(Intersection/Banana_2.off)").substr(
            0, CGAL::data_file_path(R"(Intersection/plane_dense.off)").size() - 4);

    CBaseModel inputModel[3];
    cout << " check point 1:\n";
    PreLoad(inputModel, 1, model1_name + ".obj", model2_name + ".obj");
    cout << " check point 2:\n";

    intersection_test test1;
    const clock_t start = clock();
    test1.intersection_main(model1_name, model2_name, intersection_rate);
    const double running_time = ((double)(clock() - start) / 1000);
    cerr << "intersection_main running time = " << running_time << " seconds" << endl;

    const std::vector<std::vector<Point_3>> intersection_segments =
        test1.get_intersection_segments();
    cout << " 求解得到的交线集中共包含 "
         << intersection_segments.size() << " 条线段\n";

    const string curtime = getCurrentTimestamp();
    const string folderName =
        extractAfterLastSlash(model1_name) + "---" +
        extractAfterLastSlash(model2_name) + " --- " +
        char(booleanType + '0') + " ---" + curtime;
    createDirectory(folderName);

    BooleanComputation booleanComputation(
        inputModel,
        base_point_num,
        factor_for_particle_system,
        0.0,
        lambda_penalty,
        0.5,
        sample_radius,
        lambda_normal,
        booleanType,
        0.0,
        intersection_segments,
        intersection_rate,
        e_param_bottom,
        sigma_rate,
        factor_decay
    );
    cout << " check point 3:\n";
    booleanComputation.run(
        folderName,
        base_point_num,
        factor_for_particle_system,
        factor_decay,
        intersection_rate,
        booleanComputation.get_sample_num(),
        sigma_rate,
        e_param_bottom
    );

    return {
        booleanComputation.GetLastOutputPath(),
        booleanComputation.GetLastOutputPointCount()
    };
}


// Program entry point.
int main(int argc, char** argv) {
    class NullStreamBuffer final : public std::streambuf {
    protected:
        int_type overflow(int_type ch) override {
            return traits_type::not_eof(ch);
        }
    };
    static NullStreamBuffer nullBuffer;
    auto* originalCout = std::cout.rdbuf();
    std::cout.rdbuf(&nullBuffer);
    std::cerr.rdbuf(&nullBuffer);
    std::clog.rdbuf(&nullBuffer);

    const auto start = std::chrono::steady_clock::now();
    const FinalResultInfo result = main_approximate(argc, argv);
    const double elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    std::cout.rdbuf(originalCout);
    std::cout << "Output path: " << result.outputPath << '\n'
              << "Actual points: " << result.pointCount << '\n'
              << "Elapsed time: " << std::fixed << std::setprecision(3)
              << elapsedSeconds << " s" << std::endl;
    return 0;
}



// Error-metric version retained for reference.
//#pragma once
//
//#include "intersection_test.cpp"
//#include "Distance_OBB.h"
//#include <fstream>
//#include <string>
//#include <iostream>
//#include "RichModel.h"
//#include "BooleanComputation.h"
//#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
//#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
//
//using namespace std;
//
//// Exact predicates with inexact constructions balance robustness and speed.
//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
//
//// Geometry type aliases.
//typedef Kernel::Point_3 Point_3;
//typedef Kernel::Triangle_3 Triangle_3;
//typedef Kernel::Segment_3 Segment_3;
//
///* Run a Boolean operation between two 3D models and optimize the result. */
//
//// Function declarations.
//void MixObjModel(string fileName1, string fileName2);  // Merge two OBJ meshes.
//void PreLoad(CBaseModel* inputModel, int booleanType, string fileName1, string fileName2);  // Preload input meshes.
//
//// Utility function: return the filename component of a path.
//std::string extractAfterLastSlash(const std::string& input) {
//    // Find the last forward slash.
//    size_t lastSlashPos = input.find_last_of('/');
//
//    // Return the suffix after the last slash.
//    if (lastSlashPos != std::string::npos) {
//        return input.substr(lastSlashPos + 1);
//    }
//
//    // Return the original string when no slash exists.
//    return input;
//}
//
///*------------------------------------------------------------------
// Generate a timestamp string with minute precision.
// Format: YYYYMMDD_HHMM
//------------------------------------------------------------------*/
//std::string getCurrentTimestamp() {
//    auto now = std::chrono::system_clock::now();
//    auto in_time_t = std::chrono::system_clock::to_time_t(now);
//    std::tm tm_buf;
//    localtime_s(&tm_buf, &in_time_t);  // Secure Windows variant.
//
//    std::ostringstream oss;
//    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M");
//    return oss.str();
//}
//
//
///*------------------------------------------------------------------
// Create a directory when it does not already exist.
// Parameter: path - directory path.
// Returns true on success or when the directory already exists.
//------------------------------------------------------------------*/
//bool createDirectory(const std::string& path) {
//
//    cout << path << endl;
//    // Create the directory through the Windows API.
//    if (CreateDirectoryA(path.c_str(), NULL)) {
//        return true;  // Directory created.
//    }
//
//    // Inspect the error code.
//    DWORD error = GetLastError();
//    if (error == ERROR_ALREADY_EXISTS) {
//        return true;  // Directory already exists.
//    }
//
//    // Report the failure.
//    std::cerr << "Error creating directory: " << error << std::endl;
//    return false;
//}
//
//
//// Preload meshes for the Boolean operation A OP B.
//void PreLoad(CBaseModel* inputModel, int booleanType, string fileName1, string fileName2)
//{
//    MixObjModel(fileName1, fileName2);  // Generate the combined mesh file.
//    cout << "MixObjModel finished" << endl;
//    inputModel[0] = CBaseModel("Intersection//mix_files.obj");  // Load the combined mesh.
//    inputModel[1] = CBaseModel(fileName1);
//    inputModel[2] = CBaseModel(fileName2);
//    // Load mesh data.
//    inputModel[0].LoadModel();
//    inputModel[1].LoadModel();
//    inputModel[2].LoadModel();
//}
//
//// Merge two OBJ meshes.
//void MixObjModel(string fileName1, string fileName2)
//{
//    // Open the input files.
//    ifstream inputFile1(fileName1);
//    ifstream inputFile2(fileName2);
//    ofstream outputFile("Intersection//mix_files.obj");  // Combined OBJ output.
//
//    vector<CPoint3D> mixModelVertices;  // Combined vertex set.
//    vector<CPoint3D> mixModelFaces;     // Combined face set.
//
//
//
//    
//    // Read model 1.
//    string str1, str2, str3;
//    double d1, d2, d3;
//    while (!inputFile1.eof()) {
//        inputFile1 >> str1;
//        if (str1 == "v") {  // Vertex record.
//            inputFile1 >> str1 >> str2 >> str3;
//            mixModelVertices.push_back(CPoint3D(atof(str1.c_str()), atof(str2.c_str()), atof(str3.c_str())));
//        }
//        else if (str1 == "f") {  // Face record.
//            inputFile1 >> str1 >> str2 >> str3;
//            mixModelFaces.push_back(CPoint3D(atof(str1.c_str()), atof(str2.c_str()), atof(str3.c_str())));
//        }
//    }
//    inputFile1.close();
//
//
//    // Read model 2 and offset its face indices.
//    int mode1Size = mixModelVertices.size();  // Number of vertices in model 1.
//    while (!inputFile2.eof()) {
//        inputFile2 >> str1;
//        if (str1 == "v") {
//            inputFile2 >> str1 >> str2 >> str3;
//            mixModelVertices.push_back(CPoint3D(atof(str1.c_str()), atof(str2.c_str()), atof(str3.c_str())));
//        }
//        else if (str1 == "f") {
//            inputFile2 >> str1 >> str2 >> str3;
//            // Offset face indices by the vertex count of model 1.
//            mixModelFaces.push_back(CPoint3D(
//                atof(str1.c_str()) + mode1Size,
//                atof(str2.c_str()) + mode1Size,
//                atof(str3.c_str()) + mode1Size
//            ));
//        }
//    }
//    inputFile2.close();
//
//    // Write the combined OBJ mesh.
//    for (auto v : mixModelVertices)
//        outputFile << "v " << v.x << " " << v.y << " " << v.z << endl;
//    for (auto f : mixModelFaces)
//        outputFile << "f " << (int)f.x << " " << (int)f.y << " " << (int)f.z << endl;
//    outputFile.close();
//}
//
//// Approximate Boolean reconstruction entry point.
//void main_approximate(int argc, char** argv)
//{
//    const int base_point_num = (argc > 3) ? std::stoi(argv[3]) : 1000;
//    const double rate = (argc > 4) ? std::stod(argv[4]) : 0.0;
//
//
//        //============= Parameter setup ==============//
//        int n_sample = 1000;          // Requested surface sample count.
//
//        double lambda_penlaty = 0;     // Regularization penalty.
//        double sampleRadius = 1;       // Sampling radius.
//
//        // Resolve input mesh paths from CLI arguments or defaults.
//        const std::string model1_name = (argc > 1) ? argv[1] : CGAL::data_file_path(R"(Intersection/plane_dense.off)").substr(0, CGAL::data_file_path(R"(Intersection/plane_dense.off)").size() - 4);
//        const std::string model2_name = (argc > 2) ? argv[2] : CGAL::data_file_path(R"(Intersection/moai_1000.off)").substr(0, CGAL::data_file_path(R"(Intersection/plane_dense.off)").size() - 4);
//
//        //=========== Preload input meshes ==========//
//        CBaseModel inputModel[3];  // [0] combined, [1] model A, [2] model B
//        cout << " check point 1:\n";
//        PreLoad(inputModel, 1, model1_name + ".obj", model2_name + ".obj");
//        cout << " check point 2:\n";
//        //========= Run Boolean reconstruction ==========//
//        // Parameter sweep setup.
//        double factor_ps_start = 0.30;
//        double factor_ps_end = 0.30;
//        double factor_ps_step = 0.05;
//
//        double e_param_for_seg_start = 0.0;
//        double e_param_for_seg_end = 0.2;
//        double e_param_for_seg_step = 0.01;
//
//        double e_param_for_ps_start = 1.0;
//        double e_param_for_ps_end = 1.0;
//        double e_param_for_ps_step = 0.01;
//
//
//        int booleanType = 1; // 1-union, 2-intersection, 3-subtraction.
//
//
//            std::vector<std::vector<Point_3>> intersection_empty;  // Use this empty input to disable intersection constraints.
//
//            for (double factor_for_particle_system = factor_ps_start; factor_for_particle_system <= factor_ps_end; factor_for_particle_system += factor_ps_step) // Particle-system distribution factor.
//            {
//
//                string curtime = getCurrentTimestamp();
//                // Build the output folder name.
//                string folderName = extractAfterLastSlash(model1_name) + "---" + extractAfterLastSlash(model2_name) + " --- " + char(booleanType + '0') + " ---" + curtime;
//                createDirectory(folderName);
//
//                for (double e_param_bottom = 1.0; e_param_bottom <= 1.0; e_param_bottom += 0.1) // Intersection-region density factor.
//                {
//
//                    for (double sigma_rate = 0.0; sigma_rate <= 15.0; sigma_rate += 3.0) // Intersection-region weight factor.
//                    {
//
//                        BooleanComputation booleanComputation(
//                            inputModel,
//                            n_sample,
//                            factor_for_particle_system,
//                            0.0,
//                            lambda_penlaty,
//                            0.5,            // Default weight factor.
//                            sampleRadius,
//                            1,              // Enable adaptive sampling.
//                            booleanType,
//                            0.0,             // Subtraction weight factor.
//                            intersection_empty,
//                            0,
//                            e_param_bottom,
//                            sigma_rate
//                        );
//                        cout << " check point 3:\n";
//                    }
//                }
//
//    }
//}
//
//
//// Program entry point.
//void main(int argc, char** argv) {
//    main_approximate(argc, argv);
//}
//




