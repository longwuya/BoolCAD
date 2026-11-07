/*-----------------------------------------------------------------
Non-linear optimization framework based on L-BFGS.
Features:
1. Tracks optimization progress and writes results.
2. Supports custom energy functions and constraints.
3. Integrates CGAL geometry processing.
4. Provides thread-safe timestamp generation and file handling.
------------------------------------------------------------------*/
#pragma once  // Prevent duplicate inclusion of this header.

//------------------------------ Dependencies ------------------------------//
// Linear algebra and optimization.
#include "lbfgs.h"        // L-BFGS optimizer.
#include <vector>         // STL vector.
#include <iostream>       // Standard I/O streams.
#include <algorithm>

// File operations.
#include "fstream"        // File streams.
#include "string"         // String handling.
#include <windows.h>  // Windows API.

// Geometry processing (CGAL).
#include <CGAL/Simple_cartesian.h>                   // Simple Cartesian kernel.
#include <CGAL/AABB_tree.h>                          // AABB tree.
#include <CGAL/AABB_traits.h>                        // AABB traits.
#include <CGAL/AABB_triangle_primitive.h>            // AABB triangle primitive.

// Project headers.
#include "Distance_OBB.h"           // OBB distance queries.
#include <chrono>                   // Time utilities.
#include <mutex>                    // Mutual exclusion.

// Filesystem and formatting utilities.
#include <filesystem>   // Filesystem operations.
#include <sstream>      // String streams.
#include <iomanip>      // I/O formatting.
#include <ctime>        // Time handling.
#include <cstdlib>

//------------------------ CGAL type aliases ------------------------//
typedef CGAL::Simple_cartesian<double> K;           // Cartesian kernel.
typedef K::FT FT;                                    // Scalar type.
typedef K::Ray_3 Ray;                                // 3D ray.
typedef K::Line_3 Line;                              // 3D line.
typedef K::Point_3 Point;                            // 3D point.
typedef K::Triangle_3 Triangle;                      // 3D triangle.
typedef std::list<Triangle>::iterator Iterator;      // Triangle-list iterator.
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;         // AABB primitive.
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;         // AABB traits.
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;                   // AABB tree.
typedef Tree::Point_and_primitive_id Point_and_primitive_id;          // Point/primitive ID pair.

using namespace std;

inline bool IterationConsoleOutputEnabled()
{
	static const bool enabled = []() {
		const char* silent = std::getenv("RECONSTRUCTION_SILENT");
		return silent == nullptr || std::string(silent) != "1";
	}();
	return enabled;
}

/*------------------------------------------------------------------
Base class for non-linear optimization.
Design:
1. Uses L-BFGS.
2. Follows the template-method pattern; subclasses implement the energy function.
3. Tracks optimization progress and records results.
4. Provides thread-safe timestamp generation.
------------------------------------------------------------------*/
class Non_Linear_Optimization
{
protected:
    vector<lbfgsfloatval_t> m_variables;  // Optimization variables, usually 3D point coordinates.
    ofstream statusReport;                // Optimization status report.
    int maxIteration;                     // Last recorded iteration count.
    lbfgsfloatval_t last_result[10];      // Results from the latest iteration.

protected:
    /*------------------------------------------------------------------
    Optimization progress callback, invoked once per iteration.
    Parameters:
    x     - Current variables.
    g     - Current gradient.
    fx    - Current energy.
    xnorm - Variable norm.
    gnorm - Gradient norm.
    step  - Step length.
    n     - Number of variables.
    k     - Current iteration.
    ls    - Line-search count.
    ------------------------------------------------------------------*/
    virtual int progress(
        const lbfgsfloatval_t* x,
        const lbfgsfloatval_t* g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
    )
    {
        // Write live progress to the console.
        if (IterationConsoleOutputEnabled()) {
            cerr << " Iteration #" << k << ": ";
            cerr << " Energy = " << fx << ", ";
            cerr << " step = " << step << ", ";
            cerr << " Error = " << gnorm / std::max(1.0, xnorm) << endl;
            cerr << "-----------------------------------------\n";
        }

        maxIteration = k;  // Update the iteration counter.

        last_result[1] = fx;
        last_result[2] = gnorm / std::max(1.0, xnorm);
        last_result[3] = step;
        last_result[4] = gnorm;
        last_result[5] = xnorm;
        // Write a status record: iteration, energy, gradient norm, variable norm, relative error.
        //statusReport << k << " " << fx << " " << gnorm << " " << xnorm
        //    << " " << gnorm / max(1, xnorm) << endl;

        return 0;
    }

    std::string convertToString(double num) {

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << num;
        return oss.str();

    }

    // Static adapter from the C-style callback to the member function.
    static int _progress(
        void* instance,
        const lbfgsfloatval_t* x,
        const lbfgsfloatval_t* g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
    )
    {
        return reinterpret_cast<Non_Linear_Optimization*>(instance)->progress(
            x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    /*------------------------------------------------------------------
    Pure virtual energy function implemented by subclasses.
    Computes energy and gradient for the current variables.
    Parameters:
    x    - Input variables.
    g    - Output gradient.
    n    - Number of variables.
    step - Current step length.
    Returns: Energy value.
    ------------------------------------------------------------------*/
    virtual lbfgsfloatval_t evaluate(
        lbfgsfloatval_t* x,
        lbfgsfloatval_t* g,
        const int n,
        const lbfgsfloatval_t step
    ) = 0;

    // Static adapter for the energy callback.
    static lbfgsfloatval_t _evaluate(
        void* instance,
        lbfgsfloatval_t* x,
        lbfgsfloatval_t* g,
        const int n,
        const lbfgsfloatval_t step
    )
    {
        return reinterpret_cast<Non_Linear_Optimization*>(instance)->evaluate(x, g, n, step);
    }

    //------------------------ Helper functions ------------------------//


public:
    /*------------------------------------------------------------------
    Run the main optimization workflow.
    Parameters:
    arg0 - Executable name used in folder construction.
    arg1 - First launch argument used in folder construction.
    nSamples - Number of sample points.
    factor_for_particle_system - Particle-system constraint weight.
    factor_for_intersection - Intersection constraint weight.
    lamdaForBalancingParticleSystemAndDistanceConstraints - Balance coefficient.
    e_param_for_seg - Segmentation parameter, used only in the filename.
    surface_point_num - Surface point count, used only in the filename.
    intersection_point_num - Intersection point count, used only in the filename.
    ------------------------------------------------------------------*/
    template <typename T>
    string name_tran2(T n) {    // Round to two decimals and convert to a string.
        return  convertToString(std::round(double(n) * 100) / 100);
    }

    template <typename T>
    string name_tran4(T n) {    // Round to four decimals and convert to a string.
        return  convertToString(std::round(double(n) * 10000 ) / 10000);
    }

    string name_tran(int n) {    // Convert an integer to a string.
        return  std::to_string(n);
    }

    virtual void run(
        const std::string& folder_name, // Output folder.
        int nSamples,
        double factor_for_particle_system,
        double lamdaForBalancingParticleSystemAndDistanceConstraints,
        double intersection_rate,   // Filename metadata only; computation occurs in BooleanComputation.
        double surface_point_num,   // Filename metadata only; computation occurs in BooleanComputation.
        double sigma_rate,   // Filename metadata only; computation occurs in BooleanComputation.
        double e_param_bottom   // Filename metadata only; computation occurs in BooleanComputation.
    )
    {
        lbfgsfloatval_t fx = 0;  // Final energy value.

        // Build the output path with parameter metadata.
        string fileName = folder_name + "/";  // Output file inside the result folder.

        cout << "Starting optimization..." << endl;


        // Configure and invoke L-BFGS.
            // ======== Custom solver parameters ========
        lbfgs_parameter_t custom_params;
        lbfgs_parameter_init(&custom_params);  // Initialize defaults.

        // Configure termination safeguards.
        custom_params.max_iterations = 300;  // Maximum iterations.
        custom_params.epsilon = 1e-5;        // Gradient convergence threshold.
        custom_params.delta = 1e-9;          // Objective-change threshold.
        custom_params.past = 0;               // Number of past iterations to inspect.
        custom_params.min_step = 1e-20; // Minimum step length.
        
        // Start timing.
        clock_t start = clock();

        // Run the optimizer with the custom parameters.
        int ret = lbfgs(m_variables.size(), &m_variables[0], &fx,
            _evaluate, _progress, this, &custom_params);

        // Compute elapsed optimization time.
        double running_time = ((double)(clock() - start) / 1000);

        // Report the solver exit status.
        cerr << "Optimization exit status: ";
        switch (ret) {
        case LBFGS_SUCCESS:
            cerr << "SUCCESS (梯度收敛)" << endl; break;
        case LBFGS_STOP:
            cerr << "STOP (函数值稳定)" << endl; break;
        case LBFGSERR_MAXIMUMITERATION:
            cerr << "MAX_ITERATIONS (达到上限)" << endl; break;
        case LBFGSERR_MINIMUMSTEP:
            cerr << "MIN_STEP (步长过小)" << endl; break;
        default:
            cerr << "ERROR CODE: " << ret << endl;
        }

        
        cerr << "Total running time = " << running_time << " seconds" << endl;

        // Build the final output filename.
        fileName += ("surn=" + name_tran(surface_point_num) +
            "_irate=" + name_tran2(intersection_rate) +
            "_ffps=" + name_tran2(factor_for_particle_system) +
            "_ebottom=" + name_tran2(e_param_bottom) +
            "_sigma=" + name_tran2(sigma_rate) +
            "_it=" + name_tran(maxIteration) +
            "_time=" + name_tran2(running_time) +
            "_Err=" + name_tran4(last_result[2]) +
            ".xyz");

        // Write the optimized point cloud.
        ofstream outputReconstructionModel(fileName);

        // Write variables as 3D points; m_variables stores xyz triples.
        // This output changes only when m_variables changes.
        for (vector<double>::iterator it = m_variables.begin(); it != m_variables.end(); it += 3)
        {
            outputReconstructionModel << *it << " " << *(it + 1) << " " << *(it + 2) << endl;
        }

        //// Optional: append intersection-point data.
        //ifstream intersection_line;
        //intersection_line.open("Intersection/intersection_point_dealt.xyz");
        //if (!intersection_line) {
        //    cerr << "intersection_point file read error1!" << endl;
        //    exit(0);
        //}
        //float x_temp, y_temp, z_temp;
        //string line;
        //while (getline(intersection_line, line))
        //{
        //    istringstream iss(line);
        //    if (iss >> x_temp >> y_temp >> z_temp) {
        //        outputReconstructionModel << x_temp << " " << y_temp << " " << z_temp << endl;
        //    }
        //    else {
        //        std::cerr << "intersection_point file read error3! " << std::endl;
        //    }
        //}
        //intersection_line.close();
        ////---------------------------------------------------------------

        outputReconstructionModel.close();
        //statusReport.close();

        cerr << "Optimization results saved to: " << fileName << endl;

        /* Original note retained: optional output without intersection points.
        fileName += ("without intersection.xyz" );
        ofstream outputReconstructionModel_2(fileName);
        for (vector<double>::iterator it = m_variables.begin();
            it != m_variables.end();
            it += 3)
        {
            outputReconstructionModel_2 << *it << " " << *(it + 1) << " " << *(it + 2) << endl;
        }
        outputReconstructionModel_2.close();
        */
    }

    /* Original note retained: possible coordinate conversion helper.
    Point ConvertToXYZ(double u, double v) const
    {
        double x = sin(u) * cos(v);
        double y = sin(u) * sin(v);
        double z = cos(u);
        return Point(x, y, z);
    }
    */
};
