#include "BooleanComputation.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#pragma comment(lib, "ANN.lib")  // Link the Approximate Nearest Neighbor library.

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;


typedef CGAL::Surface_mesh<Point_3> Mesh;
// Geometry type aliases.
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Triangle_3 Triangle_3;
typedef Kernel::Segment_3 Segment_3;

// Flattened segments must outlive the AABB tree that references them.
std::vector<Segment_3> m_intersection_segments_flat;

// AABB tree aliases based on segment primitives.
using SegmentPrimitive = CGAL::AABB_segment_primitive<Kernel, std::vector<Segment_3>::iterator>;
using AABBTraits = CGAL::AABB_traits<Kernel, SegmentPrimitive>;
using SegmentTree = CGAL::AABB_tree<AABBTraits>;

std::unique_ptr<SegmentTree> m_aabb_tree; // Use unique_ptr to avoid copying the tree.

namespace {
bool ReadEnvironmentFlag(const char* name, bool defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0;
}

int ReadEnvironmentInt(
    const char* name,
    int defaultValue,
    int minimumValue,
    int maximumValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    const int parsedValue = std::atoi(value);
    if (parsedValue < minimumValue || parsedValue > maximumValue) {
        return defaultValue;
    }
    return parsedValue;
}

double ReadEnvironmentDouble(
    const char* name,
    double defaultValue,
    double minimumValue,
    double maximumValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    const double parsedValue = std::atof(value);
    if (parsedValue < minimumValue || parsedValue > maximumValue) {
        return defaultValue;
    }
    return parsedValue;
}

ANNsplitRule ReadEnvironmentSplitRule(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return ANN_KD_SUGGEST;
    }
    if (std::strcmp(value, "std") == 0) return ANN_KD_STD;
    if (std::strcmp(value, "midpt") == 0) return ANN_KD_MIDPT;
    if (std::strcmp(value, "fair") == 0) return ANN_KD_FAIR;
    if (std::strcmp(value, "sl_midpt") == 0) return ANN_KD_SL_MIDPT;
    if (std::strcmp(value, "sl_fair") == 0) return ANN_KD_SL_FAIR;
    return ANN_KD_SUGGEST;
}
}



// Construct the Boolean reconstruction engine.
// Parameters:
//   inputModel[3]: combined model, model A, and model B.
//   nSamples: requested surface sample count.
//   factor_for_particle_system: particle-system factor.
//   factor_for_intersection: intersection-constraint weight.
//   lamdaForBalancing...: balance between particle and distance constraints.
//   radiusOfBall: local query radius.
//   boundingBoxRadiusOfSurface: surface bounding-box radius.
//   lambda_ofnoraml_penalty: normal penalty weight.
//   booleanType: 1-union, 2-intersection, 3-subtraction.
//   subtractionWeightFactor: subtraction-specific weight.
BooleanComputation::BooleanComputation(
    const CBaseModel (&inputModel)[3],
    int nSamples,
    double factor_for_particle_system,
    double factor_for_intersection,
    double lamdaForBalancingParticleSystemAndDistanceConstraints,
    double radiusOfBall,
    double boundingBoxRadiusOfSurface,
    double lambda_ofnoraml_penalty,
    int booleanType,
    double subtractionWeightFactor,
    std::vector<std::vector<Point_3>> intersection_segments,
    double i_intersection_rate,
    double e_e_param_bottom,
    double s_sigma_rate
) : Reconstruction_Basedon_Paricles_LBFGS
(nSamples,
    factor_for_particle_system,
    factor_for_intersection,
    lamdaForBalancingParticleSystemAndDistanceConstraints),
    m_inputModel{ inputModel[0], inputModel[1], inputModel[2] },
    m_obbtree(m_inputModel[0]),
    m_obbtree1(m_inputModel[1]),
    m_obbtree2(m_inputModel[2]),
    m_boundingBoxRadiusOfSurface(boundingBoxRadiusOfSurface),
    m_lambda_ofnoraml_penalty(lambda_ofnoraml_penalty),
    m_booleanType(booleanType),
    m_subtractionWeightFactor(subtractionWeightFactor),
    intersection_seg(std::move(intersection_segments)),
    intersection_rate(i_intersection_rate),
    e_param_bottom(e_e_param_bottom),
    sigma_rate(s_sigma_rate)
{
    // Input models are copied once during member initialization.
    m_ann_bucket_size = ReadEnvironmentInt(
        "RECONSTRUCTION_ANN_BUCKET", m_ann_bucket_size, 1, 64);
    m_particle_ann_bucket_size = ReadEnvironmentInt(
        "RECONSTRUCTION_PARTICLE_ANN_BUCKET",
        m_ann_bucket_size,
        1,
        64);
    m_ena_ann_bucket_size = ReadEnvironmentInt(
        "RECONSTRUCTION_ENA_ANN_BUCKET",
        m_ann_bucket_size,
        1,
        64);
    m_particle_neighbor_epsilon = ReadEnvironmentDouble(
        "RECONSTRUCTION_PARTICLE_EPS",
        m_particle_neighbor_epsilon,
        0.0,
        0.2);
    m_ena_neighbor_epsilon = ReadEnvironmentDouble(
        "RECONSTRUCTION_ENA_EPS",
        m_ena_neighbor_epsilon,
        0.0,
        0.2);
    m_ena_search_radius_scale = ReadEnvironmentDouble(
        "RECONSTRUCTION_ENA_RADIUS_SCALE",
        m_ena_search_radius_scale,
        0.5,
        1.5);
    m_particle_ann_split_rule =
        ReadEnvironmentSplitRule("RECONSTRUCTION_PARTICLE_ANN_SPLIT");
    m_ena_ann_split_rule =
        ReadEnvironmentSplitRule("RECONSTRUCTION_ENA_ANN_SPLIT");
    m_ann_single_pass_query = ReadEnvironmentFlag(
        "RECONSTRUCTION_ANN_SINGLE_PASS", true);
    m_ann_morton_query_order = ReadEnvironmentFlag(
        "RECONSTRUCTION_ANN_MORTON_ORDER", false);
    m_ann_csr_neighbor_storage = ReadEnvironmentFlag(
        "RECONSTRUCTION_ANN_CSR", false);
    buildVertexToTrianglesMap();
    // Lion_Hand_Our_reconstruction
    // LDNI_Hand_Lion_512_result
    // GetDensityAtPoint currently returns the constant density 1.0, so the
    // principal-curvature solve would only add startup cost.
    // Optional error metric; irrelevant to the reconstruction pipeline.
    // std::cout << ErrorToMeshSurface(m_inputModel[0], m_obbtree1, 0.0, 0.02847294, "Lion_Hand_Our_reconstruction.obj") << " ERROR result\n";

    // Generate initial surface samples.
    vector<Point> samples = GetInitialSamplesOnSurface(nSamples); // Includes fixed optimization points.

    std::cout << " 表面初始采样点的数目: " << samples.size() << endl;

    if (IterationConsoleOutputEnabled()) {
        ofstream outputReconstructionModel1("sample_origin.xyz");
        for (Point p : samples) {
            outputReconstructionModel1
                << p.x() << " " << p.y() << " " << p.z() << endl;
        }
    }




    if (booleanType == 1) {
        samples = GetValidSamplesByUnion(samples, m_obbtree1, m_obbtree2);
    }
    else if (booleanType == 2) {
        samples = GetValidSamplesByIntersection(samples, m_obbtree1, m_obbtree2);
    }
    else if (booleanType == 3) {
        samples = GetValidSamplesBySubtraction(samples, m_obbtree1, m_obbtree2);
    }

    std::cout << "合法性筛选后, 采样点的数目: " << samples.size() << endl;





    // Store samples as xyz triples in the optimization variable array.
    // Fixed points can be appended directly to samples.
    // nSamples is the requested mutable point count.
    for (int i = 0; i < samples.size(); ++i) // m_variables is passed to evaluate() as x.
    {
        m_variables.push_back(samples[i].x());
        m_variables.push_back(samples[i].y());
        m_variables.push_back(samples[i].z());
    }

    for (const auto& poly : intersection_seg) {
        if (poly.size() < 2) continue;  // Ignore single-point polylines.
        for (size_t i = 1; i < poly.size(); ++i) {

            m_variables.push_back(poly[i - 1].x());
            m_variables.push_back(poly[i - 1].y());
            m_variables.push_back(poly[i - 1].z());

            m_variables.push_back(poly[i].x());
            m_variables.push_back(poly[i].y());
            m_variables.push_back(poly[i].z());
        }
    }
     
    // The first 3 * samples.size() values are surface points; the rest encode intersection segments.

    // Compute model area and particle-system parameters.
    m_modelArea = GetSurfaceArea();
    m_sigma = m_factor_for_particle_system * sqrt(m_modelArea / m_nSamples);    // Preserve this formula.

    m_nSamples = m_variables.size() / 3;
    m_neighborLists.resize(m_nSamples);
    m_enaNeighborTreeIndices.resize(m_nSamples);
    m_enaNearbyTriangles.resize(m_nSamples);
    m_lastParticleNeighborPositions.resize(m_nSamples * 3, 0.0);
    m_lastEnaNeighborPositions.resize(m_nSamples * 3, 0.0);
    cout << "粒子系统中总点?? " << m_variables.size() / 3 << endl;



    // Flatten intersection polylines; a single point becomes a zero-length segment.
    m_intersection_segments_flat.clear();
    for (const auto& poly : intersection_seg) {
        if (poly.empty()) continue;
        if (poly.size() == 1) {
            m_intersection_segments_flat.emplace_back(poly[0], poly[0]); // degenerate segment for single point
        }
        else {
            for (size_t k = 0; k + 1 < poly.size(); ++k) {
                m_intersection_segments_flat.emplace_back(poly[k], poly[k + 1]);
            }
        }
    }

    // Build the AABB tree from the flattened segment range.
    m_aabb_tree = std::make_unique<SegmentTree>(m_intersection_segments_flat.begin(), m_intersection_segments_flat.end());
    // Optionally accelerate distance queries.
    m_aabb_tree->accelerate_distance_queries();


    if (IterationConsoleOutputEnabled()) {
        ofstream outputReconstructionModel2("sample_after_check.xyz");
        for (vector<double>::iterator it = m_variables.begin();
             it != m_variables.end();
             it += 3) {
            outputReconstructionModel2
                << *it << " " << *(it + 1) << " " << *(it + 2) << endl;
        }
    }


    // Select valid triangles for the E_NA integral.
    BooleanComputation::initFacesForENA();
    BuildENAKDTree();
    prepareENAIntegralData();
    // Debug: write the selected triangles.
    if (IterationConsoleOutputEnabled()) {
        int numFaces = m_inputModel[0].GetNumOfFaces();

        // Collect vertices of valid triangles and remap them to contiguous OBJ indices.
        vector<CPoint3D> validVertices;
        vector<array<int, 3>> validFaces;
        unordered_map<int, int> oldToNewIdx;

        for (int i = 0; i < numFaces; ++i) {
            if (!m_faceValidForENA[i]) continue;

            auto face = m_inputModel[0].Face(i);
            int newIdx[3];

            for (int j = 0; j < 3; ++j) {
                int oldIdx = face[j];
                if (oldToNewIdx.find(oldIdx) == oldToNewIdx.end()) {
                    oldToNewIdx[oldIdx] = validVertices.size();
                    validVertices.push_back(m_inputModel[0].Vert(oldIdx));
                }
                newIdx[j] = oldToNewIdx[oldIdx] + 1;  // OBJ indices are one-based.
            }

            validFaces.push_back({ newIdx[0], newIdx[1], newIdx[2] });
        }

        // Write the OBJ file.
        string outputFile = "ena_valid_model.obj";
        ofstream out(outputFile);
        if (out.is_open()) {
            for (const auto& v : validVertices) {
                out << "v " << v.x << " " << v.y << " " << v.z << endl;
            }
            for (const auto& f : validFaces) {
                out << "f " << f[0] << " " << f[1] << " " << f[2] << endl;
            }
            out.close();
            cout << "E_NA 有效模型已输出到: " << outputFile << endl;
        }
    }
}

// Release the ANN-owned resources.
BooleanComputation::~BooleanComputation() {
    const char* profile = std::getenv("RECONSTRUCTION_PROFILE");
    if (profile != nullptr && std::string(profile) == "1") {
        std::cerr
            << "[PROFILE] projection=" << m_profileProjectionSeconds
            << " neighbor_query=" << m_profileNeighborQuerySeconds
            << " particle_tree_build="
            << m_profileParticleTreeBuildSeconds
            << " particle_search=" << m_profileParticleSearchSeconds
            << " neighbor_energy=" << m_profileNeighborEnergySeconds
            << " ena_query=" << m_profileEnaQuerySeconds
            << " ena_ann=" << m_profileEnaANNSeconds
            << " ena_map=" << m_profileEnaMapSeconds
            << " ena_energy=" << m_profileEnaEnergySeconds
            << " neighbor_refresh=" << m_profileNeighborRefreshes
            << " neighbor_reuse=" << m_profileNeighborReuses
            << " ena_refresh=" << m_profileEnaRefreshes
            << " ena_reuse=" << m_profileEnaReuses
            << " max_particle_move="
            << std::sqrt(m_profileMaxParticleMove)
            << " max_ena_move=" << std::sqrt(m_profileMaxEnaMove)
            << std::endl;
    }

    if (m_ann_kdTree) {
        delete m_ann_kdTree;
        m_ann_kdTree = nullptr;
    }
    if (m_ann_dataPts) {
        annDeallocPts(m_ann_dataPts);
        m_ann_dataPts = nullptr;
        m_ann_point_capacity = 0;
        m_ann_point_num = 0;
    }
    // Other members use their default destructors.

    if (m_enaKDTree)
    {
        delete m_enaKDTree;
        m_enaKDTree = nullptr;
    }

    if (m_enaDataPts)
    {
        annDeallocPts(m_enaDataPts);
        m_enaDataPts = nullptr;
    }
}

// Ensure m_ann_dataPts can hold at least requiredPts points.
void BooleanComputation::EnsureANNCapacity(int requiredPts) {
    if (requiredPts <= 0) return;
    if (requiredPts <= m_ann_point_capacity) return;

    // Release the old point buffer when present.
    if (m_ann_dataPts) {
        annDeallocPts(m_ann_dataPts);
        m_ann_dataPts = nullptr;
        m_ann_point_capacity = 0;
    }

    // Allocate a new point buffer of requiredPts x m_ann_dim.
    m_ann_dataPts = annAllocPts(requiredPts, m_ann_dim);
    m_ann_point_capacity = requiredPts;
}

// Fill m_ann_dataPts from points and rebuild m_ann_kdTree.
void BooleanComputation::BuildOrUpdateANNKdTree(const std::vector<Point>& points) {
    int pointNum = static_cast<int>(points.size());
    if (pointNum == 0) {
        // Clear the old tree when there are no points.
        if (m_ann_kdTree) {
            delete m_ann_kdTree;
            m_ann_kdTree = nullptr;
        }
        m_ann_point_num = 0;
        return;
    }

    // Ensure sufficient buffer capacity.
    EnsureANNCapacity(pointNum);

    // Copy point coordinates.
    for (int i = 0; i < pointNum; ++i) {
        m_ann_dataPts[i][0] = points[i].x();
        m_ann_dataPts[i][1] = points[i].y();
        m_ann_dataPts[i][2] = points[i].z();
    }
    m_ann_point_num = pointNum;

    // Rebuild the KD-tree for the current point count.
    if (m_ann_kdTree) {
        delete m_ann_kdTree;
        m_ann_kdTree = nullptr;
    }
    m_ann_kdTree = new ANNkd_tree(
        m_ann_dataPts,
        m_ann_point_num,
        m_ann_dim,
        m_particle_ann_bucket_size,
        m_particle_ann_split_rule);
}

void BooleanComputation::BuildOrUpdateANNKdTreeFromCoordinates(
    const double* x,
    int pointNum)
{
    if (pointNum <= 0) {
        delete m_ann_kdTree;
        m_ann_kdTree = nullptr;
        m_ann_point_num = 0;
        return;
    }

    EnsureANNCapacity(pointNum);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < pointNum; ++i) {
        m_ann_dataPts[i][0] = x[3 * i];
        m_ann_dataPts[i][1] = x[3 * i + 1];
        m_ann_dataPts[i][2] = x[3 * i + 2];
    }
    m_ann_point_num = pointNum;

    delete m_ann_kdTree;
    m_ann_kdTree = new ANNkd_tree(
        m_ann_dataPts,
        m_ann_point_num,
        m_ann_dim,
        m_particle_ann_bucket_size,
        m_particle_ann_split_rule);
}

void BooleanComputation::BuildMortonQueryOrder(
    const double* x,
    int pointCount,
    std::vector<int>& queryOrder) const
{
    queryOrder.resize(pointCount);
    for (int i = 0; i < pointCount; ++i) {
        queryOrder[i] = i;
    }
    if (!m_ann_morton_query_order || pointCount < 2) {
        return;
    }

    double minimum[3] = { x[0], x[1], x[2] };
    double maximum[3] = { x[0], x[1], x[2] };
    for (int i = 1; i < pointCount; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate = x[3 * i + axis];
            minimum[axis] = (std::min)(minimum[axis], coordinate);
            maximum[axis] = (std::max)(maximum[axis], coordinate);
        }
    }

    double inverseRange[3];
    for (int axis = 0; axis < 3; ++axis) {
        const double range = maximum[axis] - minimum[axis];
        inverseRange[axis] =
            range > 1e-30 ? 2097151.0 / range : 0.0;
    }

    std::vector<std::uint64_t> codes(pointCount);
    for (int i = 0; i < pointCount; ++i) {
        std::uint64_t code = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(
                    (x[3 * i + axis] - minimum[axis]) *
                    inverseRange[axis]);
            code |= coordinate << (42 - axis * 21);
        }
        codes[i] = code;
    }
    std::sort(
        queryOrder.begin(),
        queryOrder.end(),
        [&codes](int lhs, int rhs) {
            return codes[lhs] < codes[rhs];
        });
}

void BooleanComputation::RefreshParticleNeighborLists(
    const double* x,
    int pointCount,
    double sigma,
    const std::vector<int>& queryOrder)
{
    const auto profileStart = std::chrono::steady_clock::now();
    ++m_profileNeighborRefreshes;
    const auto buildStart = std::chrono::steady_clock::now();
    BuildOrUpdateANNKdTreeFromCoordinates(x, pointCount);
    m_profileParticleTreeBuildSeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - buildStart).count();
    const auto searchStart = std::chrono::steady_clock::now();
    auto refreshOneParticle = [&](int queryIndex) {
        Point queryPt(
            x[queryIndex * 3],
            x[queryIndex * 3 + 1],
            x[queryIndex * 3 + 2]);
        GetclosetNeighorPointsIndices(
            m_ann_kdTree,
            queryPt,
            sigma,
            pointCount,
            m_neighborLists[queryIndex],
            m_particle_neighbor_epsilon);
    };
    if (queryOrder.empty()) {
        for (int queryIndex = 0;
             queryIndex < pointCount;
             ++queryIndex) {
            refreshOneParticle(queryIndex);
        }
    } else {
        for (int queryIndex : queryOrder) {
            refreshOneParticle(queryIndex);
        }
    }
    m_profileParticleSearchSeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - searchStart).count();
    std::copy(
        x,
        x + pointCount * 3,
        m_lastParticleNeighborPositions.begin());
    m_lastParticleNeighborEvaluation = m_iter_cnt;
    m_particleNeighborCacheValid = true;
    if (m_ann_csr_neighbor_storage) {
        BuildParticleNeighborCSR(pointCount);
    }
    m_profileNeighborQuerySeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();
}

void BooleanComputation::RefreshENANeighborTreeIndices(
    const double* x,
    int pointCount,
    const std::vector<int>& queryOrder)
{
    const auto profileStart = std::chrono::steady_clock::now();
    ++m_profileEnaRefreshes;
    auto refreshOneParticle = [&](int queryIndex) {
        Point_3 xi(
            x[queryIndex * 3],
            x[queryIndex * 3 + 1],
            x[queryIndex * 3 + 2]);
        getNearbyTreeIndicesANN(
            xi,
            m_enaNeighborTreeIndices[queryIndex]);
    };
    if (queryOrder.empty()) {
        for (int queryIndex = 0;
             queryIndex < pointCount;
             ++queryIndex) {
            refreshOneParticle(queryIndex);
        }
    } else {
        for (int queryIndex : queryOrder) {
            refreshOneParticle(queryIndex);
        }
    }
    m_profileEnaQuerySeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();
}

void BooleanComputation::BuildParticleNeighborCSR(int pointCount)
{
    m_neighborCSROffsets.resize(pointCount + 1);
    m_neighborCSROffsets[0] = 0;
    for (int i = 0; i < pointCount; ++i) {
        m_neighborCSROffsets[i + 1] =
            m_neighborCSROffsets[i] +
            static_cast<int>(m_neighborLists[i].size());
    }
    m_neighborCSRIndices.resize(m_neighborCSROffsets.back());
    for (int i = 0; i < pointCount; ++i) {
        const int begin = m_neighborCSROffsets[i];
        std::copy(
            m_neighborLists[i].begin(),
            m_neighborLists[i].end(),
            m_neighborCSRIndices.begin() + begin);
    }
}

void BooleanComputation::BuildENATriangleCSR(int pointCount)
{
    m_enaTriangleCSROffsets.resize(pointCount + 1);
    m_enaTriangleCSROffsets[0] = 0;
    for (int i = 0; i < pointCount; ++i) {
        m_enaTriangleCSROffsets[i + 1] =
            m_enaTriangleCSROffsets[i] +
            static_cast<int>(m_enaNearbyTriangles[i].size());
    }
    m_enaTriangleCSRIndices.resize(m_enaTriangleCSROffsets.back());
    for (int i = 0; i < pointCount; ++i) {
        const int begin = m_enaTriangleCSROffsets[i];
        std::copy(
            m_enaNearbyTriangles[i].begin(),
            m_enaNearbyTriangles[i].end(),
            m_enaTriangleCSRIndices.begin() + begin);
    }
}

// Alternative energy-density profile.
// d: distance, sigma: kernel width, area: normalization distance.
double BooleanComputation::e_density_calculation_func(double d, double sigma, double area) {
    if (d <= 0.0) {
        return 0.0;
    }
    // Normalize the argument to [0, 1].
    double t = d / (sigma_rate * sigma);
    //cout << "density: " << std::sin(0.5 * M_PI * std::sqrt(t)) << endl;
    if (t >= 1.0) {
        return 1.0;
    }

    //cout << std::sin(0.5 * M_PI * std::sqrt(t)) << endl;

    // Core density mapping.
    return  max(e_param_bottom, std::sin(0.5 * M_PI * std::sqrt(t)));
}


/**
 * Smootherstep interpolation:
 * f(t) = 6·t^5 ??15·t^4 + 10·t^3
 * The input t must be in [0, 1]; first and second derivatives vanish at both endpoints.
 */
double BooleanComputation::e_density_calculation_func2(double d, double sigma, double area) {
    if (d <= 0.0) return 0.0;

    double t = d / (sigma_rate * sigma);

    if (t >= 1.0) return 1.0;

    double t2 = t * t;           // t^2
    double t3 = t2 * t;          // t^3
    double t4 = t3 * t;          // t^4
    double t5 = t4 * t;          // t^5
    return  max(e_param_bottom, 6.0 * t5 - 15.0 * t4 + 10.0 * t3);
}

int BooleanComputation::get_sample_num() {
    return m_variables.size() / 3;
}

// L-BFGS objective function.
// Parameters:
//   x: current point coordinates, modified in place.
//   g: output gradient.
//   n: total variable count.
//   step: current step length.
// Returns the current energy.
lbfgsfloatval_t BooleanComputation::evaluate(
    lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step)
{
    // Debug observation hooks.
   //cout << x[m_nSamples * 3] << " - " << x[m_nSamples * 3 + 1] << " - " << x[m_nSamples * 3 + 2] << endl;
   //cout << x[3] << " - " << x[4] << " - " << x[5] << endl;
    //for (int i = 0; i < 100; i++)
    //    x points to the same storage as m_variables.

    //cout << "evalute start " << endl;
    m_iter_cnt++;
    double fx_PS = 0.0;
    double fx_NA = 0.0;
    double sigma = m_sigma;  // Particle-system kernel width.

    int totalPoints = m_variables.size() / 3;
    m_particleRho.resize(totalPoints, 1.0);


    // Project particles onto the valid Boolean surface.
    auto profileStart = std::chrono::steady_clock::now();
    if (m_booleanType == 1)  // Union.
    {
#pragma omp parallel for
        for (int i = 0; i < m_variables.size(); i += 3)
        {
            CPoint3D closetP = CPoint3D(x[i], x[i + 1], x[i + 2]);

            closetP = GetCorrectPointInUnionSituation(closetP, m_obbtree1, m_obbtree2);
            x[i] = closetP.x;
            x[i + 1] = closetP.y;
            x[i + 2] = closetP.z;

            m_particleRho[i / 3] = GetDensityAtPoint(closetP);
        }
    }
    else if (m_booleanType == 2)  // Intersection.
    {
#pragma omp parallel for
        for (int i = 0; i < m_variables.size(); i += 3)
        {
            CPoint3D closetP = CPoint3D(x[i], x[i + 1], x[i + 2]);

            closetP = GetCorrectPointInIntersectionSituation(closetP, m_obbtree1, m_obbtree2);
            x[i] = closetP.x;
            x[i + 1] = closetP.y;
            x[i + 2] = closetP.z;

            m_particleRho[i / 3] = GetDensityAtPoint(closetP);
        }
    }
    else if (m_booleanType == 3)  // Subtraction.
    {
#pragma omp parallel for
        for (int i = 0; i < m_variables.size(); i += 3)
        {
            CPoint3D closetP = CPoint3D(x[i], x[i + 1], x[i + 2]);

            closetP = GetCorrectPointInSubtractionSituation(closetP, m_obbtree1, m_obbtree2);
            x[i] = closetP.x;
            x[i + 1] = closetP.y;
            x[i + 2] = closetP.z;

            m_particleRho[i / 3] = GetDensityAtPoint(closetP);
        }
    }
    m_profileProjectionSeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();

    // Inspect projection results.
    //cout << " cast " << x[m_nSamples * 3] << " - " << x[m_nSamples * 3 + 1] << " - " << x[m_nSamples * 3 + 2] << endl;
    //cout << " cast " << x[3] << " - " << x[4] << " - " << x[5] << endl;

    // 2) Build or update the ANN KD-tree while reusing its point buffer.
    const int pointNum = static_cast<int>(m_variables.size() / 3);
    profileStart = std::chrono::steady_clock::now();
    double maxParticleMoveSq = 0.0;
    double maxEnaMoveSq = 0.0;
    if (m_particleNeighborCacheValid) {
        for (int jj = 0; jj < pointNum; ++jj) {
            const int base = jj * 3;
            const double dx =
                x[base] - m_lastParticleNeighborPositions[base];
            const double dy =
                x[base + 1] - m_lastParticleNeighborPositions[base + 1];
            const double dz =
                x[base + 2] - m_lastParticleNeighborPositions[base + 2];
            const double moveSq = dx * dx + dy * dy + dz * dz;
            if (moveSq > maxParticleMoveSq) maxParticleMoveSq = moveSq;
        }
    }
    if (m_enaNeighborCacheValid) {
        for (int jj = 0; jj < pointNum; ++jj) {
            const int base = jj * 3;
            const double dx =
                x[base] - m_lastEnaNeighborPositions[base];
            const double dy =
                x[base + 1] - m_lastEnaNeighborPositions[base + 1];
            const double dz =
                x[base + 2] - m_lastEnaNeighborPositions[base + 2];
            const double moveSq = dx * dx + dy * dy + dz * dz;
            if (moveSq > maxEnaMoveSq) maxEnaMoveSq = moveSq;
        }
    }

    const double particleMoveThreshold =
        0.6 * (5.0 * sigma);
    const double enaMoveThreshold =
        0.6 * (2.0 * m_sigma);
    if (maxParticleMoveSq > m_profileMaxParticleMove) {
        m_profileMaxParticleMove = maxParticleMoveSq;
    }
    if (maxEnaMoveSq > m_profileMaxEnaMove) {
        m_profileMaxEnaMove = maxEnaMoveSq;
    }
    const bool refreshParticleNeighbors =
        !m_particleNeighborCacheValid ||
        m_iter_cnt - m_lastParticleNeighborEvaluation >= 2 ||
        maxParticleMoveSq >
            particleMoveThreshold * particleMoveThreshold;
    const bool refreshEnaNeighbors =
        !m_enaNeighborCacheValid ||
        m_iter_cnt - m_lastEnaNeighborEvaluation >= 2 ||
        maxEnaMoveSq > enaMoveThreshold * enaMoveThreshold;

    // 3) Precompute neighbor lists with ANN. ANN query methods are not
    // thread-safe in this build, so the two indexes are queried serially.
    std::vector<int> queryOrder;
    if (m_ann_morton_query_order) {
        BuildMortonQueryOrder(x, pointNum, queryOrder);
    }
    if (refreshParticleNeighbors) {
        RefreshParticleNeighborLists(
            x, pointNum, sigma, queryOrder);
    } else {
        ++m_profileNeighborReuses;
    }
    if (refreshEnaNeighbors) {
        RefreshENANeighborTreeIndices(
            x, pointNum, queryOrder);
    } else {
        ++m_profileEnaReuses;
    }
    // Neighbor-index preprocessing complete.


    // Compute the total energy.
    // compute energy------
    lbfgsfloatval_t fx = 0.0;
    double constraints_energy = 0.0;
    fx_PS = 0;
    const double inverseTwoSigmaSquared = 0.5 / (sigma * sigma);
    const double inverseFourSigmaSquared = 0.25 / (sigma * sigma);
    // Compute each particle's contribution in parallel.
    profileStart = std::chrono::steady_clock::now();
#pragma omp parallel for reduction(+:fx)
    for (int i = 0; i < m_variables.size(); i += 3)
    {
        // Clear the gradient entries.
        g[i] = g[i + 1] = g[i + 2] = 0.0;


        double delta_fx = 0;
        double delta_g0 = 0.0;
        double delta_g1 = 0.0;
        double delta_g2 = 0.0;


        const double xx = x[i];
        const double yy = x[i + 1];
        const double zz = x[i + 2];
        Point_3 query_point = Point_3(xx, yy, zz);

        // Query the AABB tree for the nearest intersection segment.
                // Store the minimum squared distance.
        double min_sq_dist = std::numeric_limits<double>::infinity();

        if (m_aabb_tree && !m_intersection_segments_flat.empty())
        {
            // AABB_tree::squared_distance returns Kernel::FT.
            Kernel::FT ft = m_aabb_tree->squared_distance(query_point);
            min_sq_dist = CGAL::to_double(ft);
        }

        // Use the default density when there is no intersection tree.
        double density_param = 1.0;
        if (std::isfinite(min_sq_dist))
        {
            // Convert squared distance to Euclidean distance for the density function.
            double min_dist = (min_sq_dist <= 0.0) ? 0.0 : std::sqrt(min_sq_dist);
            density_param = e_density_calculation_func2(min_dist, sigma, 5.0);
            density_param = max(density_param, 0.8);
        }
        else
        {
            // The density parameter remains 1.0 when no special handling is needed.
            density_param = 1.0;
        }
        // -------------------------------------------------------------------------------
        // Compute the particle-system energy.
		//cout << density_param << " -density" << endl;

        const int particle = i / 3;
        const double rho_i = m_particleRho[particle];
        const int* neighborBegin = nullptr;
        const int* neighborEnd = nullptr;
        if (m_ann_csr_neighbor_storage) {
            neighborBegin =
                m_neighborCSRIndices.data() +
                m_neighborCSROffsets[particle];
            neighborEnd =
                m_neighborCSRIndices.data() +
                m_neighborCSROffsets[particle + 1];
        } else {
            const std::vector<int>& neighbors =
                m_neighborLists[particle];
            neighborBegin = neighbors.data();
            neighborEnd = neighborBegin + neighbors.size();
        }

        for (const int* neighbor = neighborBegin;
             neighbor != neighborEnd;
             ++neighbor) {
            const int idx = *neighbor;
            const int base = idx * 3;

            // Read neighbor coordinates directly from x to avoid Point copies.
            const double dx = xx - x[base];
            const double dy = yy - x[base + 1];
            const double dz = zz - x[base + 2];
            const double squaredDistance = dx * dx + dy * dy + dz * dz;

            const double rho_j = m_particleRho[idx];
            const double rho_effective = (rho_i + rho_j) * 0.5;
            //if (rho_effective < 0.8) rho_effective = 0.8;

            // Gaussian kernel.
            const double interParticlesEnergy =
                std::exp(-rho_effective * squaredDistance *
                         inverseFourSigmaSquared);

            // Accumulate energy and gradient; the parallel reduction is race-free.
            delta_fx += interParticlesEnergy;

            const double sqrt_rho = std::sqrt(rho_effective);
            const double gradientScale =
                sqrt_rho * interParticlesEnergy * inverseTwoSigmaSquared;

            delta_g0 -= gradientScale * dx;
            delta_g1 -= gradientScale * dy;
            delta_g2 -= gradientScale * dz;
        }


        delta_fx *= density_param;
        delta_g0 *= density_param;
        delta_g1 *= density_param;
        delta_g2 *= density_param;
        
        /*fx += delta_fx;
        g[i] += delta_g0;
        g[i + 1] += delta_g1;
        g[i + 2] += delta_g2;*/

        fx_PS = delta_fx;

        fx += lambda_PS * delta_fx;
        g[i] += lambda_PS * delta_g0;
        g[i + 1] += lambda_PS * delta_g1;
        g[i + 2] += lambda_PS * delta_g2;
    }
    m_profileNeighborEnergySeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();

    // Compute the E_NA energy.
    fx_NA = 0.0;
    const int particleCount = static_cast<int>(m_variables.size() / 3);
    if (refreshEnaNeighbors) {
        const auto mapStart = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < particleCount; ++i) {
            mapNearbyTreeIndicesToTriangles(
                m_enaNeighborTreeIndices[i],
                m_enaNearbyTriangles[i]);
        }
        m_profileEnaMapSeconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - mapStart).count();
        if (m_ann_csr_neighbor_storage) {
            BuildENATriangleCSR(particleCount);
        }
        std::copy(
            x,
            x + particleCount * 3,
            m_lastEnaNeighborPositions.begin());
        m_lastEnaNeighborEvaluation = m_iter_cnt;
        m_enaNeighborCacheValid = true;
    }
    m_profileEnaQuerySeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();
    
    double r_NA = 2.0 * m_sigma;
    double r2 = r_NA * r_NA;



    const double ac_weights[6] = { 1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0,
                                  9.0 / 30.0, 9.0 / 30.0, 9.0 / 30.0 };

    profileStart = std::chrono::steady_clock::now();
    double totalEnaEnergy = 0.0;
#pragma omp parallel for schedule(static) reduction(+:totalEnaEnergy)
     for (int i = 0; i < particleCount; ++i) {
         CPoint3D xi(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);

         double delta_g0_NA = 0.0, delta_g1_NA = 0.0, delta_g2_NA = 0.0;
         double particleE_NA = 0.0;

         const int* triangleBegin = nullptr;
         const int* triangleEnd = nullptr;
         if (m_ann_csr_neighbor_storage) {
             triangleBegin =
                 m_enaTriangleCSRIndices.data() +
                 m_enaTriangleCSROffsets[i];
             triangleEnd =
                 m_enaTriangleCSRIndices.data() +
                 m_enaTriangleCSROffsets[i + 1];
         } else {
             const std::vector<int>& nearbyTriangles =
                 m_enaNearbyTriangles[i];
             triangleBegin = nearbyTriangles.data();
             triangleEnd = triangleBegin + nearbyTriangles.size();
         }
         for (const int* triangle = triangleBegin;
              triangle != triangleEnd;
              ++triangle) {
             const int triIdx = *triangle;
             const ENAFaceCache& face = m_enaFaceCache[triIdx];
             const double normalDotXi =
                 face.normalX * xi.x +
                 face.normalY * xi.y +
                 face.normalZ * xi.z;

            for (int k = 0; k < 6; ++k) {
                const double proj =
                    normalDotXi - face.sampleProjections[k];
                 particleE_NA +=
                    ac_weights[k] * face.area * proj * proj;

                 // Add the gradient contribution.
                 const double weight =
                    2.0 * ac_weights[k] * face.area * proj;
                 delta_g0_NA += weight * face.normalX;
                 delta_g1_NA += weight * face.normalY;
                 delta_g2_NA += weight * face.normalZ;
             }
         }
         totalEnaEnergy += particleE_NA;
         g[i*3] += lambda_NA * delta_g0_NA;
         g[i*3+1] += lambda_NA * delta_g1_NA;
         g[i*3+2] += lambda_NA * delta_g2_NA;
         }
    fx_NA = totalEnaEnergy;
    m_profileEnaEnergySeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStart).count();
         

         

         //Scale the normal consistency energy to the same level as the particle system energy at the first iteration, and then only update the particle system energy by multiplying it with lambdaForBalance in the following iterations.
         if (m_iter_cnt == 1)
         {
             lambda_NA = fx_NA > 1e-30 ? fx_PS / fx_NA : 1.0;
             lambda_NA *= m_lambda_ofnoraml_penalty;
             lambda_balance = 0.98;
         }
         if (m_iter_cnt >= 2)
         {
             lambda_PS = lambda_PS * lambda_balance;
         }
         fx += lambda_NA * fx_NA;
         if (IterationConsoleOutputEnabled()) {
             cout << "粒子系统能量??fx_NA: " << fx_NA * lambda_NA << endl;
             //cout << "evalute return fx: " << fx << endl;
             cout << "lambda_NA: " << lambda_NA << endl;
             cout<<" lambda_PS: "<<lambda_PS<<endl;
         }
         // Particle-energy rebound check.
         if (fx < m_minEtot)
         {
             m_minEtot = fx;
         }
          if (fx_PS * lambda_PS > 1.05 * m_minEtot ||
              lambda_PS < 0.145)
         {
             for (int i = 0; i < m_variables.size(); ++i)
             {
                 g[i] = 0.0;
             }
         }

    //cout << "evalute return fx: " << fx << endl;

    return fx;
}

// Generate initial surface samples with area-weighted random sampling.
// nSamples: requested number of samples.
// Returns the sampled 3D points.
vector<Point> BooleanComputation::GetInitialSamplesOnSurface(int nSamples)
{
    // Compute triangle areas.
    vector<double> areaList;
    for (int i = 0; i < m_inputModel[0].GetNumOfFaces(); ++i)
    {
        CPoint3D vec1 = m_inputModel[0].Vert(m_inputModel[0].Face(i)[1]) - m_inputModel[0].Vert(m_inputModel[0].Face(i)[0]);
        CPoint3D vec2 = m_inputModel[0].Vert(m_inputModel[0].Face(i)[2]) - m_inputModel[0].Vert(m_inputModel[0].Face(i)[1]);
        double area = (vec1 * vec2).Len() * 0.5;  // Triangle area.

        auto face = m_inputModel[0].Face(i);
        CPoint3D center = (m_inputModel[0].Vert(face[0]) +
            m_inputModel[0].Vert(face[1]) +
            m_inputModel[0].Vert(face[2])) / 3.0;
        double rho = GetDensityAtPoint(center);
        
		area *= rho;  // Weight area by curvature density.
        areaList.push_back(area);
    }

    // Build the cumulative-area table.
    double before = 0;
    for (int i = 0; i < areaList.size(); ++i)
    {
        areaList[i] += before;
        before = areaList[i];
    }

    // Generate samples.
    vector<Point> samplePoints;
    // Debug output for generated samples.
    //ofstream outputInitialPoint("Intersection//initla_points_1");
    for (int i = 0; i < nSamples; ++i)
    {
        // Select a triangle with probability proportional to its area.
        double areaQuery = rand() / (double)RAND_MAX * areaList.back();
        int faceID = lower_bound(areaList.begin(), areaList.end(), areaQuery) - areaList.begin();
        if (faceID >= m_inputModel[0].GetNumOfFaces())
            faceID -= 1;

        // Generate random barycentric coordinates.
        double alpha = rand() / (double)RAND_MAX + 1e-6;
        double beta = rand() / (double)RAND_MAX + 1e-6;
        double gama = rand() / (double)RAND_MAX + 1e-6;

        // Normalize barycentric coordinates.
        double sum = alpha + beta + gama;
        alpha /= sum;
        beta /= sum;
        gama /= sum;

        // Interpolate the sample position from barycentric coordinates.
        CPoint3D sample =
            m_inputModel[0].Vert(m_inputModel[0].Face(faceID)[0]) * alpha +
            m_inputModel[0].Vert(m_inputModel[0].Face(faceID)[1]) * beta +
            m_inputModel[0].Vert(m_inputModel[0].Face(faceID)[2]) * gama;

        // Debug sample output.
        //outputInitialPoint << sample << endl;
        samplePoints.push_back(Point(sample.x, sample.y, sample.z));
    }
    // Write all samples to the debug file.
    //ofstream outputInitialPoints("Intersection//initla_points_2.xyz");
    //for (auto it : samplePoints)
    //    outputInitialPoints << it << endl;
    //outputInitialPoints.close();
    //outputInitialPoint.close();
    return samplePoints;
}

// Filter samples for union. A point is valid when it is not inside the other solid.
// This deliberately accepts some boundary noise because surface sampling is approximate.
// Intersection-curve points are handled separately and are not removed here.
vector<Point> BooleanComputation::GetValidSamplesByUnion(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
    vector<Point> res;
    for (auto& p : InitialSamples) {
        CPoint3D cdot = CPoint3D(p.x(), p.y(), p.z());
        if (IsOnSurface(cdot, obbtree1, m_inputModel[1]) == 1 || IsOnSurface(cdot, obbtree2, m_inputModel[2]) == 1) {
            res.push_back(p);
        }
    }
    return res;
}


// Filter samples for intersection. A point is valid when it lies inside or on the other solid.
// A small tolerance is retained because surface sampling is approximate.
vector<Point> BooleanComputation::GetValidSamplesByIntersection(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
    vector<Point> res;
    for (auto& p : InitialSamples) {
        CPoint3D cdot = CPoint3D(p.x(), p.y(), p.z());
        if (IsOnSurface(cdot, obbtree1, m_inputModel[1]) == -1 || IsOnSurface(cdot, obbtree2, m_inputModel[2]) == -1) {
            res.push_back(p);
        }
    }
    return res;
}


// Filter samples for subtraction A - B.
// Keep A-surface points outside B and B-surface points inside A.
vector<Point> BooleanComputation::GetValidSamplesBySubtraction(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
    vector<Point> res;
    for (auto& p : InitialSamples) {
        CPoint3D cdot = CPoint3D(p.x(), p.y(), p.z());
        // Classify points on model A.
        if (IsOnSurface(cdot, obbtree1, m_inputModel[1]) == 0) {
            if (IsOnSurface(cdot, obbtree2, m_inputModel[2]) != -1)
                res.push_back(p);
        }   // Classify points on model B.
        else if (IsOnSurface(cdot, obbtree2, m_inputModel[2]) == 0) {
            if (IsOnSurface(cdot, obbtree1, m_inputModel[1]) == -1)
                res.push_back(p);
        }
    }
    return res;
}


// Compute total model surface area.
double BooleanComputation::GetSurfaceArea()const
{
    double totalArea = 0;
    for (int i = 0; i < m_inputModel[0].GetNumOfFaces(); ++i)
    {
        CPoint3D vec1 = m_inputModel[0].Vert(m_inputModel[0].Face(i)[1]) - m_inputModel[0].Vert(m_inputModel[0].Face(i)[0]);
        CPoint3D vec2 = m_inputModel[0].Vert(m_inputModel[0].Face(i)[2]) - m_inputModel[0].Vert(m_inputModel[0].Face(i)[1]);
        double area = (vec1 * vec2).Len() * 0.5;  // Triangle area.
        totalArea += area;
    }
    return totalArea;
}





// Return the unit normal of a triangle.
// triangleID: triangle index.
// m_inputModel: source mesh.
CPoint3D BooleanComputation::GetNormalOfPointIntriangle(int triangleID, CBaseModel& m_inputModel)const
{
    // Read the three triangle vertices.
    CPoint3D p0 = m_inputModel.Vert(m_inputModel.Face(triangleID)[0]);
    CPoint3D p1 = m_inputModel.Vert(m_inputModel.Face(triangleID)[1]);
    CPoint3D p2 = m_inputModel.Vert(m_inputModel.Face(triangleID)[2]);

    // Compute two edge vectors.
    CPoint3D vec1 = CPoint3D(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    CPoint3D vec2 = CPoint3D(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

    // Compute the cross-product normal.
    Point normal = Point(vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x);

    // Normalize the normal.
    double normalLen = sqrt(normal.x() * normal.x() + normal.y() * normal.y() + normal.z() * normal.z());
    return CPoint3D(normal.x() / normalLen, normal.y() / normalLen, normal.z() / normalLen);
}

// Return neighbors of a query point using a KD-tree.
// kdTree: built KD-tree.
// queryPoint: query position.
// sigma: kernel scale used to derive the search radius.
// pointsForKdTree: source point set.
vector<Point> BooleanComputation::GetclosetNeighorPoints(ANNkd_tree* kdTree, Point queryPoint,
    double sigma, vector<Point>& pointsForKdTree) const
{
    vector<Point> queriedPoint;
    if (!kdTree) return queriedPoint;

    // Allocate and fill the ANN query point.
    ANNpoint queryPt = annAllocPt(3);
    queryPt[0] = queryPoint.x();
    queryPt[1] = queryPoint.y();
    queryPt[2] = queryPoint.z();

    // First count-only query to determine the neighbor count.
    ANNidx* tmpIdx = new ANNidx[1];
    ANNdist* tmpDists = new ANNdist[1];
    int pointNumInK = kdTree->annkFRSearch(
        queryPt,
        25 * sigma * sigma,
        0,
        tmpIdx,
        tmpDists,
        0);

    // Release temporary query arrays.
    delete[] tmpIdx;
    delete[] tmpDists;

    if (pointNumInK <= 0) {
        annDeallocPt(queryPt);
        return queriedPoint;
    }

    // Second query: allocate index and distance arrays for the reported count.
    std::vector<ANNidx> nnIdx(pointNumInK);
    std::vector<ANNdist> dists(pointNumInK);
    kdTree->annkFRSearch(
        queryPt,
        25 * sigma * sigma,
        pointNumInK,
        nnIdx.data(),
        dists.data(),
        0);

    // Collect neighbor points.
    queriedPoint.reserve(pointNumInK);
    for (int i = 0; i < pointNumInK; ++i) {
        int idx = nnIdx[i];
        if (idx >= 0 && idx < (int)pointsForKdTree.size()) {
            queriedPoint.push_back(pointsForKdTree[idx]);
        }
    }

    // Release the query point.
    annDeallocPt(queryPt);

    return queriedPoint;
}

// Return neighbor indices instead of copying Point objects.
void BooleanComputation::GetclosetNeighorPointsIndices(
    ANNkd_tree* kdTree,
    const Point& queryPoint,
    double sigma,
    int pointCount,
    std::vector<int>& indices,
    double eps) const
{
    indices.clear();
    if (!kdTree) return;

    // Allocate and initialize the ANN query point.
    ANNcoord queryPt[3] = {
        queryPoint.x(), queryPoint.y(), queryPoint.z()
    };

    int kcount = 0;
    if (m_ann_single_pass_query) {
        int queryCapacity =
            (std::max)(32, static_cast<int>(indices.capacity()));
        indices.resize(queryCapacity);
        kcount = kdTree->annkFRSearch(
            queryPt,
            25 * sigma * sigma,
            queryCapacity,
            indices.data(),
            nullptr,
            eps);
        if (kcount > queryCapacity) {
            indices.resize(kcount);
            kdTree->annkFRSearch(
                queryPt,
                25 * sigma * sigma,
                kcount,
                indices.data(),
                nullptr,
                eps);
        }
    } else {
        // Reference path: count first, then repeat the query to fill results.
        kcount = kdTree->annkFRSearch(
            queryPt,
            25 * sigma * sigma,
            0,
            nullptr,
            nullptr,
            eps);
        if (kcount > 0) {
            indices.resize(kcount);
            kdTree->annkFRSearch(
                queryPt,
                25 * sigma * sigma,
                kcount,
                indices.data(),
                nullptr,
                eps);
        }
    }
    if (kcount <= 0) {
        return;
    }

    // ANN writes directly into the target integer vector.
    int validCount = 0;
    const int filledCount =
        (std::min)(kcount, static_cast<int>(indices.size()));
    for (int i = 0; i < filledCount; ++i) {
        if (indices[i] >= 0 && indices[i] < pointCount) {
            indices[validCount++] = indices[i];
        }
    }
    indices.resize(validCount);

}

// Build a KD-tree.
// dim: point dimension.
// pointNum: number of points.
// points: source point set.
ANNkd_tree* BooleanComputation::GetKDTree(int dim, int pointNum, vector <Point>& points)const
{
    // Allocate the point buffer.
    ANNpointArray dataPts = annAllocPts(pointNum, dim); // allocate data points

    // Fill the point buffer.
    for (int i = 0; i < pointNum; i++)
    {
        dataPts[i][0] = points[i].x();
        dataPts[i][1] = points[i].y();
        dataPts[i][2] = points[i].z();
    }

    // Construct the KD-tree.
    ANNkd_tree* kdTree = new ANNkd_tree(dataPts, pointNum, dim);
    return kdTree;
}

// Classify a point relative to a surface.
// Returns 0 on the surface, 1 outside, and -1 inside.
int BooleanComputation::IsOnSurface(CPoint3D m_p, Distance_OBB& obbtree, CBaseModel& inputModel)
{
    // Query the closest surface point and triangle.
    Distance_OBB::QueryResult queryResult = obbtree.Query(m_p);
    CPoint3D closePnt = queryResult.closestPnt;

    // Compute the vector to the closest point.
    CPoint3D vecOfPAndCloseP = (m_p - closePnt);
    //vecOfPAndCloseP.Normalize();

    // Get the closest triangle's normal.
    CPoint3D normalOfModelTraingleOfP =
        GetNormalOfPointIntriangle(queryResult.triangleID, inputModel);
    normalOfModelTraingleOfP.Normalize();

    // Determine the side of the surface.
    if (abs(normalOfModelTraingleOfP ^ vecOfPAndCloseP) <= 0.0001)
        return 0;    // On the surface.
    else if ((normalOfModelTraingleOfP ^ vecOfPAndCloseP) > 0)
        return 1;    // Outside.
    else
        return -1;   // Inside.
}

// Project a point for union reconstruction.
// The projected point should remain valid for at least one operand.
CPoint3D BooleanComputation::GetCorrectPointInUnionSituation(const CPoint3D& m_p,
    Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
    // Query the closest points on both operands.
    Distance_OBB::QueryResult queryResult1 = obbtree1.Query(m_p);
    Distance_OBB::QueryResult queryResult2 = obbtree2.Query(m_p);
    CPoint3D closePnt;

    //struct QueryResult {
    //    int triangleID;
    //    double distance;
    //    CPoint3D closestPnt; };

    // Start with the nearer projection.
    if ((m_p - queryResult1.closestPnt).Len() >
        (m_p - queryResult2.closestPnt).Len())
    {
        closePnt = queryResult2.closestPnt;
        // Project to the other model when the current projection is inside it.
        if (IsOnSurface(closePnt, obbtree1, m_inputModel[1]) == -1)
        {
            closePnt = obbtree1.Query(m_p).closestPnt;
        }
    }
    else
    {
        closePnt = queryResult1.closestPnt;
        // Project to the other model when the current projection is inside it.
        if (IsOnSurface(closePnt, obbtree2, m_inputModel[2]) == -1)
        {
            closePnt = obbtree2.Query(m_p).closestPnt;
        }
    }


    return closePnt;
}

// Project a point for intersection reconstruction.
// The result should remain valid for both operands.
CPoint3D BooleanComputation::GetCorrectPointInIntersectionSituation(const CPoint3D& m_p,
    Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
    // Query the closest points on both operands.
    Distance_OBB::QueryResult queryResult1 = obbtree1.Query(m_p);
    Distance_OBB::QueryResult queryResult2 = obbtree2.Query(m_p);
    CPoint3D closePnt;

    // Start with the nearer projection.
    if ((m_p - queryResult1.closestPnt).Len() >
        (m_p - queryResult2.closestPnt).Len())
    {
        closePnt = queryResult2.closestPnt;
        // Project to the other model when the current projection is outside it.
        if (IsOnSurface(closePnt, obbtree1, m_inputModel[1]) == 1)
        {
            closePnt = obbtree1.Query(m_p).closestPnt;
        }
    }
    else
    {
        closePnt = queryResult1.closestPnt;
        // Project to the other model when the current projection is outside it.
        if (IsOnSurface(closePnt, obbtree2, m_inputModel[2]) == 1)
        {
            closePnt = obbtree2.Query(m_p).closestPnt;
        }
    }
    return closePnt;
}

// Project a point for subtraction A - B.
// The result should lie inside A and outside B.
CPoint3D BooleanComputation::GetCorrectPointInSubtractionSituation(const CPoint3D& m_p,
    Distance_OBB& obbtree1, Distance_OBB& obbtree2)
{
	// Original implementation retained for reference; it appeared incorrect.
    
    //// First project onto model A.
    //Distance_OBB::QueryResult queryResult1 = obbtree1.Query(m_p);
    //CPoint3D closePnt = queryResult1.closestPnt;

    //// Handle points that fall inside model B.
    //if (IsOnSurface(closePnt, obbtree2, m_inputModel[2]) == -1)
    //{
    //    // First approach: simple projection.
    //    Distance_OBB::QueryResult queryResult2 = obbtree2.Query(closePnt);
    //    closePnt = queryResult2.closestPnt;
    //    closePnt = obbtree1.Query(closePnt).closestPnt;

    //}
    //return closePnt;
    // 
    // -------------------------------------
    
    // Current implementation.
    
    // Query the closest points on both operands.
    Distance_OBB::QueryResult queryResult1 = obbtree1.Query(m_p);
    Distance_OBB::QueryResult queryResult2 = obbtree2.Query(m_p);
    CPoint3D closePnt;

    // Start with the nearer projection.
    if ((m_p - queryResult1.closestPnt).Len() >
        (m_p - queryResult2.closestPnt).Len())
    {
        // Select model B.
        closePnt = queryResult2.closestPnt;
        // Project to A when the current point is outside A.
        if (IsOnSurface(closePnt, obbtree1, m_inputModel[1]) == 1)
        {
            closePnt = obbtree1.Query(m_p).closestPnt;
        }
    }
    else
    {
		// Select model A.
        closePnt = queryResult1.closestPnt;
        // Project to B when the current point is inside B.
        if (IsOnSurface(closePnt, obbtree2, m_inputModel[2]) == -1)
        {
            closePnt = obbtree2.Query(m_p).closestPnt;
        }
    }
    return closePnt;
}

// Compute the distance from a point to the nearest surface point.
double BooleanComputation::GetPointToMeshSurface(const CPoint3D& m_p,
    Distance_OBB& obbtree)
{
    // Project onto the model.
    Distance_OBB::QueryResult queryResult = obbtree.Query(m_p);
    CPoint3D closePnt = queryResult.closestPnt;

    // Compute the Euclidean distance.
    double dx = m_p.x - closePnt.x;
    double dy = m_p.y - closePnt.y;
    double dz = m_p.z - closePnt.z;
    double dis = std::sqrt(dx * dx + dy * dy + dz * dz);

    return dis;
}



// Compute vertex-to-target-mesh distances, map them to [0, 1], and write an OBJ.
// Vertex records use "v x y z".
// Texture records use "vt gray 0", where gray is in [0, 1].
// Face records use "f v1/vt1 v2/vt2 v3/vt3" with zero-based source indices.
// The function prints min/max/mean error for debugging.
// Returns true when the file is written successfully.
bool BooleanComputation::ErrorToMeshSurface(
    const CBaseModel& input,
    Distance_OBB& obbtree2,
    double bound_min,                    // Lower error-mapping bound.
    double bound_max,                    // Upper error-mapping bound.
    const std::string& outObjFilename)
{
    // Validate the input.
    int numVertices = input.GetNumOfVerts();
    if (numVertices <= 0) {
        std::cerr << "[ErrorToMeshSurfaceAndWriteOBJ_WithVT] 输入模型顶点数为 0，取消操作。\n";
        return false;
    }

    // Preallocate the per-vertex error array.
    std::vector<double> perVertexErrors;
    perVertexErrors.reserve(numVertices);

    // Use Kahan summation for the mean.
    long double sum = 0.0L;
    long double c = 0.0L;
    double maxErr = 0.0;
    double minErr = std::numeric_limits<double>::infinity();

    // Compute per-vertex distances.
    for (int i = 0; i < numVertices; ++i) {
        const CPoint3D& v = input.Vert(i);

        // Query the closest surface point through Distance_OBB.
        Distance_OBB::QueryResult qr = obbtree2.Query(v);
        CPoint3D closeP = qr.closestPnt;

        double dx = v.x - closeP.x;
        double dy = v.y - closeP.y;
        double dz = v.z - closeP.z;

         double d = std::sqrt(dx*dx + dy*dy + dz*dz);

        perVertexErrors.push_back(d);

        // Kahan summation.
        long double y = (long double)d - c;
        long double t = sum + y;
        c = (t - sum) - y;
        sum = t;

        if (d > maxErr) maxErr = d;
        if (d < minErr) minErr = d;
    }

    double meanErr = static_cast<double>(sum / (long double)numVertices);
    if (!std::isfinite(minErr)) minErr = 0.0; // Defensive fallback.
    std::cout << "[ErrorToMeshSurfaceAndWriteOBJ_WithVT] computed errors -> min: "
        << minErr << " , mean: " << meanErr << " , max: " << maxErr
        << " , vertices: " << numVertices << std::endl;

    // A very small maximum error means the meshes nearly coincide.
    // Adjust bound_min/bound_max to control the normalized visualization.

    std::ofstream ofs(outObjFilename);
    if (!ofs.is_open()) {
        std::cerr << "[ErrorToMeshSurfaceAndWriteOBJ_WithVT] 无法打开输出文件: " << outObjFilename << std::endl;
        return false;
    }
    ofs << std::fixed << std::setprecision(8);

    // Compute the mapping denominator.
    double denom = bound_max - bound_min;
    bool denom_is_zero = (std::abs(denom) < 1e-20);

    ofs << "mtllib texture.mtl" << endl;

    // 1) Write vertex positions.
    for (int i = 0; i < numVertices; ++i) {
        const CPoint3D& v = input.Vert(i);
        ofs << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    // 2) Write texture coordinates "vt gray 0"; gray is stored in u.
    // OBJ texture indices are one-based, with one texture entry per vertex.
    for (int i = 0; i < numVertices; ++i) {
        double e = perVertexErrors[i];
        double gray = 0.0;
        if (denom_is_zero) {
            // When the bounds are equal, map values to 0 or 1.
            gray = (e <= bound_min) ? 0.0 : 1.0;
        }
        else {
            gray = (e - bound_min) / denom;
            if (gray < 0.0) gray = 0.0;
            if (gray > 1.0) gray = 1.0;
        }
        // Store gray in u and set v to zero.
        ofs << "vt " << min(max(gray, 0.001), 0.999)  << " " << 0.0 << "\n";
    }

    // 3) Write faces using the shared vertex/texture indices.
    int numFaces = input.GetNumOfFaces();
    for (int fi = 0; fi < numFaces; ++fi) {
        auto face = input.Face(fi); // Triangle faces have three indices.
        // OBJ indices are one-based.
        int a = face[0] + 1;
        int b = face[1] + 1;
        int c = face[2] + 1;
        // Vertex i and texture-coordinate i share the same one-based index.
        ofs << "f " << a << "/" << a << " " << b << "/" << b << " " << c << "/" << c << "\n";
    }

    ofs.close();
    std::cout << "[ErrorToMeshSurfaceAndWriteOBJ_WithVT] 已写??OBJ（含 vt?? " << outObjFilename << std::endl;
    return true;
}


// Initialize m_faceValidForENA for the E_NA integral.
void BooleanComputation::initFacesForENA()
{
    int numFaces = m_inputModel[0].GetNumOfFaces();
    m_faceValidForENA.resize(numFaces, false);

    // Obtain the vertex count of model A for triangle ownership.
    int numVertsA = m_inputModel[1].GetNumOfVerts();
    const int numVerts = m_inputModel[0].GetNumOfVerts();
    std::vector<char> vertexValid(numVerts, 0);
#pragma omp parallel for schedule(static)
    for (int vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
        const int otherModelId = (vertIdx < numVertsA) ? 2 : 1;
        vertexValid[vertIdx] =
            isVertexValidForENA(m_inputModel[0].Vert(vertIdx), otherModelId);
    }

    int validCount = 0;

    for (int i = 0; i < numFaces; ++i) {
        auto face = m_inputModel[0].Face(i);
        int v0_idx = face[0];
        int v1_idx = face[1];
        int v2_idx = face[2];

        // Determine which operand owns the triangle.
        // Indices below numVertsA belong to A; the remaining vertices belong to B.
        bool fromA = (v0_idx < numVertsA || v1_idx < numVertsA || v2_idx < numVertsA);

        CPoint3D v0 = m_inputModel[0].Vert(v0_idx);
        CPoint3D v1 = m_inputModel[0].Vert(v1_idx);
        CPoint3D v2 = m_inputModel[0].Vert(v2_idx);

        bool v0_valid = false, v1_valid = false, v2_valid = false;

        if (fromA) {
            // Triangles from A are tested against model B.
            v0_valid = vertexValid[v0_idx] != 0;
            v1_valid = vertexValid[v1_idx] != 0;
            v2_valid = vertexValid[v2_idx] != 0;
        }
        else {
            // Triangles from B are tested against model A.
            v0_valid = vertexValid[v0_idx] != 0;
            v1_valid = vertexValid[v1_idx] != 0;
            v2_valid = vertexValid[v2_idx] != 0;
        }

        // Include the triangle when at least one vertex is valid.
        if (v0_valid || v1_valid|| v2_valid) {
            m_faceValidForENA[i] = true;
            validCount++;
        }
    }

    cout << "E_NA 积分区域筛?? " << validCount << " / " << numFaces << " 个三角形有效" << endl;
}

bool BooleanComputation::isVertexValidForENA(const CPoint3D& v, int otherModelId)
{
    // otherModelId: 1 tests against model A; 2 tests against model B.
    Distance_OBB* obbtree = (otherModelId == 1) ? &m_obbtree1 : &m_obbtree2;
    CBaseModel* model = (otherModelId == 1) ? &m_inputModel[1] : &m_inputModel[2];

    int status = IsOnSurface(v, *obbtree, *model);

    if (m_booleanType == 1) {  // Union.
        // Points outside or on the other model are visible.
        return (status == 1 || status == 0);
    }
    else if (m_booleanType == 2) {  // Intersection.
        // Points inside or on the other model are visible.
        return (status == -1 || status == 0);
    }
    else if (m_booleanType == 3) {  // Subtraction A - B.
        if (otherModelId == 2) {
            // A-surface points use B: keep points outside or on B.
            return (status == 1 || status == 0);
        }
        else {
            // B-surface points use A: keep points inside or on A.
            return (status == -1 || status == 0);
        }
    }

    return false;
}


void BooleanComputation::buildVertexToTrianglesMap()
{
    int numVerts = m_inputModel[0].GetNumOfVerts();
    int numFaces = m_inputModel[0].GetNumOfFaces();

    // Initialize an empty triangle list for every vertex.
    m_vertToTris.clear();
    m_vertToTris.resize(numVerts);

    // Visit every triangle.
    for (int triIdx = 0; triIdx < numFaces; ++triIdx) {
        auto face = m_inputModel[0].Face(triIdx);

        // Register the triangle with each incident vertex.
        m_vertToTris[face[0]].push_back(triIdx);
        m_vertToTris[face[1]].push_back(triIdx);
        m_vertToTris[face[2]].push_back(triIdx);
    }

    // Store the same adjacency in CSR form for cache-friendly lookup from the
    // serial ANN query results.
    m_vertToTriOffsets.resize(numVerts + 1, 0);
    for (int vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
        m_vertToTriOffsets[vertIdx + 1] =
            m_vertToTriOffsets[vertIdx] +
            static_cast<int>(m_vertToTris[vertIdx].size());
    }
    m_vertToTriIndices.resize(m_vertToTriOffsets.back());
    std::vector<int> cursor(
        m_vertToTriOffsets.begin(),
        m_vertToTriOffsets.end() - 1);
    for (int vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
        for (int triIdx : m_vertToTris[vertIdx]) {
            m_vertToTriIndices[cursor[vertIdx]++] = triIdx;
        }
    }

    // Print adjacency statistics.
    int totalRefs = 0;
    int maxRefs = 0;
    for (const auto& tris : m_vertToTris) {
        int cnt = (int)tris.size();
        totalRefs += cnt;
        if (cnt > maxRefs) maxRefs = cnt;
    }

    std::cout << "顶点-三角形映射构建完?? 顶点??" << numVerts
        << ", 三角形数=" << numFaces
        << ", 总引用数=" << totalRefs << " (应为 " << numFaces * 3 << ")"
        << ", 最大三角形??顶点=" << maxRefs << std::endl;

}

void BooleanComputation::testVertexMapping(int testVertexIndex)
{
    if (testVertexIndex < 0 || testVertexIndex >= (int)m_vertToTris.size()) {
        std::cout << "顶点索引 " << testVertexIndex << " 超出范围 (0-" << m_vertToTris.size() - 1 << ")" << std::endl;
        return;
    }

    const auto& triIndices = m_vertToTris[testVertexIndex];

    std::cout << "========== 顶点 " << testVertexIndex << " 的三角形信息 ==========" << std::endl;
    std::cout << "该顶点属??" << triIndices.size() << " 个三角形" << std::endl;

    // Read the vertex position.
    CPoint3D vertPos = m_inputModel[0].Vert(testVertexIndex);
    std::cout << "顶点坐标: (" << vertPos.x << ", " << vertPos.y << ", " << vertPos.z << ")" << std::endl;

    // Print details for every incident triangle.
    for (int i = 0; i < (int)triIndices.size(); ++i) {
        int triIdx = triIndices[i];
        auto face = m_inputModel[0].Face(triIdx);

        std::cout << "  三角??" << i + 1 << ": 索引=" << triIdx
            << ", 顶点=[" << face[0] << "," << face[1] << "," << face[2] << "]";


    }
}

std::set<int> BooleanComputation::getNearbyTriangles(const Point_3& xi) const
{
    std::set<int> triangleSet;

    int numFaces = m_inputModel[0].GetNumOfFaces();
    double searchRadius = 2.0 * m_sigma / 5.0;
    double searchRadiusSq = searchRadius * searchRadius;
    double xi_x = xi.x(), xi_y = xi.y(), xi_z = xi.z();

    for (int triIdx = 0; triIdx < numFaces; ++triIdx) {
        // Test ENA validity first.
        if (! m_faceValidForENA[triIdx]) continue;

        auto face = m_inputModel[0].Face(triIdx);

        // Read vertex positions.
        CPoint3D v0 = m_inputModel[0].Vert(face[0]);
        CPoint3D v1 = m_inputModel[0].Vert(face[1]);
        CPoint3D v2 = m_inputModel[0].Vert(face[2]);

        // Compute squared distances.
        double d0 = (xi_x - v0.x) * (xi_x - v0.x) +
            (xi_y - v0.y) * (xi_y - v0.y) +
            (xi_z - v0.z) * (xi_z - v0.z);
        if (d0 < searchRadiusSq) {
            triangleSet.insert(triIdx);
            continue;
        }

        double d1 = (xi_x - v1.x) * (xi_x - v1.x) +
            (xi_y - v1.y) * (xi_y - v1.y) +
            (xi_z - v1.z) * (xi_z - v1.z);
        if (d1 < searchRadiusSq) {
            triangleSet.insert(triIdx);
            continue;
        }

        double d2 = (xi_x - v2.x) * (xi_x - v2.x) +
            (xi_y - v2.y) * (xi_y - v2.y) +
            (xi_z - v2.z) * (xi_z - v2.z);
        if (d2 < searchRadiusSq) {
            triangleSet.insert(triIdx);
            continue;
        }
    }

    return triangleSet;
}

Point_3 BooleanComputation::toPoint3(const CPoint3D& p) const
{
    return Point_3(p.x, p.y, p.z);
}

std::array<CPoint3D, 6> BooleanComputation::getSamplesOnTriangle(int triangleIdx) const
{
    std::array<CPoint3D, 6> samplesOnTriangle;

    auto face = m_inputModel[0].Face(triangleIdx);
    CPoint3D v0 = m_inputModel[0].Vert(face[0]);
    CPoint3D v1 = m_inputModel[0].Vert(face[1]);
    CPoint3D v2 = m_inputModel[0].Vert(face[2]);

    // Midpoints of the three edges.
    samplesOnTriangle[0] = v0 * 0.5 + v1 * 0.5;
    samplesOnTriangle[1] = v0 * 0.5 + v2 * 0.5;
    samplesOnTriangle[2] = v1 * 0.5 + v2 * 0.5;

    // Three interior sample points.
    samplesOnTriangle[3] = v0 * (1 / 6.0) + v1 * (1 / 6.0) + v2 * (2 / 3.0);
    samplesOnTriangle[4] = v0 * (1 / 6.0) + v2 * (1 / 6.0) + v1 * (2 / 3.0);
    samplesOnTriangle[5] = v1 * (1 / 6.0) + v2 * (1 / 6.0) + v0 * (2 / 3.0);

    return samplesOnTriangle;
}

// Compute the unit triangle normal.
CPoint3D BooleanComputation::computeTriangleNormal(int triIdx) const
{
    auto face = m_inputModel[0].Face(triIdx);
    CPoint3D v0 = m_inputModel[0].Vert(face[0]);
    CPoint3D v1 = m_inputModel[0].Vert(face[1]);
    CPoint3D v2 = m_inputModel[0].Vert(face[2]);

    // Compute two edge vectors.
    CPoint3D vec1 = v1 - v0;
    CPoint3D vec2 = v2 - v0;

    // Compute the cross-product normal.
    CPoint3D normal = vec1 * vec2;
    double len = normal.Len();

    if (len > 1e-10) {
        normal.x /= len;
        normal.y /= len;
        normal.z /= len;
    }

    return normal;
}

// Compute the triangle area.
double BooleanComputation::computeTriangleArea(int triIdx) const
{
    auto face = m_inputModel[0].Face(triIdx);
    CPoint3D v0 = m_inputModel[0].Vert(face[0]);
    CPoint3D v1 = m_inputModel[0].Vert(face[1]);
    CPoint3D v2 = m_inputModel[0].Vert(face[2]);

    CPoint3D vec1 = v1 - v0;
    CPoint3D vec2 = v2 - v0;

    CPoint3D cross = vec1 * vec2;
    return cross.Len() * 0.5;
}

void BooleanComputation::prepareENAIntegralData()
{
    const int numFaces = m_inputModel[0].GetNumOfFaces();
    m_enaFaceCache.resize(numFaces);

#pragma omp parallel for schedule(static)
    for (int triIdx = 0; triIdx < numFaces; ++triIdx) {
        const CPoint3D normal = computeTriangleNormal(triIdx);
        const std::array<CPoint3D, 6> samples =
            getSamplesOnTriangle(triIdx);

        ENAFaceCache& cache = m_enaFaceCache[triIdx];
        cache.normalX = normal.x;
        cache.normalY = normal.y;
        cache.normalZ = normal.z;
        cache.area = computeTriangleArea(triIdx);
        for (int k = 0; k < 6; ++k) {
            cache.sampleProjections[k] =
                normal.x * samples[k].x +
                normal.y * samples[k].y +
                normal.z * samples[k].z;
        }
    }
}

void BooleanComputation::buildENAIndex()
{
    std::cout << "构建 E_NA 索引..." << std::endl;

    int numVerts = m_inputModel[0].GetNumOfVerts();
    m_enaVertices.clear();
    m_enaVertices.reserve(numVerts);

    for (int vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
        // Read all triangles incident to this vertex.
        const auto& allTriIndices = m_vertToTris[vertIdx];

        // Select valid ENA triangles.
        std::vector<int> validTriIndices;
        for (int triIdx : allTriIndices) {
            if (m_faceValidForENA[triIdx]) {
                validTriIndices.push_back(triIdx);
            }
        }

        // Keep only vertices with at least one valid triangle.
        if (!validTriIndices.empty()) {
            CPoint3D v = m_inputModel[0].Vert(vertIdx);
            ENAVertexData data;
            data.position = Point_3(v.x, v.y, v.z);
            data.triangleIndices = std::move(validTriIndices);
            m_enaVertices.push_back(data);
        }
    }

    m_enaIndexBuilt = true;

    std::cout << "E_NA index built: " << m_enaVertices.size()
        << " / " << numVerts << " valid vertices" << std::endl;
}

// Query E_NA vertices with ANN. The returned tree indices are mapped to
// triangles in a separate parallel pass.
void
BooleanComputation::getNearbyTreeIndicesANN(
    const Point_3& xi,
    std::vector<int>& treeIndices
)
{
    treeIndices.clear();

    if (!m_enaKDTree)
        return;

    const double searchRadius =
        2.0 * m_sigma * m_ena_search_radius_scale;
    const double radiusSq = searchRadius * searchRadius;
    ANNcoord queryPt[3] = { xi.x(), xi.y(), xi.z() };
    const auto annStart = std::chrono::steady_clock::now();
    int kcount = 0;
    if (m_ann_single_pass_query) {
        int queryCapacity =
            (std::max)(32, static_cast<int>(treeIndices.capacity()));
        treeIndices.resize(queryCapacity);
        kcount = m_enaKDTree->annkFRSearch(
            queryPt,
            radiusSq,
            queryCapacity,
            treeIndices.data(),
            nullptr,
            m_ena_neighbor_epsilon);
        if (kcount > queryCapacity) {
            treeIndices.resize(kcount);
            m_enaKDTree->annkFRSearch(
                queryPt,
                radiusSq,
                kcount,
                treeIndices.data(),
                nullptr,
                m_ena_neighbor_epsilon);
        }
    } else {
        // Reference path: count first, then repeat the query to fill results.
        kcount = m_enaKDTree->annkFRSearch(
            queryPt,
            radiusSq,
            0,
            nullptr,
            nullptr,
            m_ena_neighbor_epsilon);
        if (kcount > 0) {
            treeIndices.resize(kcount);
            m_enaKDTree->annkFRSearch(
                queryPt,
                radiusSq,
                kcount,
                treeIndices.data(),
                nullptr,
                m_ena_neighbor_epsilon);
        }
    }
    if (kcount <= 0) {
        treeIndices.clear();
    } else {
        treeIndices.resize(
            (std::min)(kcount, static_cast<int>(treeIndices.size())));
    }
    m_profileEnaANNSeconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - annStart).count();
}

void BooleanComputation::mapNearbyTreeIndicesToTriangles(
    const std::vector<int>& treeIndices,
    std::vector<int>& triangleIndices)
{
    triangleIndices.clear();
    if (treeIndices.empty())
        return;

    thread_local std::vector<uint32_t> triangleVisitStamp;
    thread_local uint32_t queryStamp = 0;
    const int numFaces = m_inputModel[0].GetNumOfFaces();
    if (triangleVisitStamp.size() !=
        static_cast<size_t>(numFaces)) {
        triangleVisitStamp.assign(numFaces, 0);
        queryStamp = 0;
    }
    ++queryStamp;
    if (queryStamp == 0) {
        std::fill(
            triangleVisitStamp.begin(),
            triangleVisitStamp.end(),
            0);
        ++queryStamp;
    }

    for (int treeIndex : treeIndices) {
        if (treeIndex < 0 || treeIndex >= m_enaPointNum)
            continue;
        const int vertIdx = m_enaKdTreeToVertex[treeIndex];
        const int begin = m_vertToTriOffsets[vertIdx];
        const int end = m_vertToTriOffsets[vertIdx + 1];
        for (int pos = begin; pos < end; ++pos) {
            const int triIdx = m_vertToTriIndices[pos];
            if (m_faceValidForENA[triIdx] &&
                triangleVisitStamp[triIdx] != queryStamp) {
                triangleVisitStamp[triIdx] = queryStamp;
                triangleIndices.push_back(triIdx);
            }
        }
    }
}

void BooleanComputation::BuildENAKDTree()
{
    int numVerts = m_inputModel[0].GetNumOfVerts();
    std::vector<char> keepVertex(numVerts, 0);
    for (int triIdx = 0;
         triIdx < m_inputModel[0].GetNumOfFaces();
         ++triIdx) {
        if (!m_faceValidForENA[triIdx]) {
            continue;
        }
        const CBaseModel::CFace& face = m_inputModel[0].Face(triIdx);
        keepVertex[face[0]] = 1;
        keepVertex[face[1]] = 1;
        keepVertex[face[2]] = 1;
    }

    m_enaKdTreeToVertex.clear();
    m_enaKdTreeToVertex.reserve(numVerts);
    for (int vertIdx = 0; vertIdx < numVerts; ++vertIdx) {
        if (keepVertex[vertIdx]) {
            m_enaKdTreeToVertex.push_back(vertIdx);
        }
    }
    m_enaPointNum = static_cast<int>(m_enaKdTreeToVertex.size());

    m_enaDataPts = annAllocPts(m_enaPointNum, 3);

    for (int i = 0; i < m_enaPointNum; i++)
    {
        CPoint3D p = m_inputModel[0].Vert(m_enaKdTreeToVertex[i]);

        m_enaDataPts[i][0] = p.x;
        m_enaDataPts[i][1] = p.y;
        m_enaDataPts[i][2] = p.z;
    }

    m_enaKDTree = new ANNkd_tree(
        m_enaDataPts,
        m_enaPointNum,
        3,
        m_ena_ann_bucket_size,
        m_ena_ann_split_rule
    );

    std::cout
        << "ENA KDTree Build Finished : "
        << m_enaPointNum
        << " vertices"
        << std::endl;
}

void BooleanComputation::ComputeVertexCurvatureDensity()
{
    using Kernel = CGAL::Simple_cartesian<double>;
    using Point_3 = Kernel::Point_3;
    using Mesh = CGAL::Surface_mesh<Point_3>;
    using FT = Kernel::FT;

    int nVerts = m_inputModel[0].GetNumOfVerts();
    m_vertexRho.assign(nVerts, 1.0);

    if (nVerts == 0) {
        std::cout << "[ComputeVertexCurvatureDensity] 模型无顶点，跳过" << std::endl;
        return;
    }

    // 1. Convert to CGAL::Surface_mesh.
    Mesh cgalMesh;
    std::vector<Mesh::Vertex_index> vertIndices(nVerts);

    for (int i = 0; i < nVerts; ++i) {
        const auto& pt = m_inputModel[0].Vert(i);
        vertIndices[i] = cgalMesh.add_vertex(Point_3(pt.x, pt.y, pt.z));
    }

    int nFaces = m_inputModel[0].GetNumOfFaces();
    for (int i = 0; i < nFaces; ++i) {
        auto face = m_inputModel[0].Face(i);
        cgalMesh.add_face(vertIndices[face[0]], vertIndices[face[1]], vertIndices[face[2]]);
    }

    std::cout << "[ComputeVertexCurvatureDensity] CGAL mesh: "
        << cgalMesh.number_of_vertices() << " vertices, "
        << cgalMesh.number_of_faces() << " faces" << std::endl;

    // 2. Create the curvature property map.
    namespace PMP = CGAL::Polygon_mesh_processing;
    using PC = PMP::Principal_curvatures_and_directions<Kernel>;
    Mesh::Property_map<Mesh::Vertex_index, PC> principal_map =
        cgalMesh.add_property_map<Mesh::Vertex_index, PC>("v:principal", PC{}).first;

    // 3. Compute principal curvatures with CGAL.
    PMP::interpolated_corrected_curvatures(
        cgalMesh,
        PMP::parameters::vertex_principal_curvatures_and_directions_map(principal_map)
    );

    std::cout << "[ComputeVertexCurvatureDensity] curvature complete" << std::endl;

    // 4. Compute raw density values and statistics.
    std::vector<double> rhoRaw(nVerts);
    double rhoMin = 1e10;
    double rhoMax = -1e10;
    double rhoSum = 0.0;

    for (int i = 0; i < nVerts; ++i) {
        auto pc = principal_map[vertIndices[i]];

        double k1 = CGAL::to_double(pc.max_curvature);
        double k2 = CGAL::to_double(pc.min_curvature);

        double rho = (std::abs(k1) + std::abs(k2)) / 2.0;

        rhoRaw[i] = rho;
        rhoSum += rho;
        if (rho < rhoMin) rhoMin = rho;
        if (rho > rhoMax) rhoMax = rho;
    }

    double rhoAvg = rhoSum / nVerts;

    std::cout << "[ComputeVertexCurvatureDensity] 原始曲率统计: min=" << rhoMin
        << ", avg=" << rhoAvg << ", max=" << rhoMax << std::endl;

    // ============================================================
    // 5. Use the minimum as the planar baseline.
    // ============================================================

    // Handle a perfectly planar mesh.
    if (rhoMax < 1e-30) {
        m_vertexRho.assign(nVerts, 1.0);
        std::cout << "[ComputeVertexCurvatureDensity] 纯平面模型，所有密度设??1.0" << std::endl;
        return;
    }

    // Handle near-zero minima caused by numerical error.
    double baseRho = rhoMin;

    // Use a robust baseline when the minimum is too small.
    if (rhoMin < 1e-12) {
        // Collect nonzero density values.
        std::vector<double> validRho;
        validRho.reserve(nVerts);
        for (int i = 0; i < nVerts; ++i) {
            if (rhoRaw[i] > 1e-20) {
                validRho.push_back(rhoRaw[i]);
            }
        }

        if (!validRho.empty()) {
            std::sort(validRho.begin(), validRho.end());
            // Use the 10th percentile as the baseline.
            int idx = max(0, (int)(validRho.size() * 0.1));
            baseRho = validRho[idx];
        }
        else {
            // All values are zero; the mesh is planar.
            m_vertexRho.assign(nVerts, 1.0);
            std::cout << "[ComputeVertexCurvatureDensity] 无有效曲率，所有密度设??1.0" << std::endl;
            return;
        }
    }

    std::cout << "[ComputeVertexCurvatureDensity] 使用基准?? " << baseRho << std::endl;

    // 6. Normalize density; planar regions map around 1.0 and curved regions above it.
    double newSum = 0.0;
    double newMin = 1e10, newMax = -1e10;

    for (int i = 0; i < nVerts; ++i) {
        double normalizedRho =rhoRaw[i];

        

        m_vertexRho[i] = normalizedRho;
        newSum += normalizedRho;
        if (normalizedRho < newMin) newMin = normalizedRho;
        if (normalizedRho > newMax) newMax = normalizedRho;
    }

    std::cout << "[ComputeVertexCurvatureDensity] 归一化后: min=" << newMin
        << ", avg=" << newSum / nVerts << ", max=" << newMax << std::endl;
	m_rho_avg = newSum / nVerts;
	m_rho_base = baseRho;
    m_rho_base *= 2;
    std::cout << "[ComputeVertexCurvatureDensity] 平面密度 = 1.0，曲面密??> 1.0" << std::endl;
}


double BooleanComputation::GetDensityAtPoint(const CPoint3D& pt)
{
    return 1;
    
    // 1. Check whether curvature density data exists.
    if (m_vertexRho.empty() || m_vertexRho.size() < 3)
    {
        cout<<"Error: Vertex curvature density data not computed."<<endl;
        return 1.0;
    }

    // 2. Query the closest point through the OBB tree.
    Distance_OBB::QueryResult qr = m_obbtree.Query(pt);
    int triIdx = qr.triangleID;

    // 3. Validate the triangle index.
    if (triIdx < 0 || triIdx >= m_inputModel[0].GetNumOfFaces())
    {
        return 1.0;
    }

    // 4. Read the three triangle vertices.
    auto face = m_inputModel[0].Face(triIdx);
    int v0 = face[0], v1 = face[1], v2 = face[2];

    CPoint3D A = m_inputModel[0].Vert(v0);
    CPoint3D B = m_inputModel[0].Vert(v1);
    CPoint3D C = m_inputModel[0].Vert(v2);

    // 5. Compute barycentric coordinates of the closest point.
    CPoint3D projPnt = qr.closestPnt;
    double w0, w1, w2;
    if (!ComputeBarycentricWeights(projPnt, A, B, C, w0, w1, w2))
    {
        // Fall back to the mean vertex density on failure.
        return (m_vertexRho[v0] + m_vertexRho[v1] + m_vertexRho[v2]) / 3.0;
    }

    // 6. Interpolate density using barycentric coordinates.
    double rho = w0 * m_vertexRho[v0] + w1 * m_vertexRho[v1] + w2 * m_vertexRho[v2];
    
    rho /= m_rho_base;

    CPoint3D n0 = m_inputModel[0].Normal(v0);
    CPoint3D n1 = m_inputModel[0].Normal(v1);
    CPoint3D n2 = m_inputModel[0].Normal(v2);

    auto angleBetween = [](const CPoint3D& a, const CPoint3D& b) -> double {
        double dot = (a ^ b);
        if (dot > 1.0) dot = 1.0;
        if (dot < -1.0) dot = -1.0;
        return std::acos(dot);
        };

    double maxAngle = max(max(angleBetween(n0, n1), angleBetween(n0, n2)), angleBetween(n1, n2));

    const double EDGE_THRESHOLD = 0.5;  // Approximately 28.6 degrees; adjustable.

    bool isEdgeTriangle = (maxAngle > EDGE_THRESHOLD);

    // ============================================================
    // Check one-ring neighbors for sharp edges.
    // ============================================================
    bool neighborHasEdge = false;

    // Test whether a vertex lies on an edge using adjacent normals.
    auto isVertexOnEdge = [&](int vertIdx) -> bool {
        const auto& adjTris = m_vertToTris[vertIdx];
        if (adjTris.size() < 2) return true;  // Boundary vertices count as edges.

        CPoint3D firstNormal = computeTriangleNormal(adjTris[0]);
        for (size_t i = 1; i < adjTris.size(); ++i) {
            CPoint3D normal = computeTriangleNormal(adjTris[i]);
            double dot = firstNormal.x * normal.x + firstNormal.y * normal.y + firstNormal.z * normal.z;
            if (dot < 0.5) return true;  // Normal angle exceeds roughly 60 degrees.
        }
        return false;
        };

    // Check whether any of the three vertex neighborhoods contains an edge.
    auto checkNeighborEdge = [&](int vertIdx) -> bool {
        const auto& adjTris = m_vertToTris[vertIdx];
        for (int triIdx : adjTris) {
            auto f = m_inputModel[0].Face(triIdx);
            if (isVertexOnEdge(f[0]) || isVertexOnEdge(f[1]) || isVertexOnEdge(f[2])) {
                return true;
            }
        }
        return false;
        };

    neighborHasEdge = checkNeighborEdge(v0) || checkNeighborEdge(v1) || checkNeighborEdge(v2);

  
    if (isEdgeTriangle) {
        rho = 1.0;  
    }
    else if (neighborHasEdge)
    {
        rho = 0.8;
    }
    else {
        if (rho < 0.8) rho =0.8;
        else if (rho > 2.5) rho = 2.5;
        //else rho *= 1.2;
    }
    if(rho > 1.0) rho = 1.0 + std::sqrt(rho - 1.0); 
    return rho;
}

bool BooleanComputation::ComputeBarycentricWeights(
    const CPoint3D& p,
    const CPoint3D& a,
    const CPoint3D& b,
    const CPoint3D& c,
    double& w0,
    double& w1,
    double& w2) const
{
    const CPoint3D v0 = b - a;
    const CPoint3D v1 = c - a;
    const CPoint3D v2 = p - a;
    const double d00 = v0 ^ v0;
    const double d01 = v0 ^ v1;
    const double d11 = v1 ^ v1;
    const double d20 = v2 ^ v0;
    const double d21 = v2 ^ v1;
    const double denom = d00 * d11 - d01 * d01;
    if (std::fabs(denom) <= 1e-12)
    {
        return false;
    }

    w1 = (d11 * d20 - d01 * d21) / denom;
    w2 = (d00 * d21 - d01 * d20) / denom;
    w0 = 1.0 - w1 - w2;
    return std::isfinite(w0) && std::isfinite(w1) && std::isfinite(w2);
}
