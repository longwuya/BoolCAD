#pragma once
#include"Reconstruction_Basedon_Paricles_LBFGS.h"
#include <cstdlib>						// C standard library
#include <cstdio>						// C I/O (for sscanf)
#include <cstring>						// string manipulation
#include <fstream>						// file I/O
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_segment_primitive.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include "ANN.h"					// ANN declarations
#include <array>
#include <cstdint>
#include <vector>
#include <set>

// Exact predicates with inexact constructions balance robustness and speed.
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

// Geometry type aliases.
typedef Kernel::Point_3 Point_3;
typedef Kernel::Triangle_3 Triangle_3;
typedef Kernel::Segment_3 Segment_3;

class BooleanComputation : public Reconstruction_Basedon_Paricles_LBFGS
{
private:
	double m_radiusOfBall;
	double m_boundingBoxRadiusOfSurface;
	// The OBB trees hold references, so the copied models must be declared first.
	CBaseModel m_inputModel[3];
	Distance_OBB m_obbtree, m_obbtree1, m_obbtree2;
	double m_lambda_ofnoraml_penalty;
	double m_modelArea;
	double m_sigma;
	int m_booleanType;
	double m_subtractionWeightFactor;
	std::vector<std::vector<Point_3>> intersection_seg;
	double intersection_rate;
	double e_param_bottom;
	double sigma_rate;
	// Per-particle adjacency lists remain stable across evaluations so their
	// capacities are reused by the serial ANN query.
	std::vector<std::vector<int>> m_neighborLists;
	std::vector<std::vector<int>> m_enaNeighborTreeIndices;
	std::vector<std::vector<int>> m_enaNearbyTriangles;
	std::vector<int> m_neighborCSROffsets;
	std::vector<int> m_neighborCSRIndices;
	std::vector<int> m_enaTriangleCSROffsets;
	std::vector<int> m_enaTriangleCSRIndices;
	std::vector<double> m_lastParticleNeighborPositions;
	std::vector<double> m_lastEnaNeighborPositions;
	bool m_particleNeighborCacheValid = false;
	bool m_enaNeighborCacheValid = false;
	int m_lastParticleNeighborEvaluation = 0;
	int m_lastEnaNeighborEvaluation = 0;

	// Reusable ANN KD-tree storage.
	ANNpointArray m_ann_dataPts = nullptr;      // ANN data buffer allocated by annAllocPts.
	ANNkd_tree* m_ann_kdTree = nullptr;       // ANN tree owned by this object.
	int           m_ann_dim = 3;                // Fixed point dimension.
	int           m_ann_point_capacity = 0;     // Allocated point capacity.
	int           m_ann_point_num = 0;          // Currently stored point count.
	int           m_ann_bucket_size = 8;        // ANN leaf bucket size.
	int           m_particle_ann_bucket_size = 8;
	int           m_ena_ann_bucket_size = 8;
	double        m_particle_neighbor_epsilon = 0.01;
	double        m_ena_neighbor_epsilon = 0.02;
	double        m_ena_search_radius_scale = 1.0;
	ANNsplitRule  m_particle_ann_split_rule = ANN_KD_SUGGEST;
	ANNsplitRule  m_ena_ann_split_rule = ANN_KD_SUGGEST;
	bool          m_ann_single_pass_query = true;
	bool          m_ann_morton_query_order = false;
	bool          m_ann_csr_neighbor_storage = false;

	// Flattened intersection segments and their AABB tree.
	std::vector<Segment_3> m_intersection_segments_flat; // Flattened segment storage.
	using SegmentPrimitive = CGAL::AABB_segment_primitive<Kernel, std::vector<Segment_3>::iterator>;
	using AABBTraits = CGAL::AABB_traits<Kernel, SegmentPrimitive>;
	using SegmentTree = CGAL::AABB_tree<AABBTraits>;
	std::unique_ptr<SegmentTree> m_aabb_tree; // AABB tree for segment distance queries.

	// Internal helper functions.
	void EnsureANNCapacity(int requiredPts); // Ensure the ANN data buffer is large enough.
	void BuildOrUpdateANNKdTree(const std::vector<Point>& points); // Rebuild the KD-tree while reusing its buffer.
	void BuildOrUpdateANNKdTreeFromCoordinates(const double* x, int pointNum);
	void BuildMortonQueryOrder(
		const double* x,
		int pointCount,
		std::vector<int>& queryOrder) const;
	void RefreshParticleNeighborLists(
		const double* x,
		int pointCount,
		double sigma,
		const std::vector<int>& queryOrder);
	void RefreshENANeighborTreeIndices(
		const double* x,
		int pointCount,
		const std::vector<int>& queryOrder);
	void BuildParticleNeighborCSR(int pointCount);
	void BuildENATriangleCSR(int pointCount);

	// Marks triangles that contribute to the E_NA integral.
	std::vector<uint8_t> m_faceValidForENA;
	void initFacesForENA();
	bool isVertexValidForENA(const CPoint3D& v, int otherModelId);

	// Vertex-to-triangle adjacency.
	std::vector<std::vector<int>> m_vertToTris;
	std::vector<int> m_vertToTriOffsets;
	std::vector<int> m_vertToTriIndices;

	// Build the adjacency map.
	void buildVertexToTrianglesMap();

	// Debug helper for printing the vertex-to-triangle mapping.
	void testVertexMapping(int testVertexIndex);

	double lambda_PS = 1.0;   // Particle-system energy weight.
	double lambda_NA = 1.0;   // Normal-consistency energy weight.
	double r_NA;        // Integration radius.
	double lambda_balance;
	int m_iter_cnt = 0;
	
	

	std::set<int> getNearbyTriangles(const Point_3& xi) const;
	double m_EnaSearchRadius;
	int m_EnaPointNum;

	Point_3 toPoint3(const CPoint3D& p) const;
	std::array<CPoint3D, 6> getSamplesOnTriangle(int triangleIdx) const;

	CPoint3D computeTriangleNormal(int triIdx) const;
	double computeTriangleArea(int triIdx) const;
	void prepareENAIntegralData();

	// Compact E_NA triangle cache. The six integration samples are stored
	// through their projection onto the triangle normal, which is the only
	// sample value used by the energy and gradient. Keeping the normal, area,
	// and projections together removes repeated random reads from three arrays.
	struct ENAFaceCache {
		double normalX;
		double normalY;
		double normalZ;
		double area;
		std::array<double, 6> sampleProjections;
	};
	std::vector<ENAFaceCache> m_enaFaceCache;

	// Vertex KD-tree state.
	ANNkd_tree* m_vertexKdTree;
	ANNpointArray m_vertexDataPts;
	int m_vertexPointNum;
	int m_vertexPointCapacity;
	const int m_vertexDim = 3;
	bool m_vertexKdTreeBuilt;  // Whether the vertex KD-tree has been built.

	// Vertex KD-tree construction.
	void buildVertexKDTree();
	void EnsureVertexANNCapacity(int requiredPts);

	// Precomputed E_NA data.
	struct ENAVertexData {
		Point_3 position;                    // Vertex position.
		std::vector<int> triangleIndices;    // Valid incident triangle indices.
	};

	std::vector<ENAVertexData> m_enaVertices;  // Deduplicated valid vertices.
	bool m_enaIndexBuilt;                     // Whether the ENA index is ready.

	// Build the E_NA index once during construction.
	void buildENAIndex();

	

	double m_minEtot = DBL_MAX;

	// KD-tree used for E_NA neighbor queries.
	ANNkd_tree* m_enaKDTree = nullptr;

	ANNpointArray m_enaDataPts = nullptr;
	std::vector<int> m_enaKdTreeToVertex;

	int m_enaPointNum = 0;

	void BooleanComputation::BuildENAKDTree();
	void getNearbyTreeIndicesANN(
		const Point_3& xi,
		std::vector<int>& treeIndices);
	void mapNearbyTreeIndicesToTriangles(
		const std::vector<int>& treeIndices,
		std::vector<int>& triangleIndices);

	std::vector<double> m_vertexRho;
	std::vector<double> m_particleRho;
	double m_rho_avg;
	double m_rho_base;

	// Optional accumulator timings enabled with RECONSTRUCTION_PROFILE=1.
	double m_profileProjectionSeconds = 0.0;
	double m_profileNeighborQuerySeconds = 0.0;
	double m_profileParticleTreeBuildSeconds = 0.0;
	double m_profileParticleSearchSeconds = 0.0;
	double m_profileNeighborEnergySeconds = 0.0;
	double m_profileEnaQuerySeconds = 0.0;
	double m_profileEnaANNSeconds = 0.0;
	double m_profileEnaMapSeconds = 0.0;
	double m_profileEnaEnergySeconds = 0.0;
	int m_profileNeighborRefreshes = 0;
	int m_profileNeighborReuses = 0;
	int m_profileEnaRefreshes = 0;
	int m_profileEnaReuses = 0;
	double m_profileMaxParticleMove = 0.0;
	double m_profileMaxEnaMove = 0.0;
	void ComputeVertexCurvatureDensity();     // Compute curvature density with CGAL.
	double GetDensityAtPoint(const CPoint3D& pt); // Query and interpolate density through the OBB tree.

	bool ComputeBarycentricWeights(
		const CPoint3D& p,
		const CPoint3D& a,
		const CPoint3D& b,
		const CPoint3D& c,
		double& w0,
		double& w1,
		double& w2) const;
public:
	// Release ANN-owned resources.
	~BooleanComputation();

	BooleanComputation(const CBaseModel (&inputModel)[3], int nSamples,
		double factor_for_particle_system, 
		double factor_for_intersection,
		double lamdaForBalancingParticleSystemAndDistanceConstraints,
		double radiusOfBall,
		double boundingBoxRadiusOfSurface,
		double lambda_ofnoraml_penalty,
		int booleanType,
		double subtractionWeightFactor,
		std::vector<std::vector<Point_3>> intersection_segments, // Intersection polylines.
		double i_intersection_rate,
		double e_e_param_bottom,
		double s_sigma_rate
		);
	int get_sample_num();
	double e_density_calculation_func(double d, double sigma, double area);
	double e_density_calculation_func2(double d, double sigma, double area);
protected:
	lbfgsfloatval_t evaluate(
		lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		);

	 vector<Point> GetInitialSamplesOnSurface(int nSamples);
	 vector<Point> GetValidSamplesByUnion(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2);
	 vector<Point> GetValidSamplesByIntersection(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2);
	 vector<Point> GetValidSamplesBySubtraction(vector<Point>& InitialSamples, Distance_OBB& obbtree1, Distance_OBB& obbtree2);
	 double GetSurfaceArea()const;
	 CPoint3D GetNormalOfPointIntriangle(int triangleID, CBaseModel& m_inputModel)const;

	 vector<Point> GetclosetNeighorPoints(ANNkd_tree * kdTree, Point queryPoint, double sigma, vector<Point>& pointsForKdTree)const;
	 // Return neighbor indices instead of Point copies to reduce allocation overhead.
	 // eps is ANN's approximation parameter; 0.0 is exact and 0.01-0.05 trades accuracy for speed.
	 void GetclosetNeighorPointsIndices(
		 ANNkd_tree* kdTree,
		 const Point& queryPoint,
		 double sigma,
		 int pointCount,
		 std::vector<int>& indices,
		 double eps = 0.0) const;
	 ANNkd_tree * GetKDTree(int dim, int pointNum, vector <Point> &points) const;
	 int IsOnSurface(CPoint3D p, Distance_OBB& obbtree, CBaseModel& inputModel);
	 CPoint3D GetCorrectPointInUnionSituation(const CPoint3D& m_p, Distance_OBB &obbtree1,
	 Distance_OBB& obbtree2);
	 CPoint3D GetCorrectPointInIntersectionSituation(const CPoint3D &m_p,
		 Distance_OBB &obbtree1, Distance_OBB& obbtree2);
	 CPoint3D GetCorrectPointInSubtractionSituation(const CPoint3D &m_p,
		 Distance_OBB &obbtree1, Distance_OBB& obbtree2);
	 CPoint3D GetCorrectPointInEdgeSituation(const CPoint3D& m_p);
	 double BooleanComputation::GetPointToMeshSurface(const CPoint3D& m_p, Distance_OBB& obbtree);
	 bool BooleanComputation::ErrorToMeshSurface(const CBaseModel& input, Distance_OBB& obbtree2, double bound_min, double bound_max, const std::string& outObjFilename);

};


