		#include "Reconstruction_Basedon_Paricles_LBFGS.h"
		#include <algorithm>
		using namespace std;

		Reconstruction_Basedon_Paricles_LBFGS::Reconstruction_Basedon_Paricles_LBFGS(
			int nSamples, 
			double factor_for_particle_system,
			double factor_for_intersection,
			double lamdaForBalancingParticleSystemAndDistanceConstraints):m_nSamples(nSamples), 
			m_factor_for_particle_system(factor_for_particle_system),
			m_factor_for_intersection(factor_for_intersection),
			m_lamdaForBalancingParticleSystemAndDistanceConstraints(lamdaForBalancingParticleSystemAndDistanceConstraints)
		{	
		}

		


