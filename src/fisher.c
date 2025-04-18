#include <ksw_fisher_internal.h>
#include <stdio.h>

inline ptrdiff_t _max(ptrdiff_t a, ptrdiff_t b) {
	return ((a) > (b) ? a : b);
}

inline ptrdiff_t _min(ptrdiff_t a, ptrdiff_t b) {
	return ((a) < (b) ? a : b);
}

void compute_associated_legendre_sp(const double *thetas, float *p_theta_ell, int ntheta, int lmax) {

	int nell = lmax + 1;
	double epsilon = 1e-300;

#pragma omp parallel
	{
		Ylmgen_C ygen;

		Ylmgen_init(&ygen, lmax, 0, 0, 0, epsilon);
		Ylmgen_set_theta(&ygen, thetas, ntheta);

#pragma omp for schedule(dynamic, 5)
		for (ptrdiff_t tidx = 0; tidx < ntheta; tidx++) {

			Ylmgen_prepare(&ygen, tidx, 0);
			Ylmgen_recalc_Ylm(&ygen);

			ptrdiff_t firstl = *ygen.firstl;

			for (ptrdiff_t lidx = firstl; lidx < nell; lidx++) {

				// Convert from SH to associated Legendre.
				p_theta_ell[tidx * nell + lidx] = (float)(ygen.ylm[lidx] * sqrt(4 * PI / (double)(2 * lidx + 1)));
			}
		}

		Ylmgen_destroy(&ygen);
	} // End of parallel region.
}

void unique_nxn_on_ring_sp(const float *sqrt_icov_ell, const float *f_ell_i, const float *p_ell,
	const float *prefactor, float *work_i, float *unique_nxn, int nufact,
	int nell, int npol) {

	// Set output to zero.
	for (ptrdiff_t i = 0; i < nufact * nufact; i++) {
		unique_nxn[i] = 0.0;
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {

		// sqrt_icov @ f_i -> work_i.
		cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, npol, nufact,
			1.0, sqrt_icov_ell + lidx * npol * npol, npol,
			f_ell_i + lidx * npol * nufact, nufact, 0.0, work_i, nufact);

		// P * work^T x work -> unique_nxn.
		cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, nufact, npol,
			p_ell[lidx] * prefactor[lidx], work_i, nufact, 1.0, unique_nxn, nufact);
	}
}

void fisher_nxn_on_ring_sp(const float *unique_nxn, const long long *rule,
	const float *weights, float *fisher_nxn, double ct_weight,
	int nufact, int nrule) {

	for (ptrdiff_t ridx = 0; ridx < nrule; ridx++) {

		long long rx = rule[ridx * 3];
		long long ry = rule[ridx * 3 + 1];
		long long rz = rule[ridx * 3 + 2];

		float wx = weights[ridx * 3];
		float wy = weights[ridx * 3 + 1];
		float wz = weights[ridx * 3 + 2];

		// We only fill upper triangular part.
		for (ptrdiff_t rjdx = ridx; rjdx < nrule; rjdx++) {

			long long rpx = rule[rjdx * 3];
			long long rpy = rule[rjdx * 3 + 1];
			long long rpz = rule[rjdx * 3 + 2];

			float wpx = weights[rjdx * 3];
			float wpy = weights[rjdx * 3 + 1];
			float wpz = weights[rjdx * 3 + 2];

			// r and rp are indices into unique_nxn. Min/max to only acces uppper tri part.
			float tmp = unique_nxn[_min(rx, rpx) * nufact + _max(rx, rpx)] * unique_nxn[_min(ry, rpy) * nufact + _max(ry, rpy)] * unique_nxn[_min(rz, rpz) * nufact + _max(rz, rpz)];

			// + 5 permutations.
			tmp += unique_nxn[_min(rx, rpz) * nufact + _max(rx, rpz)] * unique_nxn[_min(ry, rpx) * nufact + _max(ry, rpx)] * unique_nxn[_min(rz, rpy) * nufact + _max(rz, rpy)];

			tmp += unique_nxn[_min(rx, rpy) * nufact + _max(rx, rpy)] * unique_nxn[_min(ry, rpz) * nufact + _max(ry, rpz)] * unique_nxn[_min(rz, rpx) * nufact + _max(rz, rpx)];

			tmp += unique_nxn[_min(rx, rpx) * nufact + _max(rx, rpx)] * unique_nxn[_min(ry, rpz) * nufact + _max(ry, rpz)] * unique_nxn[_min(rz, rpy) * nufact + _max(rz, rpy)];

			tmp += unique_nxn[_min(rx, rpy) * nufact + _max(rx, rpy)] * unique_nxn[_min(ry, rpx) * nufact + _max(ry, rpx)] * unique_nxn[_min(rz, rpz) * nufact + _max(rz, rpz)];

			tmp += unique_nxn[_min(rx, rpz) * nufact + _max(rx, rpz)] * unique_nxn[_min(ry, rpy) * nufact + _max(ry, rpy)] * unique_nxn[_min(rz, rpx) * nufact + _max(rz, rpx)];
			// eq 29
			fisher_nxn[ridx * nrule + rjdx] += tmp * wx * wy * wz * wpx * wpy * wpz * (float)(ct_weight * 2 * PI * PI / 9);
		}
	}
}

void fisher_nxn_sp(const float *sqrt_icov_ell, const float *f_ell_i, const double *thetas,
	const double *ct_weights, const long long *rule, const float *weights,
	float *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol) {

	int nell = lmax + 1;

	float *p_theta_ell = malloc(sizeof * p_theta_ell * ntheta * nell);
	float *prefactor = malloc(sizeof * prefactor * nell);

	if (p_theta_ell == NULL || prefactor == NULL) {
		free(p_theta_ell);
		free(prefactor);
		exit(1);
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {
		// for eq 30
		prefactor[lidx] = (2 * lidx + 1) / 4. / PI;
	}

	compute_associated_legendre_sp(thetas, p_theta_ell, ntheta, lmax);

#pragma omp parallel
	{
		mkl_set_num_threads_local(1);

		float *work_i = malloc(sizeof * work_i * npol * nufact);
		float *unique_nxn = malloc(sizeof * unique_nxn * nufact * nufact);
		float *fisher_nxn_priv = calloc(nrule * nrule, sizeof * fisher_nxn_priv);

		if (work_i == NULL || unique_nxn == NULL || fisher_nxn_priv == NULL) {
			free(work_i);
			free(unique_nxn);
			free(fisher_nxn_priv);
			exit(1);
		}

#pragma omp for schedule(dynamic)
		for (ptrdiff_t tidx = 0; tidx < ntheta; tidx++) {

			unique_nxn_on_ring_sp(sqrt_icov_ell, f_ell_i, p_theta_ell + tidx * nell,
				prefactor, work_i, unique_nxn, nufact, nell, npol);

			fisher_nxn_on_ring_sp(unique_nxn, rule, weights, fisher_nxn_priv,
				ct_weights[tidx], nufact, nrule);
		}

#pragma omp critical
		{
			for (ptrdiff_t idx = 0; idx < nrule; idx++) {
				for (ptrdiff_t jdx = idx; jdx < nrule; jdx++) {
					fisher_nxn[idx * nrule + jdx] += fisher_nxn_priv[idx * nrule + jdx];
				}
			}
		}

		free(work_i);
		free(unique_nxn);
		free(fisher_nxn_priv);

		mkl_set_num_threads_local(0);
	} // End of parallel region

	free(p_theta_ell);
	free(prefactor);
}

/* Double precision versions */

void compute_associated_legendre_dp(const double *thetas, double *p_theta_ell,
	int ntheta, int lmax) {

	int nell = lmax + 1;
	double epsilon = 1e-300;

#pragma omp parallel
	{
		Ylmgen_C ygen;

		Ylmgen_init(&ygen, lmax, 0, 0, 0, epsilon);
		Ylmgen_set_theta(&ygen, thetas, ntheta);

#pragma omp for schedule(dynamic, 5)
		for (ptrdiff_t tidx = 0; tidx < ntheta; tidx++) {

			Ylmgen_prepare(&ygen, tidx, 0);
			Ylmgen_recalc_Ylm(&ygen);

			ptrdiff_t firstl = *ygen.firstl;

			for (ptrdiff_t lidx = firstl; lidx < nell; lidx++) {

				// Convert from SH to associated Legendre.
				p_theta_ell[tidx * nell + lidx] = (ygen.ylm[lidx] * sqrt(4 * PI / (double)(2 * lidx + 1)));
			}
		}

		Ylmgen_destroy(&ygen);
	} // End of parallel region.
}

void unique_nxn_on_ring_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *p_ell,
	const double *prefactor, double *work_i, double *unique_nxn, int nufact,
	int nell, int npol) {

	// Set output to zero.
	for (ptrdiff_t i = 0; i < nufact * nufact; i++) {
		unique_nxn[i] = 0.0;
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {

		// sqrt_icov @ f_i -> work_i.
		cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, npol, nufact,
			1.0, sqrt_icov_ell + lidx * npol * npol, npol,
			f_ell_i + lidx * npol * nufact, nufact, 0.0, work_i, nufact);

		// P * work^T @ work -> unique_nxn.
		cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nufact, npol,
			p_ell[lidx] * prefactor[lidx], work_i, nufact, 1.0, unique_nxn, nufact);
	}
}

void fisher_nxn_on_ring_dp(const double *unique_nxn, const long long *rule,
	const double *weights, double *fisher_nxn, double ct_weight,
	int nufact, int nrule) {

	for (ptrdiff_t ridx = 0; ridx < nrule; ridx++) {

		long long rx = rule[ridx * 3];
		long long ry = rule[ridx * 3 + 1];
		long long rz = rule[ridx * 3 + 2];

		double wx = weights[ridx * 3];
		double wy = weights[ridx * 3 + 1];
		double wz = weights[ridx * 3 + 2];

		// We only fill upper triangular part.
		for (ptrdiff_t rjdx = ridx; rjdx < nrule; rjdx++) {

			long long rpx = rule[rjdx * 3];
			long long rpy = rule[rjdx * 3 + 1];
			long long rpz = rule[rjdx * 3 + 2];

			double wpx = weights[rjdx * 3];
			double wpy = weights[rjdx * 3 + 1];
			double wpz = weights[rjdx * 3 + 2];

			// r and rp are indices into unique_nxn. Min/max to only acces uppper tri part.
			double tmp = unique_nxn[_min(rx, rpx) * nufact + _max(rx, rpx)] * unique_nxn[_min(ry, rpy) * nufact + _max(ry, rpy)] * unique_nxn[_min(rz, rpz) * nufact + _max(rz, rpz)];

			// + 5 permutations.
			tmp += unique_nxn[_min(rx, rpz) * nufact + _max(rx, rpz)] * unique_nxn[_min(ry, rpx) * nufact + _max(ry, rpx)] * unique_nxn[_min(rz, rpy) * nufact + _max(rz, rpy)];

			tmp += unique_nxn[_min(rx, rpy) * nufact + _max(rx, rpy)] * unique_nxn[_min(ry, rpz) * nufact + _max(ry, rpz)] * unique_nxn[_min(rz, rpx) * nufact + _max(rz, rpx)];

			tmp += unique_nxn[_min(rx, rpx) * nufact + _max(rx, rpx)] * unique_nxn[_min(ry, rpz) * nufact + _max(ry, rpz)] * unique_nxn[_min(rz, rpy) * nufact + _max(rz, rpy)];

			tmp += unique_nxn[_min(rx, rpy) * nufact + _max(rx, rpy)] * unique_nxn[_min(ry, rpx) * nufact + _max(ry, rpx)] * unique_nxn[_min(rz, rpz) * nufact + _max(rz, rpz)];

			tmp += unique_nxn[_min(rx, rpz) * nufact + _max(rx, rpz)] * unique_nxn[_min(ry, rpy) * nufact + _max(ry, rpy)] * unique_nxn[_min(rz, rpx) * nufact + _max(rz, rpx)];

			fisher_nxn[ridx * nrule + rjdx] += tmp * wx * wy * wz * wpx * wpy * wpz * (ct_weight * 2 * PI * PI / 9);
		}
	}
}

void fisher_nxn_dp(const double *sqrt_icov_ell, const double *f_ell_i, const double *thetas,
	const double *ct_weights, const long long *rule, const double *weights,
	double *fisher_nxn, int nufact, int nrule, int ntheta, int lmax, int npol) {

	int nell = lmax + 1;

	double *p_theta_ell = malloc(sizeof * p_theta_ell * ntheta * nell);
	double *prefactor = malloc(sizeof * prefactor * nell);

	if (p_theta_ell == NULL || prefactor == NULL) {
		free(p_theta_ell);
		free(prefactor);
		exit(1);
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {
		prefactor[lidx] = (2 * lidx + 1) / 4. / PI;
	}

	compute_associated_legendre_dp(thetas, p_theta_ell, ntheta, lmax);

#pragma omp parallel
	{
		mkl_set_num_threads_local(1);

		double *work_i = malloc(sizeof * work_i * npol * nufact);
		double *unique_nxn = malloc(sizeof * unique_nxn * nufact * nufact);
		double *fisher_nxn_priv = calloc(nrule * nrule, sizeof * fisher_nxn_priv);

		if (work_i == NULL || unique_nxn == NULL || fisher_nxn_priv == NULL) {
			free(work_i);
			free(unique_nxn);
			free(fisher_nxn_priv);
			exit(1);
		}

#pragma omp for schedule(dynamic)
		for (ptrdiff_t tidx = 0; tidx < ntheta; tidx++) {

			unique_nxn_on_ring_dp(sqrt_icov_ell, f_ell_i, p_theta_ell + tidx * nell,
				prefactor, work_i, unique_nxn, nufact, nell, npol);

			fisher_nxn_on_ring_dp(unique_nxn, rule, weights, fisher_nxn_priv,
				ct_weights[tidx], nufact, nrule);
		}

#pragma omp critical
		{
			for (ptrdiff_t idx = 0; idx < nrule; idx++) {
				for (ptrdiff_t jdx = idx; jdx < nrule; jdx++) {
					fisher_nxn[idx * nrule + jdx] += fisher_nxn_priv[idx * nrule + jdx];
				}
			}
		}

		free(work_i);
		free(unique_nxn);
		free(fisher_nxn_priv);

		mkl_set_num_threads_local(0);
	} // End of parallel region

	free(p_theta_ell);
	free(prefactor);
}


void unique_axb_on_ring_sp(const float *sqrt_icov_ell, const float *f_ell_i_a, const float *f_ell_i_b, const float *p_ell,
	const float *prefactor, float *work_i, float *work_j, float *unique_nxn, int nufact_a, int nufact_b, int nell, int npol) {

	// Set output to zero.
	for (ptrdiff_t i = 0; i < nufact_a * nufact_b; i++) {
		unique_nxn[i] = 0.0;
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {

		// sqrt_icov @ f_i -> work_i.
		cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, // symmetric matrix-matrix multiplication
			npol, // number of rows in B and C
			nufact_a, // number of columns of B and C
			1.0, //multi for product of A@B
			sqrt_icov_ell + lidx * npol * npol, // A matrix, adding to pointer to shift
			npol, // leading dim of A
			f_ell_i_a + lidx * npol * nufact_a, // the B matrix
			nufact_a, // leading dim of B
			0.0, //scalar multipler for C
			work_i, // result, C
			nufact_a // leading dim of C
		);

		cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, // symmetric matrix-matrix multiplication
			npol, // number of rows in B and C
			nufact_b, // number of columns of B and C
			1.0, //multi for product of A@B
			sqrt_icov_ell + lidx * npol * npol, // A matrix, adding to pointer to shift
			npol, // leading dim of A
			f_ell_i_b + lidx * npol * nufact_b, // the B matrix
			nufact_b, // leading dim of B
			0.0, //scalar multipler for C
			work_j, // result, C
			nufact_b // leading dim of C
		);

		cblas_sgemm(
			CblasRowMajor,    // Layout: Row-major storage for matrices.
			CblasTrans,       // TransA: Transpose the first input matrix (work_i).
			CblasNoTrans,     // TransB: Do not transpose the second input matrix (work_j).
			nufact_a,         // M: Number of rows op(A) and C.
			nufact_b,         // N: Number of columns op(B) and C.
			npol,             // K: Shared dimension of work_i and work_j (number of columns in work_i and rows in work_j).
			p_ell[lidx] * prefactor[lidx], // alpha: Scalar multiplier for the product of work_i^T and work_j.
			work_i,           // A: The first input matrix (work_i), size lda*k.
			nufact_a,             // lda: Leading dimension of work_i (number of columns in work_i).
			work_j,           // B: The second input matrix (work_j) size ldb * k.
			nufact_b,             // ldb: Leading dimension of work_j (number of columns in work_j).
			1.0,              // beta: Scalar multiplier for the existing values in unique_nxn.
			unique_nxn,       // C: The result matrix (unique_nxn).
			nufact_b          // ldc: Leading dimension of unique_nxn (number of columns in unique_nxn).
		);
	}
}

void fisher_axb_on_ring_sp(const float *unique_nxn, const long long *rule_a, const long long *rule_b,
	const float *weights_a, const float *weights_b, float *fisher_nxn, double ct_weight,
	int nufact_a, int nufact_b, int nrule_a, int nrule_b) {

	for (ptrdiff_t ridx = 0; ridx < nrule_a; ridx++) {

		long long rx = rule_a[ridx * 3];
		long long ry = rule_a[ridx * 3 + 1];
		long long rz = rule_a[ridx * 3 + 2];

		float wx = weights_a[ridx * 3];
		float wy = weights_a[ridx * 3 + 1];
		float wz = weights_a[ridx * 3 + 2];

		// We only fill upper triangular part.
		// for (ptrdiff_t rjdx = ridx; rjdx < nrule_b; rjdx++) {
		for (ptrdiff_t rjdx = 0; rjdx < nrule_b; rjdx++) {

			long long rpx = rule_b[rjdx * 3];
			long long rpy = rule_b[rjdx * 3 + 1];
			long long rpz = rule_b[rjdx * 3 + 2];

			float wpx = weights_b[rjdx * 3];
			float wpy = weights_b[rjdx * 3 + 1];
			float wpz = weights_b[rjdx * 3 + 2];

			int nufact = nufact_b;
			float tmp = unique_nxn[rx * nufact + rpx] * unique_nxn[ry * nufact + rpy] * unique_nxn[rz * nufact + rpz];
			tmp += unique_nxn[rx * nufact + rpz] * unique_nxn[ry * nufact + rpx] * unique_nxn[rz * nufact + rpy];
			tmp += unique_nxn[rx * nufact + rpy] * unique_nxn[ry * nufact + rpz] * unique_nxn[rz * nufact + rpx];
			tmp += unique_nxn[rx * nufact + rpx] * unique_nxn[ry * nufact + rpz] * unique_nxn[rz * nufact + rpy];
			tmp += unique_nxn[rx * nufact + rpy] * unique_nxn[ry * nufact + rpx] * unique_nxn[rz * nufact + rpz];
			tmp += unique_nxn[rx * nufact + rpz] * unique_nxn[ry * nufact + rpy] * unique_nxn[rz * nufact + rpx];
			fisher_nxn[ridx * nrule_b + rjdx] += tmp * wx * wy * wz * wpx * wpy * wpz * (float)(ct_weight * 2 * PI * PI / 9);
		}
	}
}

void fisher_axb_sp(const float *sqrt_icov_ell, const float *f_ell_i_a, const float *f_ell_i_b, const double *thetas,
	const double *ct_weights, const long long *rule_a, const long long *rule_b, const float *weights_a, const float *weights_b,
	float *fisher_nxn, int nufact_a, int nufact_b, int nrule_a, int nrule_b, int ntheta, int lmax, int npol) {

	int nell = lmax + 1;

	float *p_theta_ell = calloc(ntheta * nell, sizeof * p_theta_ell);
	float *prefactor = calloc(nell, sizeof * prefactor);

	if (p_theta_ell == NULL || prefactor == NULL) {
		free(p_theta_ell);
		free(prefactor);
		exit(1);
	}

	for (ptrdiff_t lidx = 0; lidx < nell; lidx++) {
		prefactor[lidx] = (2 * lidx + 1) / 4. / PI;
	}

	compute_associated_legendre_sp(thetas, p_theta_ell, ntheta, lmax);

#pragma omp parallel
	{
		mkl_set_num_threads_local(1);

		float *work_i = calloc(npol * nufact_a, sizeof * work_i);
		float *work_j = calloc(npol * nufact_b, sizeof * work_j);
		float *unique_nxn = calloc(nufact_a * nufact_b, sizeof * unique_nxn);
		float *fisher_nxn_priv = calloc(nrule_a * nrule_b, sizeof * fisher_nxn_priv);

		if (work_i == NULL || work_j == NULL || unique_nxn == NULL || fisher_nxn_priv == NULL) {
			free(work_i);
			free(work_j);
			free(unique_nxn);
			free(fisher_nxn_priv);
			exit(1);
		}

#pragma omp for schedule(dynamic)
		for (ptrdiff_t tidx = 0; tidx < ntheta; tidx++) {
			unique_axb_on_ring_sp(sqrt_icov_ell, f_ell_i_a, f_ell_i_b, p_theta_ell + tidx * nell,
				prefactor, work_i, work_j, unique_nxn, nufact_a, nufact_b, nell, npol);

			fisher_axb_on_ring_sp(unique_nxn, rule_a, rule_b, weights_a, weights_b, fisher_nxn_priv,
				ct_weights[tidx], nufact_a, nufact_b, nrule_a, nrule_b);
		}

#pragma omp critical
		{
			for (ptrdiff_t idx = 0; idx < nrule_a; idx++) {
				// for (ptrdiff_t jdx = idx; jdx < nrule_b; jdx++) {
				for (ptrdiff_t jdx = 0; jdx < nrule_b; jdx++) {
					fisher_nxn[idx * nrule_b + jdx] += fisher_nxn_priv[idx * nrule_b + jdx];
				}
			}
		}

		free(work_i);
		free(work_j);
		free(unique_nxn);
		free(fisher_nxn_priv);

		mkl_set_num_threads_local(0);
	} // End of parallel region

	free(p_theta_ell);
	free(prefactor);
}