import numpy as np
from ksw import utils
from itertools import product

import cython
cimport cfisher_core

def fisher_nxn(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights, fisher_nxn):
    '''
    Calculate upper-triangular part of (nrule x nrule) fisher matrix.

    Arguments
    ---------
    sqrt_icov_ell : (nell, npol, npol) array
        Square root of icov per multipole (symmetric in pol).
    f_ell_i : (nell, npol, nufact) array
        Unique factors of reduced bispectrum.
    thetas : (ntheta) array
        Theta value for each ring.
    ct_weights : (ntheta) array
        Quadrature weights for cos(theta) on each ring.
    rule : (nrule, 3) array
        Rule to combine unique bispectrum factors 
    weights : (nrule, 3) array
        Amplitude for each element of rule.    
    fisher_nxn : (nrule * nrule) array
        Fisher matrix for these rings, only upper-tringular part is filled.

    Raises
    ------
    ValueError 
        If input shapes do not match.
    '''

    nell, npol, nufact = f_ell_i.shape
    lmax = nell - 1
    nrule = rule.shape[0]
    ntheta = thetas.size

    if sqrt_icov_ell.shape[-2:] != (npol, npol):
        raise ValueError(f'Pol dimensions of sqrt_icov : {sqrt_icov_ell.shape[-2:]} '
                         f'do not match npol of f_ell_i : {npol}')

    if sqrt_icov_ell.shape[0] != nell:
        raise ValueError(f'nell dimension of sqrt_icov : {sqrt_icov_ell.shape[0]} '
                         f'do not match nell of f_ell_i {nell}')

    if thetas.size != ct_weights.size:
        raise ValueError(f'Size thetas : {thetas.size} is not equal to size '
	                 f'ct_weights : {ct_weights.size}')

    if rule.shape != weights.shape:
        raise ValueError(f'Shape rule : {rule.shape} is not equal to shape weights : '
                         f'{weights.shape}')

    if f_ell_i.dtype == np.float32:
        _fisher_nxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
	               fisher_nxn, nufact, nrule, ntheta, lmax, npol)
    elif f_ell_i.dtype == np.float64:
        _fisher_nxn_dp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
	               fisher_nxn, nufact, nrule, ntheta, lmax, npol)
    else:
        raise ValueError(f'dtype : {f_ell_i.dtype} not supported')

def _fisher_nxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
                   fisher_nxn, nufact, nrule, ntheta, lmax, npol):
    ''' Single precision version.'''
    cdef float [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef float [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] fisher_nxn_ = fisher_nxn.reshape(-1)   
    
    cfisher_core.fisher_nxn_sp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)

def _fisher_nxn_dp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
                   fisher_nxn, nufact, nrule, ntheta, lmax, npol):
    ''' Double precision version.'''

    cdef double [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef double [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef double [::1] weights_ = weights.reshape(-1)
    cdef double [::1] fisher_nxn_ = fisher_nxn.reshape(-1)    
    
    cfisher_core.fisher_nxn_dp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)

def fisher_multi(sqrt_icov_ell, f_ell_is, thetas, ct_weights, rules, weights, comm):
    cdef:
        unsigned int i, j
        unsigned int n_bispec = len(f_ell_is)
        int ntheta = thetas.size
        int nell, nell_b, npol, nufact_a, nufact_b, lmax, nrule_a, nrule_b
        float[::1] tmp
        float[::1] f_ell_a, f_ell_b, weight_a, weight_b
        float[:, ::1] tmp_data
        long long[::1] rule_a, rule_b

        # reshape the data for the c-code
        float[::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
        double[::1] thetas_ = thetas.reshape(-1)
        double[::1] ct_weights_ = ct_weights.reshape(-1)

    d_type = f_ell_is[0].dtype
    fisher_mat = np.zeros((n_bispec, n_bispec), dtype=d_type)

    with cython.boundscheck(False), cython.wraparound(False):
        for i, j in product(range(n_bispec), repeat=2):
            # if j < i:
            #     # symmetric matrix so we only need to calculate the upper triangular part
            #     continue

            print("Calculating fisher matrix for bispectrum %d and %d" % (i, j))
            nell, npol, nufact_a = f_ell_is[i].shape
            nell_b, npol_b, nufact_b = f_ell_is[j].shape

            if nell != nell_b:
                raise ValueError(f'nell dimension of f_ell_i : {nell} '
                                f'does not match nell of f_ell_j : {nell_b}')
            if npol != npol_b:
                raise ValueError(f'npol dimension of f_ell_i : {npol} '
                                f'does not match npol of f_ell_j : {npol_b}')

            lmax = nell - 1
            nrule_a = rules[i].shape[0]
            nrule_b = rules[j].shape[0]

            f_ell_a, weight_a, rule_a = f_ell_is[i].reshape(-1), weights[i].reshape(-1), rules[i].reshape(-1)
            f_ell_b, weight_b, rule_b = f_ell_is[j].reshape(-1), weights[j].reshape(-1), rules[j].reshape(-1)

            tmp = np.zeros((nrule_a*nrule_b), dtype=d_type)
            
            cfisher_core.fisher_axb_sp(
                &sqrt_icov_ell_[0], 
                &f_ell_a[0], 
                &f_ell_b[0], 
                &thetas_[0],
                &ct_weights_[0], 
                &rule_a[0], 
                &rule_b[0],
                &weight_a[0],
                &weight_b[0], 
                &tmp[0], 
                nufact_a, nufact_b, nrule_a, nrule_b, ntheta, lmax, npol)

            tmp = utils.allreduce_array(tmp, comm)
            tmp_data = np.array(tmp).reshape(nrule_a, nrule_b)
            fisher_mat[i, j] = np.sum(np.triu(tmp_data, 1)) + np.sum(np.triu(tmp_data))

    return fisher_mat