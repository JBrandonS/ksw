cimport cfisher_core
import numpy as np

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
    print("sqrt_icov_ell is C-contiguous:", sqrt_icov_ell.flags['C_CONTIGUOUS'], flush=True)
    print("f_ell_i is C-contiguous:", f_ell_i.flags['C_CONTIGUOUS'], flush=True)
    print("thetas is C-contiguous:", thetas.flags['C_CONTIGUOUS'], flush=True)
    print("ct_weights is C-contiguous:", ct_weights.flags['C_CONTIGUOUS'], flush=True)
    print("rule is C-contiguous:", rule.flags['C_CONTIGUOUS'], flush=True)
    print("weights is C-contiguous:", weights.flags['C_CONTIGUOUS'], flush=True)
    print("fisher_nxn is C-contiguous:", fisher_nxn.flags['C_CONTIGUOUS'], flush=True)

    cdef float [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef float [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] fisher_nxn_ = fisher_nxn.reshape(-1)    

    print("im here! calling cfisher_core", flush=True)
    print("", flush=True)
    
    cfisher_core.fisher_nxn_sp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)
    print("finished with cfisher_core", flush=True)

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

def fisher_nxnxnxn(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights, fisher_nxnxnxn):
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
    n_bispec = len(f_ell_i)
    for i, j in zip(range(n_bispec), range(n_bispec)):
        print("processing bispectrum", i, j, flush=True)
        if i == j:
            nell, npol, nufact = f_ell_i[i].shape
            lmax = nell - 1
            nrule = rule[i].shape[0]
            ntheta = thetas.size

            print("computing diagional fisher_nxnxnxn for bispectrum", i, flush=True)
            print("f_ell_i[i].shape:", f_ell_i[i].shape, "nell:", nell, "npol:", npol, "nufact:", nufact, "nrule:", rule[i].shape, rule[i].shape[0], "ct_weights:", ct_weights.shape, "weights:", weights[i].shape, "ntheta:", thetas.size, "lmax:", lmax, "nrule:", nrule, "ntheta:", ntheta, flush=True)
            print("", flush=True)

            if f_ell_i[i].dtype == np.float32:
                tmp = np.ascontiguousarray(fisher_nxnxnxn[i, j])
                tf = np.ascontiguousarray(f_ell_i[i])
                tr = np.ascontiguousarray(rule[i])
                tw = np.ascontiguousarray(weights[i])

                _fisher_nxn_sp(sqrt_icov_ell, tf, thetas, ct_weights, tr, tw, tmp, nufact, nrule, ntheta, lmax, npol)
                print("tmp shape:", tmp.shape, tmp, flush=True)
                fisher_nxnxnxn[i, j] = tmp +1
            elif f_ell_i[i].dtype == np.float64:
                print("Using double precision for fisher_nxnxnxn not supported rn", flush=True)
            else:
                raise ValueError(f'dtype : {f_ell_i.dtype} not supported')


def _fisher_nxnxnxn_sp(sqrt_icov_ell, f_ell_i, thetas, ct_weights, rule, weights,
                   fisher_nxn, nufact, nrule, ntheta, lmax, npol):
    ''' Single precision version.'''

    cdef float [::1] sqrt_icov_ell_ = sqrt_icov_ell.reshape(-1)
    cdef float [::1] f_ell_i_ = f_ell_i.reshape(-1)
    cdef double [::1] thetas_ = thetas.reshape(-1)
    cdef double [::1] ct_weights_ = ct_weights.reshape(-1)
    cdef long long [::1] rule_ = rule.reshape(-1)
    cdef float [::1] weights_ = weights.reshape(-1)
    cdef float [::1] fisher_nxn_ = fisher_nxn.reshape(-1)    
    
    cfisher_core.fisher_nxnxnxn_sp(&sqrt_icov_ell_[0], &f_ell_i_[0], &thetas_[0],
    		   &ct_weights_[0], &rule_[0], &weights_[0], &fisher_nxn_[0], 
		   nufact, nrule, ntheta, lmax, npol)