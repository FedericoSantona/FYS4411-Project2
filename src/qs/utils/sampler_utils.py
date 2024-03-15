from joblib import delayed
from joblib import Parallel # instead of from pathos.pools import ProcessPool


def multiproc(proc_sample, wf, nchains, nsamples, state, scale, seeds):
    """Enable multiprocessing for jax."""
    #params = wf.params
    #placeholder for params
    params = 0 # usually a dictionary

    # Handle iterable
    wf = [wf] * nchains
    nsamples = [nsamples] * nchains
    state = [state] * nchains
    params = [params] * nchains
    scale = [scale] * nchains
    chain_ids = list(range(nchains))

    # Define a helper function to package the delayed computation
    def compute(i):
        return proc_sample(
            wf[i], nsamples[i], state[i], scale[i], seeds[i], chain_ids[i]
        )

    results = Parallel(n_jobs=-1)(delayed(compute)(i) for i in range(nchains))

    # Assuming that proc_sample returns a tuple (result, energy), you can unpack them
    results, energies = zip(*results)

    return results, energies