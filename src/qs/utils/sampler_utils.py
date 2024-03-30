from joblib import delayed
from joblib import Parallel  # instead of from pathos.pools import ProcessPool


# In here I've rewritten the function to accept the parameters that we use in the sampler.
# def multiproc(proc_sample, wf, nchains, nsamples, state, scale, seeds):
def multiproc(proc_sample, nsamples, nchains, seeds):
    """Enable multiprocessing for jax."""
    # params = wf.params
    # placeholder for params
    # params = 0 # usually a dictionary

    # Handle iterable
    # wf = [wf] * nchains
    # state = [state] * nchains
    # params = [params] * nchains
    # scale = [scale] * nchains

    nsamples = [nsamples] * nchains
    chain_ids = list(range(nchains))

    # Define a helper function to package the delayed computation
    # def compute(i):
    #     return proc_sample(
    #         wf[i], nsamples[i], state[i], scale[i], seeds[i], chain_ids[i]
    #     )

    def compute(i):
        return proc_sample(nsamples[i], chain_ids[i], seeds[i])

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute)(i) for i in range(nchains)
    )
    # Assuming that proc_sample returns a tuple (result, energy), you can unpack them
    results, sampled_positions, energies = zip(*results)

    return results, sampled_positions, energies
