#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pathos as pa


def check_and_set_nchains(nchains, logger=None):
    """Check and set passed number of jobs.

    Checks whether passed `nchains` has correct type and value, raises if not.

    If nchains exceeds the number of available CPUs found by `Pathos` (this
    might include hardware threads), `nchains` is set to the number found by
    `Pathos`.

    If `nchains=-1`, then nchains is set to half of the CPUs found by `Pathos`
    (we assume half of the CPUs are only hardware threads and ignore those).

    Parameters
    ----------
    nchains : :obj:`int`
        Number of processes (workers) passed by user.
    logger : :obj:`logging.Logger`
        Logger object.

    Returns
    -------
    nchains : :obj:`int`
        Possibly corrected number of processes (workers).
    """
    if not isinstance(nchains, int):
        msg = "nchains must be passed as int."
        raise TypeError(msg)

    if nchains < -1 or nchains == 0:
        msg = "With the exception of nchains=-1, negative nchains cannot be " "passed."
        raise ValueError(msg)

    n_cpus = pa.helpers.cpu_count()

    if nchains > n_cpus:
        if logger is not None:
            logger.warn(f"nchains={nchains} exceeds the found threads={n_cpus}")
            logger.warn("Reducing nchains to match number of threads in system")
        nchains = n_cpus
    elif nchains == -1:
        nchains = n_cpus // 2

    return nchains


def generate_seed_sequence(user_seed=None, pool_size=None):
    """Process a user-provided seed and convert it into initial states for
    parallel pool workers.

    Parameters
    ----------
    user_seed : :obj:`int`
        User-provided seed. Default is None.
    pool_size : :obj:`int`
        The number of spawns that will be passed to child processes.

    Returns
    -------
    seeds : :obj:`list`
        Initial states for parallel pool workers.
    """
    seed_sequence = np.random.SeedSequence(user_seed)
    seeds = seed_sequence.spawn(pool_size)
    return seeds


def advance_PRNG_state(seed, delta):
    """Advance the underlying PRNG as-if delta draws have occurred.

    In the ABC samplers, the random values are simulated using a
    rejection-based method and so, typically, more than one value from the
    underlying PRNG is required to generate a single posterior draw.

    Advancing a PRNG updates the underlying PRNG state as if a number
    of delta calls to the underlying PRNG have been made.

    Parameters
    ----------
    seed : SeedSequence
        Initial state.
    delta : :obj:`int`
        Number of draws to advance the PRNG.

    Returns
    -------
    object : PCG64
        PRNG advanced delta steps.
    """
    return np.random.PCG64(seed).advance(delta)
