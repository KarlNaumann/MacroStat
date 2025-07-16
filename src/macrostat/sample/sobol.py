"""
Class designed to facilitate the sampling of the model's
parameterspace
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

# Default libraries
import logging

# Third-party libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

import macrostat.util.batchprocessing as msbatchprocessing

# Custom imports
from macrostat.core.model import Model
from macrostat.sample.sampler import BaseSampler

logger = logging.getLogger(__name__)


class SobolSampler(BaseSampler):
    """A Sobol sequence sampler for efficient parameter space exploration.

    The SobolSampler class implements quasi-random low-discrepancy sequence sampling
    to systematically explore a model's parameter space. The benefit of Sobol
    sequences is that even the sub-samples are uniform distributed, which is not
    the case for random sampling from a uniform distribution.

    The sampler works by generating a Sobol sequence in the unit hypercube [0,1]^d,
    transforming the sequence to the desired parameter ranges and creating model
    instances with the sampled parameters.

    Example
    -------
    >>> model = MyModel()
    >>> bounds = {
    ...     'param1': (0.1, 1.0),
    ...     'param2': (1.0, 10.0)
    ... }
    >>> sampler = SobolSampler(
    ...     model=model,
    ...     bounds=bounds,
    ...     sample_power=8,  # 2^8 = 256 samples
    ...     logspace=True,   # Sample in log space
    ...     cpu_count=4      # Use 4 CPUs
    ... )
    >>> sampler.sample()
    """

    def __init__(
        self,
        model: Model,
        bounds: dict | None = None,
        sample_power: int = 10,
        sobol_seed: int = 0,
        logspace: bool = False,
        worker_function: callable = msbatchprocessing.timeseries_worker,
        simulation_args: tuple = (),
        output_folder: str = "sobol_samples",
        cpu_count: int = 1,
        batchsize: int = None,
        output_filetype: str = "csv",
        output_compression: str | None = None,
    ):
        """Initialize a Sobol sequence sampler.

        For most initialization see the BaseSampler class, in addition, this method
        stores the sobol_seed and sample_power attributes.

        Parameters
        ----------
        model: msmodel.Model
            Model to be sampled
        bounds: dict[str, tuple] | None (default None)
            Dictionary containing the bounds for each parameter to be sampled.
            If None, the bounds are taken from the model's parameters and all
            parameters are sampled.
        sample_power: int (default 10)
            A power of 2 to determine the number of samples to be generated,
            i.e. 2**sample_power samples will be generated
        sobol_seed: int (default 0)
            Seed for the Sobol sequence
        logspace: bool (default False)
            Whether to sample the parameters in logspace
        worker_function: callable (default batchprocessing.timeseries_worker)
            Function to be used for the parallel processing
        simulation_args: tuple (default ())
            Arguments to be passed to the model's simulate method irrespective
            of the parameters
        output_folder: str (default "samples")
            Folder to save the output files
        cpu_count: int (default 1)
            Number of CPUs to use for the parallel processing
        batchsize: int (default None)
            Size of each batch to be processed in parallel
        """
        super().__init__(
            model=model,
            bounds=bounds,
            logspace=logspace,
            worker_function=worker_function,
            simulation_args=simulation_args,
            output_folder=output_folder,
            cpu_count=cpu_count,
            batchsize=batchsize,
            output_filetype=output_filetype,
            output_compression=output_compression,
        )

        # Sampling parameters
        self.sobol_seed = sobol_seed
        self.sample_power = sample_power

    def generate_parameters(self):
        """Generate points in the parameterspace for the parallel processor
        based on a Sobol sequence.

        Here the scipy.stats.qmc.Sobol class is used to generate the Sobol sequence,
        specifically the random_base2 method is used to generate the samples, as it
        is has slightly better space filling properties than with a custom
        number of samples.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the Sobol points in the parameterspace. Rows are
            the samples, columns are the parameters
        """
        # Generate the bounds to scale by
        bounds_array = np.array(list(self.bounds.values()))

        if self.logspace:
            # Take the sign and then log the bounds.
            bounds_sign = np.sign(bounds_array)
            bounds_array = np.log(np.abs(bounds_array))

        # Generate the Sobol sequence
        sobol_sampler = stats.qmc.Sobol(len(self.bounds), rng=self.sobol_seed)
        sobol_sample = sobol_sampler.random_base2(self.sample_power)
        sample = stats.qmc.scale(sobol_sample, bounds_array[:, 0], bounds_array[:, 1])

        if self.logspace:
            # Take the exponential and multiply by the sign to get the correct bounds
            sample = np.exp(sample) * bounds_sign[:, 0]

        return pd.DataFrame(sample, columns=self.bounds.keys())
