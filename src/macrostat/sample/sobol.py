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
import copy
import inspect
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# Third-party libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

# Custom imports
import macrostat.models.model as msmodel
import macrostat.sample.sampler as mssampler
import macrostat.util.batchprocessing as msbatchprocessing


class SobolSampler(mssampler.Sampler):
    def __init__(
        self,
        model: msmodel.Model,
        bounds: dict,
        sample_power: int = 10,
        seed: int = 0,
        logspace: bool = False,
        worker_function: callable = msbatchprocessing.timeseries_worker,
        simulation_args: tuple = (),
        output_folder: str = "sobol_samples",
        cpu_count: int = 1,
        batchsize: int = None,
    ):
        """Generalized class to facilitate the sampling of the model's 
        parameterspace using python's multiprocessing library.

        Parameters
        ----------
        model: msmodel.Model
            Model to be sampled
        bounds: dict[str, tuple]
            Dictionary containing the bounds for each parameter to be sampled
        sample_power: int (default 10)
            A power of 2 to determine the number of samples to be generated, 
            i.e. 2**sample_power samples will be generated
        seed: int (default 0)
            Seed for the random number generator
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
            worker_function=worker_function,
            output_folder=output_folder,
            cpu_count=cpu_count,
            batchsize=batchsize,
        )

        # Boundaries for the parameters
        self.logspace = logspace
        self._verify_bounds(bounds)
        self.bounds = bounds

        # Sampling parameters
        self.sample_power = sample_power
        self.seed = seed

        self.simulation_args = simulation_args

    def generate_tasks(self):
        """Generate tasks for the parallel processor based on the Sobol sequence
        for the model's parameterspace using the bounds set in the class.

        Here the scipy.stats.qmc.Sobol class is used to generate the Sobol sequence,
        specifically the random_base2 method is used to generate the samples, as it
        is has slightly better space filling properties than with a custom 
        number of samples.

        Returns
        -------
        list[tuple]
            List of tuples containing the model and the task to be processed
        """
        # Generate the Sobol points
        points = self._generate_sobol_points()

        tasks = []
        for i in points.index:
            # Copy base parameters to ensure a full set of parameters
            newparams = copy.deepcopy(self.base_parameters)
            for key in points.columns:
                newparams[key] = points.loc[i, key]

            # Generate the task to execute
            newmodel = self.modelclass(parameters=newparams, **self.model_kwargs)
            tasks.append((i, newmodel, *self.simulation_args))

        return tasks

    def _verify_bounds(self, bounds:dict) -> None:
        """ Verify that the bounds are correctly set, in particular
        0. Check that the parameters are in the model
        1. That there is a lower and upper bound for each parameter
        2. That the lower bound is smaller than the upper bound
        3. That the bounds are in the correct order
        4. If the bounds are in logspace, that the bounds are either 
        both positive or both negative
        5. If the bounds are in logspace, that either bound is not zero

        Parameters
        ----------
        bounds: dict[str, tuple]
            Dictionary containing the bounds for each parameter to be sampled
        logspace: bool
            Whether to sample the parameters in logspace

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the bounds are not correctly set
        """
        # Check that the bounds are correctly set
        for param, bound in bounds.items():
            if param not in self.model.parameters:
                raise ValueError(f"Parameter {param} not in the model's parameters")
            if len(bound) != 2:
                raise ValueError(f"Bounds should be a list-like of length 2. {param}: {bound}")
            if self.logspace and (bound[0] < 0) != (bound[1] < 0):
                raise ValueError(f"Bounds should be either both positive or both negative. {param}: {bound}")
            if self.logspace and (bound[0] == 0 or bound[1] == 0):
                raise ValueError(f"Bounds cannot be zero when using logspace. {param}: {bound}")
            if bound[0] >= bound[1]:
                raise ValueError(f"Lower bound should be smaller than the upper bound. {param}: {bound}")

    def _generate_sobol_points(self):
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
        np.random.seed(self.seed)
        sobol_sampler = stats.qmc.Sobol(len(self.bounds))
        sobol_sample = sobol_sampler.random_base2(self.sample_power)
        sample = stats.qmc.scale(sobol_sample, bounds_array[:, 0], bounds_array[:, 1])

        if self.logspace:
            # Take the exponential and multiply by the sign to get the correct bounds
            sample = np.exp(sample) * bounds_sign[:, 0]

        return pd.DataFrame(sample, columns=self.bounds.keys())
