"""
Class designed to facilitate the sampling of the model's
parameterspace, generally by means of a Sobol sequence
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

# Default libraries
import copy
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
from eirinpy.eirinMatlab import EirinMatlab
import eirinpy.constructors.batchProcessing as batchProcessing


class Sampler(batchProcessing.BatchProcessing):
    def __init__(
        self,
        model,
        output_folder,
        parameters_to_sample: list = None,
        skipparams: list = None,
        cpu_count: int = 1,
    ):
        """Generalized class to use multiprocessing for computing the a Sobol
        sampling of the model parameters

        Parameters
        ----------
        model
            Some type of model. Generally an EirinMatlab object
        output_folder
            Directory to store the resulting samples
        cpu_count: int (default=1)
            How many CPUs to use for the Sobol sampling process


        Methods
        -------
        compute - will compute the Jacobian by central difference derivatives
        save - saves the current instance as a pickle file in the output folder
        load - (Class method) to load an instance from a pickle file
        """

        # Model parameters
        self.model = model
        self.modelclass = type(model)
        self.base_parameters = copy.deepcopy(model.parameters)
        self.bounds = None

        # Set several hyperparameters from the given model
        if isinstance(model, EirinMatlab):
            self.model_kwargs = {
                "hyperparameters": copy.deepcopy(model.hyperparameters),
                "initialconditions": copy.deepcopy(model.initialconditions),
                "matlab_folder": copy.deepcopy(model.matlab_folder),
                "output_folder": f"{output_folder}{os.sep}outputs{os.sep}",
                "calib_folder": f"{output_folder}{os.sep}parameters{os.sep}",
                "scenarios": copy.deepcopy(model.scenarios),
                "country_info": copy.deepcopy(model.country_info),
                "name": "jacobian",
            }
        else:
            # Otherwise simply use all possible attributes set in the model
            initargs = [
                i for i in inspect.signature(model.__init__).parameters if i != "self"
            ]
            self.model_kwargs = {a: getattr(self.model, a) for a in initargs}

        # Computation parameters
        self.cpu_count = min([mp.cpu_count(), cpu_count])
        self.output_folder = output_folder

        # Generate the list of parameters to perturb
        if parameters_to_sample is not None:
            self.parameters_to_sample = parameters_to_sample
        elif skipparams is not None:
            allparams = set(self.model.parameters.keys())
            self.parameters_to_sample = list(allparams.difference(skipparams))
        else:
            self.parameters_to_sample = list(self.model.parameters.keys())

        # Set the output folder
        if not (output_folder / "outputs").is_dir():
            os.makedirs(output_folder / "outputs", exist_ok=True)
        if not (output_folder / "parameters").is_dir():
            os.makedirs(output_folder / "parameters", exist_ok=True)

        # Initialize later output features
        self.trajectories = {}
        self.trajectories_iiasa = {}

    def set_bounds(self, method: str = "default", **kwargs):
        """Set the bounds based on some method"""
        if method == "default":
            bounds = EirinMatlab().get_default_boundaries()
            self.bounds = {k: bounds[k] for k in self.parameters_to_sample}
        elif method == "custom":
            self.bounds = kwargs["bounds"]
        else:
            raise NotImplementedError

    def save(self, name: str = "sampler"):
        """Save the Sampler object as a PKL for later use"""
        filename = f"{self.output_folder}{os.sep}{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Class method to load an instance of Sampler. Usage:

        sampler = Sampler.load(filename)

        Parameters
        ----------
        filename: str or Path
            path to the targeted Sampler
        """
        with open(filename, "rb") as f:
            new = pickle.load(f)
        return new

    def run_sobol_sampling(
        self,
        pow2: int = 5,
        scenarioID: int = 0,
        randomseed: int = 0,
        logsample: bool = False,
        batchsize: int = None,
        iiasa_formatting: bool = False,
        worker_kwargs: dict = {},
    ):
        """Run the Sobol sequence over the parameter space

        Parameters
        ----------
        scenarioID: int (default 0)
            Scenario of the model that is to be run
        iiasa_formatting: bool (default False)
            Whether to also compute the results as an iiasa formatted dictionary
        worker_kwargs: dict (default None)
            Any extra arguments for the worker. E.g. for the iiasa formatting this
            is the parameters passed to model.get_iiasa_results()
        """
        # Set up
        np.random.seed(randomseed)
        self._check_bounds(logsample)

        # Generate the tasks to run
        self._generate_sobol_points(pow2, logsample)
        self._generate_sobol_tasks(scenarioID=scenarioID, worker_kwargs=worker_kwargs)

        # Save the parameters
        self.sobol_points.to_csv(self.output_folder / "parameters.csv")

        # Select the correct worker function
        if iiasa_formatting:
            worker = batchProcessing.timeseries_worker_iiasa
        else:
            worker = batchProcessing.timeseries_worker

        # Run the parallel processing in batches to conserve memory
        # This will write results to disk, clear memory, and proceed
        batchsize = batchsize if batchsize is not None else len(self.all_tasks)
        batchcount = int(len(self.all_tasks) / batchsize) + (
            len(self.all_tasks) % batchsize > 0
        )

        for batch in range(batchcount):
            # Set tasks to run now
            self.tasks = self.all_tasks[
                batch * batchsize : min([(batch + 1) * batchsize, len(self.all_tasks)])
            ]
            # Execute those tasks
            raw_outputs = self.parallel_processor(
                worker=worker, tqdm_info=f"Sobol Batch {batch}"
            )
            # We only need to process iiasa formats, since those are not automatically written to disk
            if iiasa_formatting:
                self._process_iiasa_results(raw_outputs, scenarioID)

        # Convert all non-IIASA outputs to a sing
        self._post_processing()

    def run_custom_sampling(
        self,
        parametersets: dict,
        scenarioID: int = 0,
        batchsize: int = None,
        iiasa_formatting: bool = False,
        worker_kwargs: dict = {},
    ):
        """Run the Sobol sequence over the parameter space

        Parameters
        ----------
        scenarioID: int (default 0)
            Scenario of the model that is to be run
        iiasa_formatting: bool (default False)
            Whether to also compute the results as an iiasa formatted dictionary
        worker_kwargs: dict (default None)
            Any extra arguments for the worker. E.g. for the iiasa formatting this
            is the parameters passed to model.get_iiasa_results()
        """
        # Generate the tasks to run
        self._generate_generic_tasks(
            parametersets=parametersets,
            scenarioID=scenarioID,
            worker_kwargs=worker_kwargs,
        )

        # Save the parameters
        pd.DataFrame().from_dict(parametersets, orient="index").to_csv(
            self.output_folder / "parameters.csv"
        )

        # Select the correct worker function
        if iiasa_formatting:
            worker = batchProcessing.timeseries_worker_iiasa
        else:
            worker = batchProcessing.timeseries_worker

        # Run the parallel processing in batches to conserve memory
        # This will write results to disk, clear memory, and proceed
        batchsize = batchsize if batchsize is not None else len(self.all_tasks)
        batchcount = int(len(self.all_tasks) / batchsize) + (
            len(self.all_tasks) % batchsize > 0
        )

        for batch in range(batchcount):
            # Set tasks to run now
            self.tasks = self.all_tasks[
                batch * batchsize : min([(batch + 1) * batchsize, len(self.all_tasks)])
            ]
            # Execute those tasks
            raw_outputs = self.parallel_processor(worker=worker, tqdm_info=f"Sampling")
            # We only need to process iiasa formats, since those are not automatically written to disk
            if iiasa_formatting:
                self._process_iiasa_results(raw_outputs, scenarioID)

        # Convert all non-IIASA outputs to a sing
        self._post_processing()

    def extract_variables(
        self,
        variables: list = None,
        pids: list = None,
        periods: list = None,
        scenarios: list = None,
        use_iiasa: bool = False,
        chunksize: int = 100000,
    ):
        """Use the pandas chunkreader to extract a series of variables from the
        set of sampled parameterizations

        Parameters
        ----------
        variables: list
            List of variables to extract.
        periods: list (default None)
            Optionally extract only a subset of the periods from the file
        use_iiasa: bool (default False)
            Parse the IIASA file rather than the regular output file

        Returns
        -------
        results: pd.DataFrame
            Dataframe with (parameterID, scenarioID, variable) as multiindex and
            the periods as columns
        """

        file = "outputs_iiasa.csv" if use_iiasa else "outputs.csv"
        csv_kwargs = dict(header=0, index_col=[0, 1, 2])

        # Get the periods to extract from the file
        header = pd.read_csv(self.output_folder / file, nrows=0, **csv_kwargs)
        periods = header.columns if periods is None else periods

        # Read in chunks
        reader = pd.read_csv(
            self.output_folder / file, chunksize=chunksize, iterator=True, **csv_kwargs
        )
        output = []
        for i, chunk in tqdm(enumerate(reader), desc="Chunk Reading"):
            chunk.index.names = [i.lower() for i in chunk.index.names]

            if periods is not None:
                ix_per = chunk.columns.isin(periods)
                chunk = chunk.loc[:, ix_per]

            if variables is not None:
                ix_var = chunk.index.isin(variables, level="variable")
                chunk = chunk.loc[ix_var]

            if pids is not None:
                ix_pid = chunk.index.isin(pids, level="id")
                chunk = chunk.loc[ix_pid]

            if scenarios is not None:
                ix_scid = chunk.index.isin(scenarios, level="scenario")
                chunk = chunk.loc[ix_scid]

            output.append(chunk)

        output = pd.concat(output, axis=0).reorder_levels(
            ["id", "scenario", "variable"]
        )
        return output

    def _check_bounds(self, logsample: bool = False):
        """Check the validity of the submitted boundaries

        Parameters
        ----------
        logsample: bool (default false)
            Whether to check for diverging signs in the boundaries
        """
        if self.bounds is None:
            raise ValueError("Bounds are None")

        check = {k: i[0] >= i[1] for k, i in self.bounds.items()}
        if any(check.values()):
            found = [i for i, v in check.items() if v]
            raise ValueError(f"Bounds are incongrouent (min>=max)\n{found}")

        if not all([i in self.bounds for i in self.parameters_to_sample]):
            raise ValueError("Bounds are missing")

        # Check if sign(lower)!=sign(upper)
        if logsample:
            arr = np.array(list(self.bounds.values()))
            sign = np.sign(arr[arr == 0])
            sign[sign == 0] = 1
            arr[arr == 0.0] = sign * 1e-20
            test = np.sign(arr[:, 0]) != np.sign(arr[:, 1])
            if any(test):
                print([list(self.bounds.keys())[i] for i, t in enumerate(test) if t])
                raise ValueError("Can't sample for min<0 and max>0")

    def _generate_sobol_points(
        self, pow2: int = 5, logsample: bool = False, logmin: float = 1e-6
    ):
        """Generate a set of Sobol points within the given boundaries

        Parameters
        ----------
        pow2: int (default 5)
            2**pow2 samples will be generated
        logsample: bool (default False)
            Whether to sample the space in log terms rather than nominal.
            Useful for the case where boundaries span orders of magnitude
        """
        # Generate the sobol sequence
        sampler = stats.qmc.Sobol(d=len(self.bounds))
        sample = sampler.random_base2(m=pow2)

        # Generate the bounds to scale by
        bounds_array = np.array(list(self.bounds.values()))

        if logsample:
            # Can't take log of zeros, so floor them to 1e-6
            sign = np.sign(bounds_array[bounds_array == 0])
            sign[sign == 0] = 1
            bounds_array[bounds_array == 0] = sign * logmin
            # Take the sign and then log the bounds
            # TODO: deal with boundaries where lower<0 and upper>0 (i.e. symlog)
            bounds_sign = np.sign(bounds_array)
            if any(bounds_sign[:, 0] != bounds_sign[:, 1]):
                print("Boundaries with different signs found")
                raise ValueError
            bounds_array = np.log(np.abs(bounds_array))

        # Generate the sample
        sample = stats.qmc.scale(sample, bounds_array[:, 0], bounds_array[:, 1])

        if logsample:
            sample = np.exp(sample) * bounds_sign[:, 0]

        self.sobol_points = pd.DataFrame(sample, columns=self.bounds.keys())

    def _generate_sobol_tasks(
        self,
        scenarioID: int = 0,
        iiasa_formatting: bool = False,
        worker_kwargs: dict = {},
    ):
        """Generate tasks for the parallel processor. The points should already have
        been generated by means of the generator

        Parameters
        ----------
        scenarioID: int (default 0)
            Scenario of the model that is to be run
        iiasa_formatting: bool (default False)
            Whether to also compute the results as an iiasa formatted dictionary
        worker_kwargs: dict (default None)
            Any extra arguments for the worker. E.g. for the iiasa formatting this
            is the parameters passed to model.get_iiasa_results()
        """

        # Generate the tasks
        self.all_tasks = []
        for i in self.sobol_points.index:
            # Copy base parameters to ensure a full set of parameters
            newparams = copy.deepcopy(self.base_parameters)
            for key in self.sobol_points.columns:
                newparams[key] = self.sobol_points.loc[i, key]

            # Generate the task to execute
            newmodel = self.modelclass(parameters=newparams, **self.model_kwargs)
            self.all_tasks.append(
                (
                    newmodel,
                    i,
                    scenarioID,
                    worker_kwargs,
                )
            )

    def _generate_generic_tasks(
        self,
        parametersets: dict,
        scenarioID: int = 0,
        iiasa_formatting: bool = False,
        worker_kwargs: dict = {},
    ):
        """Generate tasks for the parallel processor. The points should already have
        been generated by means of the generator

        Parameters
        ----------
        parametersets: dict
            key:dict pairs of different parameters
        scenarioID: int (default 0)
            Scenario of the model that is to be run
        iiasa_formatting: bool (default False)
            Whether to also compute the results as an iiasa formatted dictionary
        worker_kwargs: dict (default None)
            Any extra arguments for the worker. E.g. for the iiasa formatting this
            is the parameters passed to model.get_iiasa_results()
        """

        # Generate the tasks
        self.all_tasks = []
        for i, pdict in parametersets.items():
            # Copy base parameters to ensure a full set of parameters
            newparams = copy.deepcopy(self.base_parameters)
            for key, v in pdict.items():
                newparams[key] = v

            # Generate the task to execute
            newmodel = self.modelclass(parameters=newparams, **self.model_kwargs)
            self.all_tasks.append(
                (
                    newmodel,
                    i,
                    scenarioID,
                    worker_kwargs,
                )
            )

    def _process_iiasa_results(self, raw_results: list, scenarioID: int = 0):
        """
        Process the results of the sobol sampling in terms of the IIASA formatted
        dataframes. To do so, simply save the frame to CSV

        Parameters
        ----------
        raw_results: list
            Output from the multiprocessing script
        scenarioID:int
            ID of the scenario to save it in the appropriate dictionary key
        """
        # Process the raw results into key:df dict, with df having T columns, K rows
        processed = {
            pid: df.droplevel(["Model", "Unit"])
            for pid, scid, _, df in raw_results
            if isinstance(df, pd.DataFrame)
        }

        if len(processed) == 0:
            return

        # Convert to one large dataframe
        processed = pd.concat(
            processed.values(),
            axis=0,
            keys=processed.keys(),
            names=["ID", "Scenario", "Variable"],
        )
        self.variablelist_iiasa = processed.index.get_level_values("Variable").unique()

        # Append to the output CSV if it exists, else create a new one
        outfile = Path(self.output_folder / "outputs_iiasa.csv")
        if not outfile.is_file():
            processed.to_csv(outfile)
        else:
            processed.to_csv(outfile, mode="a+", header=False)

    def _post_processing(self):
        """As the number of samples increases, as does the number of excel files that would have
        to be read (multiple per parameterization). To simplify later parsing and searching, this
        function sequentially reads the _successfully_ run parameter samples and concatenates them
        into a CSV file

        Parameters
        ----------
        scenarioID: int (default 0)
        """
        # Generate an output csv file to which we will append
        cols = ["scenario", "variable"] + list(
            np.arange(0, self.model.hyperparameters["T"])
        )
        out = pd.DataFrame(columns=cols)
        out.index.name = "id"
        out.to_csv(self.output_folder / "outputs.csv")

        # Gather successful runs
        runlist = sorted(
            [
                int(i.split("_")[0])
                for i in os.listdir(self.output_folder / "outputs")
                if i != "logs"
            ]
        )

        # Process them individually
        for runid in tqdm(runlist, desc="Post-Processing"):
            # Get scenario IDs
            tables = os.listdir(self.output_folder / "outputs" / f"{runid}_tables")
            scenarioIDs = [int(i.split("_")[1]) for i in tables]

            for scenarioID in scenarioIDs:
                # Load the completed file
                df = pd.read_excel(
                    self.output_folder
                    / "outputs"
                    / f"{runid}_tables"
                    / f"{runid}_{scenarioID}_results.xlsx",
                    sheet_name="Results",
                )
                df = df.loc[: self.model.hyperparameters["T"], :]
                # Columns = time, index = variables
                df = df.T
                self.variablelist = df.index.to_list()
                df.index = pd.MultiIndex.from_product([[runid], [scenarioID], df.index])
                # Append
                df.to_csv(self.output_folder / "outputs.csv", mode="a+", header=False)
