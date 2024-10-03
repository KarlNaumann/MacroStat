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
from tqdm import tqdm

# Custom imports
import macrostat.models.model as msmodel
import macrostat.util.batchprocessing as msbatchprocessing


class Sampler():
    def __init__(
        self,
        model: msmodel.Model,
        worker_function: callable = msbatchprocessing.timeseries_worker,
        output_folder: str = "samples",
        cpu_count: int = 1,
        batchsize: int = None,
    ):
        """Generalized class to facilitate the sampling of the model's 
        parameterspace using python's multiprocessing library.

        Parameters
        ----------
        model: msmodel.Model
            Model to be sampled
        worker_function: callable (default batchprocessing.timeseries_worker)
            Function to be used for the parallel processing
        output_folder: str (default "samples")
            Folder to save the output files
        cpu_count: int (default 1)
            Number of CPUs to use for the parallel processing
        batchsize: int (default None)
            Size of each batch to be processed in parallel
        """
        # Model parameters
        self.model = model
        self.modelclass = type(model)
        self.base_parameters = copy.deepcopy(model.parameters)
        self.bounds = None

        # Store all possible attributes set in the model
        initargs = [
            i for i in inspect.signature(model.__init__).parameters if i != "self"
        ]
        self.model_kwargs = {a: getattr(self.model, a) for a in initargs if a!="parameters"}


        # Computation parameters
        self.worker_function = worker_function
        self.cpu_count = min([mp.cpu_count(), cpu_count])
        self.batchsize = batchsize
        self.output_folder = Path(output_folder)
        os.makedirs(output_folder, exist_ok=True)

    def generate_tasks(self, *args, **kwargs) -> list[tuple]:
        """Generate tasks for the parallel processor. 

        This method should return a list of tuples that will be passed to 
        the worker function. By default, the first item in the tuple is 
        the model object, and all remaining items are the arguments that 
        will be passed to the model.simulate() function.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def sample(self, tqdm_info: str = "Sampling"):        
        """Run in parallel the sampling of the model's parameterspace
        by generating a set of tasks and executing them in parallel

        Parameters
        ----------
        tqdm_info: str (default "Sampling")
            Information to be displayed in the tqdm progress bar
        """
        # Generate the tasks to run
        self.tasks = self.generate_tasks()

        # Save the parameters
        parameters = {v[0]: v[1].parameters for v in self.tasks}
        parameters = pd.DataFrame(parameters).T.to_csv(
            self.output_folder / "parameters.csv", index_label="id"
        )

        # Run the parallel processing in batches to conserve memory
        # This will write results to disk, clear memory, and proceed
        if self.batchsize is None:
            self.batchsize = len(self.tasks)
        
        batchcount = int(len(self.tasks) / self.batchsize) + (
            len(self.tasks) % self.batchsize > 0
        )

        for batch in range(batchcount):
            # Set tasks to run now
            batch_tasks = self.tasks[
                batch * self.batchsize : min([(batch + 1) * self.batchsize, len(self.tasks)])
            ]
            # Execute those tasks
            raw_outputs = msbatchprocessing.parallel_processor(
                tasks=batch_tasks,
                worker=self.worker_function, 
                cpu_count=self.cpu_count,
                tqdm_info=tqdm_info
                )

            # Save the outputs to disk
            self.save_outputs(raw_outputs, batch=batch)

    def save_outputs(self, raw_outputs: list, batch: int):
        """ Save the raw outputs to disk. 

        The model's outputs are in the form of a pandas DataFrame.
        This method should save the outputs to disk in a format that
        can be easily read back in later. Generically, it writes a 
        CSV file with the outputs in a MultiIndex format. However,
        this can be overwritten to save in a different format.

        Parameters
        ----------
        raw_outputs: list
            List of outputs from the parallel processing. By default,
            batchprocessing.timeseries_worker returns a tuple of
            (*task_arguments, output)
        batch: int (default None)
            Batch number to save the outputs. Assumes that
            the batchsize is constant.
        """
        # Concatenate the outputs
        index_names = list(raw_outputs[0][-1].index.names)
        data = {v[0]: v[-1] for v in raw_outputs}
        data = pd.concat(data.values(), keys=data.keys(), names=["ID"]+index_names, axis=0)

        self.index_count = data.index.nlevels
        self.header_count = data.columns.nlevels

        # Check if the file exists
        outputfile = self.output_folder / "outputs.csv"
        if not os.path.exists(outputfile):
            data.to_csv(outputfile, header=True)
        else:
            data.to_csv(outputfile, mode="a", header=False)

    def extract(self, columns: list = None, indices: list = None, chunksize: int = 100000):
        """Extract the results from the output file.
        
        The function uses a pandas chunkreader to extract the data from the
        output file. It is possible to extract only a subset of the columns,
        parameter IDs, or indices. This reduces the memory footprint when 
        dealing with a large number of parameterizations.

        Parameters
        ----------
        columns: list
            List of columns to extract
        pids: list
            List of parameter IDs to extract i.e. the batch number
        indices: list
            List of indices to extract
        chunksize: int (default 100000)
            Chunksize to read in the data
        """
        filename = "outputs.csv"

        header_count = 1
        if columns is not None and isinstance(columns[0], tuple):
            header_count = len(columns[0])

        index_count = 1
        if indices is not None and isinstance(indices[0], tuple):
            index_count = len(indices[0])

        csv_kwargs = dict(
            header=np.arange(header_count), 
            index_col=np.arange(index_count+1)
        )

        # Get the columns to extract from the file
        header = pd.read_csv(self.output_folder / filename, nrows=0, **csv_kwargs)
        column_targets = header.columns if columns is None else columns

        # Get the indices to extract from the file (add slice(None) to the front)
        # The indices may be a list of tuples or of non-iterable objects
        if indices is not None:
            index_targets = indices
        else:
            index_targets = None

        # Read in chunks
        reader = pd.read_csv(
            self.output_folder / filename, chunksize=chunksize, iterator=True, **csv_kwargs
        )

        # Extract the data
        output = []
        for i, chunk in tqdm(enumerate(reader), desc="Chunk Reading"):

            # Match the columns
            if column_targets is not None:
                ix = chunk.columns.isin(column_targets)
                chunk = chunk.loc[:, ix]

            # Match the index
            if index_targets is not None:
                if index_count == 1:
                    chunk = chunk.loc[chunk.index.isin(index_targets)]
                else:
                    masks = [True*np.ones(chunk.shape[0])]
                    for i in np.arange(index_count):
                        masks.append(chunk.index.isin([j[i] for j in index_targets], level=i))
                    chunk = chunk.loc[np.all(masks, axis=0), :]

            output.append(chunk)
        output = pd.concat(output, axis=0)
        return output

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