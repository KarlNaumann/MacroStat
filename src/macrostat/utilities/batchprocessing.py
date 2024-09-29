"""
Batch processing functionionality
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def timeseries_worker(task: tuple):
    """Worker function for parallel_processor, which will execute a
    simulation with the given parameters and return the output.

    Parameters
    ----------
    task : tuple
        Tuple of model, name (simulation_id), and scenarioID

    Returns
    -------
    tuple
        Tuple of simulation_id, scenarioID and output from the simulation.
    """
    model = task[0]
    _ = model.simulate(*task[1:])
    return (*task[1:], model.output)


class BatchProcessing:
    def parallel_processor(
        self, worker: callable = timeseries_worker, tqdm_info: str = ""
    ):
        """Run all of the tasks in parallel using the
        ProcessPoolExecutor.

        Parameters
        ----------
        worker : callable
            Worker function to be used for the parallel processing.
            Default is timeseries_worker.
        tqdm_info : str
            Information to be displayed in the tqdm progress bar.

        Returns
        -------
        list
            List of tuple results from the worker function
        """
        results = []
        process_count = min(self.cpu_count, len(self.tasks))
        tqdmargs = dict(total=len(self.tasks), desc=tqdm_info)
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            for i in tqdm(executor.map(worker, self.tasks), **tqdmargs):
                results.append(i)
        return results
