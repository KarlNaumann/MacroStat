"""
Batch processing functionionality
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from concurrent.futures import ProcessPoolExecutor


def timeseries_worker(task: tuple):
    """Worker function for parallel_processor, which will execute a
    simulation with the given parameters and return the output.

    Parameters
    ----------
    task : tuple
        Tuple of (name, model, *args) where name is the name of the
        simulation, model is the model to be simulated and *args are
        the arguments to be passed to the model's simulate method.

    Returns
    -------
    tuple
        Tuple of (name, *args, output) where name is the name of the
        simulation, *args are the arguments passed to the model's
        simulate method and output is the output of the simulation.
    """
    model = task[1]
    _ = model.simulate(*task[2:])
    return (task[0], *task[2:], model.output)


def parallel_processor(
    tasks: list = [],
    worker: callable = timeseries_worker,
    cpu_count: int = 1,
):
    """Run all of the tasks in parallel using the
    ProcessPoolExecutor.

    Parameters
    ----------
    tasks : list[tuple]
        List of tasks to be processed in parallel.
        Each task should be a tuple
    worker : callable
        Worker function to be used for the parallel processing.
        Each task will be passed to the worker function as a tuple
    cpu_count : int (default=1)
        Number of CPUs to be used for the parallel processing.

    Returns
    -------
    list
        List of tuple results from the worker function
    """
    if len(tasks) == 0:
        raise ValueError("No tasks to process.")

    results = []
    process_count = min(cpu_count, len(tasks))
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for i in executor.map(worker, tasks):
            results.append(i)
    return results
