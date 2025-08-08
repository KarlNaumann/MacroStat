"""
Batch processing functionionality
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging
import traceback
from contextlib import contextmanager

import torch.multiprocessing as mp
from torch.multiprocessing import Pool

logger = logging.getLogger(__name__)


@contextmanager
def pool_context(*args, **kwargs):
    """Context manager for process pool to ensure proper cleanup."""
    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        logger.debug("Cleaning up process pool")
        pool.terminate()
        pool.join()
        logger.debug("Process pool cleanup completed")


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
    try:
        model = task[1]
        _ = model.simulate(*task[2:])
        return (task[0], *task[2:], model.output)
    except Exception as e:
        logger.error(f"Worker failed for task {task[0]}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def parallel_processor(
    tasks: list = [],
    worker: callable = timeseries_worker,
    cpu_count: int = 1,
):
    """Run all of the tasks in parallel using the ProcessPoolExecutor."""

    # Set multiprocessing start method to spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Set sharing strategy
    mp.set_sharing_strategy("file_system")

    if len(tasks) == 0:
        raise ValueError("No tasks to process.")

    process_count = min(cpu_count, len(tasks))
    logger.debug(f"Creating process pool with {process_count} workers")

    try:
        with pool_context(processes=process_count) as pool:
            logger.debug("Process pool created successfully")
            results = pool.map(worker, tasks)
            logger.debug("Parallel processing completed")
            return results
    except Exception as e:
        logger.error(f"Error in process pool: {str(e)}")
        logger.error(traceback.format_exc())
        raise
