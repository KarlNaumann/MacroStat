# -*- coding: utf-8 -*-
"""
Generic model class as a wrapper to specific implementations
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging
import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger(__name__)



class Model:
    """A class representing a macroeconomic model.

    This class provides a wrapper for users to write their underlying model
    behavior while maintaining a uniformly accessible interface. Specifically,
    the user is expected to adapt the model.simulate() function to their needs,
    respecting only that the return of that function is a pandas dataframe.

    Attributes
    ----------
    name: str
        Name of the model, such as "model". Used for file and database names
    parameters : dict
        Dictionary of all parameters
    hyper_parameters : dict
        Dictionary of all hyperparameters
    output : pd.DataFrame
        None, or the latest simulation run for given parameters


    Example
    -------
    A general workflow for a model might look like

    >>> model = Model(parameters, hyper_parameters)
    >>> output = model.simulate()
    >>> model.save()

    """

    def __init__(
        self,
        parameters: dict,
        hyper_parameters: dict,
        name: str = "model",
    ):
        """Initialization of the model class.

        If SQL is true, will check for the existing model database or create a
        new one if none is found.

        Parameters
        ----------
        parameters : dict
            dictionary of the named parameters of the model
        hyper_parameters : dict
            dictionary of hyper-parameters related to the model
        initial_conditions : dict
            dictionary of the models initial values
        name : str (default 'model')
            name of the model (for use in filenaming)
        sql : bool (default True)
            whether to use a sqlite3 database to store output or not
        db_path : str (default output/)
            directory where the database is stored
        db_name : str (default None)
            filename of the database
        debug : bool (default False)
            enable to get stepwise printed output
        """
        # Essential attributes
        self.parameters = parameters
        self._validate_parameters()
        self.hyper_parameters = hyper_parameters
        self.name = name

        # Attributes generated later on
        self.output = None

    def simulate(self, *args, **kwargs) -> pd.DataFrame:
        """Simulate a model run using the stored parameters

        This function is designed to be overwritten by the user's 
        specific implementation of their model. Note that it is
        expected for the user to set the ''self.output'' attribute
        to the output generated.

        Returns
        -------
        output : pd.DataFrame
            Output of the model. Generically it should have a "time"-like
            index and variables across the columns
        """
        raise NotImplementedError

    def save(self, path=None):
        """Save the model object as a pickled file
        
        Parameters
        ----------
        path, optional
            path where the model will be stored. If it is None then
            the model's name will be used and the file stored in the
            working directory.

        Notes
        -----
        .. note:: This implementation is dependent on your pickling version
        """
        if path is None:
            path = f"{self.name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path=None):
        """Class method to load a model instance from a pickled file. 

        Parameters
        ----------
        path, optional
            path to the targeted Sampler

        Notes
        -----
        .. note:: This implementation is dependent on your pickling version

        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    
    def __eq__(self, other): 
        """ Check if two models are equivalent in their core attributes """

        if self.parameters.keys() != other.parameters.keys():
            return False
        else:
            parameter_equivalence = [
                np.allclose(v,other.parameters[k]) for k,v in self.parameters.items()
            ]
        
        if self.hyper_parameters.keys() != other.hyper_parameters.keys():
            return False
        else:
            hyper_parameter_equivalence = []
            for k,v in self.hyper_parameters.items():
                try:
                    if np.issubdtype(np.array(v).dtype, np.number):
                        hyper_parameter_equivalence.append(np.allclose(v,other.hyper_parameters[k]))
                    else:
                        hyper_parameter_equivalence.append(v == other.hyper_parameters[k])
                except Exception:
                    hyper_parameter_equivalence.append(False)

        if all([self.output is not None, other.output is not None]):
            output_equivalence = np.allclose(self.output, other.output)
        else:
            output_equivalence = all([self.output is None, other.output is None])

        conditions = [
            all(parameter_equivalence),
            all(hyper_parameter_equivalence),
            self.name == other.name,
            output_equivalence
        ]

        return all(conditions)

    def _validate_parameters(self):
        """ Validate whether all of the parameters are numeric.

        We assume that the parameter dictionary may contain numeric
        values, whether these are float/int or arrays. We verify this
        by checking whether the cast to array of a given 
        """
        condition = {}
        for k, v in self.parameters.items():
            condition[k] = np.issubdtype(np.array(v).dtype, np.number)

        if not all(condition.values()):
            fails = [k for k, v in condition.items() if not v]
            raise ValueError(f"Parameters are not all numeric: {fails}")