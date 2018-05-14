from collections import OrderedDict

from projectpredict.pdf import DurationPdf, GaussianPdf
from projectpredict.task import TimeUnits


class GaussianProcessRegressorModel(object):
    """Learns the duration of a task from data using scikit-learn's GaussianProcessRegressor

    Attributes:
        model (GaussianProcessRegressor): The underlying model used to predict the data
        units (TimeUnits, optional): The time units the resulting durations should be in. Defaults to TimeUnits.seconds
        is_trained (bool): A boolean value indicating if the model has been trained.
        ordering (list[str]): The ordering of the input data used to construct input data

    Args:
        units (TimeUnits, optional): The time units the resulting durations should be in. Defaults to TimeUnits.seconds

    Keyword Args:
        kernel: The kernel to use in the regressor model. Defaults to
            ConstantKernel() + Matern(length_scale=1, nu=3 / 2) + WhiteKernel(noise_level=1)
    """
    def __init__(self, units=TimeUnits.seconds, **kwargs):
        try:
            from sklearn import gaussian_process
            from pandas import DataFrame
        except ImportError:
            print('GaussianProcessRegressorModel requires scikit-learn and pandas')
            raise
        if 'kernel' not in kwargs:
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
            kwargs['kernel'] = ConstantKernel() + Matern(length_scale=1, nu=3 / 2) + WhiteKernel(noise_level=1)
        self.model = gaussian_process.GaussianProcessRegressor(**kwargs)
        self.units = units
        self.is_trained = False
        self.ordering = None
        self.__dataframe_class = DataFrame

    def train(self, input_data, durations, ordering=None):
        """Trains the model from input data and durations

        Note:
            If a Pandas DataFrame is used for the input data, the ordering of the data will be determined by the
            ordering of the colunms. If a pandas DataFrame is not used, then the ordering will need to be provided. Each
            Task must provide data as a dictionary in which the keys are the same as the names in the ordering/column
            names of the DataFrame

        Args:
            input_data (array-like): The data to train the data from
            durations (array-like): The durations associated with the data
            ordering (list[str], optional): The ordering of the data

        Raises:
            ValueError: When a non-DataFrame is provided as the input_data and no ordering is provided

        """
        if isinstance(input_data, self.__dataframe_class):
            self.ordering = [
                key for key, value in OrderedDict(input_data.dtypes).items() if value in ('float64', 'int64')]
        else:
            if ordering is None:
                raise ValueError('Ordering must be provided for non-pandas dataframe inputs')
            self.ordering = ordering

        self.model.fit(input_data, durations)
        self.is_trained = True

    def predict(self, input_data):
        """Predicts the duration of a task given its data


        Args:
            input_data (dict): A dict containing the data necessary to predict the duration. The format must be as
            key-value pairs in which the key is the name of the data and the value is its value.

        Returns:
            DurationPdf: The estimated duration of the task.
        """
        if not self.is_trained:
            raise ValueError('Cannot predict with untrained model')
        ordered_input_data = [[input_data[field] for field in self.ordering]]
        predicted = self.model.predict(ordered_input_data, return_std=True)
        return DurationPdf(GaussianPdf(predicted[0][0], predicted[1][0]**2), units=self.units)
