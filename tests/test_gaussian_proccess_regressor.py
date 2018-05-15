import pytest
from pandas import DataFrame
from sklearn import gaussian_process

from projectpredict import TimeUnits, DurationPdf
from projectpredict.learningmodels import GaussianProcessRegressorModel
from projectpredict.pdf import GaussianPdf
import mock
import sys


def test_init():
    model = GaussianProcessRegressorModel()
    assert isinstance(model.model, gaussian_process.GaussianProcessRegressor)
    assert not model.is_trained
    assert model.units == TimeUnits.seconds


def test_init_import_error(mocker):
    with mock.patch.dict(sys.modules, {'sklearn': None}):
        with pytest.raises(ImportError):
            GaussianProcessRegressorModel()
    with mock.patch.dict(sys.modules, {'pandas': None}):
        with pytest.raises(ImportError):
            GaussianProcessRegressorModel()


def test_init_with_units():
    model = GaussianProcessRegressorModel(units=TimeUnits.days)
    assert isinstance(model.model, gaussian_process.GaussianProcessRegressor)
    assert not model.is_trained
    assert model.units == TimeUnits.days


def test_init_with_kernel():
    kernel = gaussian_process.kernels.Matern(length_scale=1, nu=3.0 / 2)
    model = GaussianProcessRegressorModel(kernel=kernel)
    assert isinstance(model.model.kernel, gaussian_process.kernels.Matern)
    assert model.model.kernel.nu == 1.5
    assert model.model.kernel.length_scale == 1
    assert isinstance(model.model, gaussian_process.GaussianProcessRegressor)
    assert not model.is_trained
    assert model.units == TimeUnits.seconds


def test_train_with_dataframe(mocker):
    model = GaussianProcessRegressorModel()
    mock_fit = mocker.patch.object(model.model, 'fit')
    inputs = DataFrame(data={'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    durations = [4, 5, 6]
    model.train(inputs, durations)
    assert model.is_trained
    mock_fit.assert_called_once_with(inputs, durations)
    assert model.ordering == ['col1', 'col2']


def test_train_with_list_no_ordering():
    model = GaussianProcessRegressorModel()
    inputs = [1, 2, 3]
    durations = [4, 5, 6]
    with pytest.raises(ValueError):
        model.train(inputs, durations)


def test_train_with_list(mocker):
    model = GaussianProcessRegressorModel()
    mock_fit = mocker.patch.object(model.model, 'fit')
    inputs = [1, 2, 3]
    durations = [4, 5, 6]
    model.train(inputs, durations, ordering=['a', 'b'])
    assert model.is_trained
    mock_fit.assert_called_once_with(inputs, durations)
    assert model.ordering == ['a', 'b']


def test_predict_untrained():
    model = GaussianProcessRegressorModel()
    model.is_trained = False
    with pytest.raises(ValueError):
        model.train([1, 2, 3], [4, 5, 6])


def test_predict(mocker):
    model = GaussianProcessRegressorModel(units=TimeUnits.days)
    mock_fit = mocker.patch.object(model.model, 'fit')
    mock_predict = mocker.patch.object(model.model, 'predict', return_value=([1], [2]))
    inputs = [[1, 2, 3], [2, 5, 4]]
    durations = [4, 5, 6]
    model.train(inputs, durations, ordering=['a', 'b'])
    assert model.predict({'a': 1, 'b': 2}) == DurationPdf(GaussianPdf(1, 2 ** 2), units=TimeUnits.days)
    mock_predict.assert_called_once_with([[1, 2]], return_std=True)
