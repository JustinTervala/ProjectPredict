from datetime import timedelta, datetime

import pytest
from scipy.stats import halfnorm

from projectpredict.pdf import GaussianPdf, DeterministicPdf, TimeUnits, DurationPdf, DatePdf, SciPyPdf


def test_scipy_pdf_init():
    halfnorm_pdf = halfnorm(loc=1, scale=3)
    pdf = SciPyPdf(halfnorm_pdf)
    assert pdf.pdf is halfnorm_pdf


def test_scipy_pdf_mean_variance():
    halfnorm_pdf = halfnorm(loc=1, scale=3)
    pdf = SciPyPdf(halfnorm_pdf)
    assert pdf.mean == halfnorm_pdf.mean()
    assert pdf.variance == halfnorm_pdf.var()


def test_scipy_pdf_sample(mocker):
    halfnorm_pdf = halfnorm(loc=1, scale=3)
    pdf = SciPyPdf(halfnorm_pdf)
    mock_rvs = mocker.patch.object(pdf.pdf, 'rvs')
    pdf.sample()
    mock_rvs.assert_called_once()


def test_gaussian_pdf():
    pdf = GaussianPdf(0, 1)
    pdf.pdf.rvs = lambda: 0.25
    assert pdf.sample() == 0.25


def test_gaussian_pdf_equals():
    assert GaussianPdf(10, 4) == GaussianPdf(10, 4)
    assert GaussianPdf(10, 4) != GaussianPdf(10, 3)
    assert GaussianPdf(10, 4) != GaussianPdf(11, 4)


def test_gaussian_pdf_from_dict():
    dict_in = {'mean': 12, 'variance': 100}
    pdf = GaussianPdf.from_dict(dict_in)
    assert pdf.mean == 12
    assert pdf.variance - 100 <= 1e-6


def test_deterministic_pdf():
    value = 42
    pdf = DeterministicPdf(value)
    assert pdf.mean == value
    assert pdf.sample() == value
    assert pdf.variance == 0


def test_deterministic_pdf_equals():
    assert DeterministicPdf(12) == DeterministicPdf(12)


@pytest.mark.parametrize('test_input, expected', [
    ({'value': 12, 'units': TimeUnits.milliseconds}, timedelta(milliseconds=12)),
    ({'value': 42, 'units': TimeUnits.seconds}, timedelta(seconds=42)),
    ({'value': 13, 'units': TimeUnits.minutes}, timedelta(minutes=13)),
    ({'value': 3, 'units': TimeUnits.hours}, timedelta(hours=3)),
    ({'value': -1, 'units': TimeUnits.days}, timedelta(days=-1)),
    ({'value': 132, 'units': TimeUnits.weeks}, timedelta(weeks=132))
])
def test_time_units_to_timedelta(test_input, expected):
    assert TimeUnits.to_timedelta(test_input['units'], test_input['value']) == expected


@pytest.mark.parametrize('test_input, expected', [
    ('milliseconds', TimeUnits.milliseconds),
    ('seconds', TimeUnits.seconds),
    ('minutes', TimeUnits.minutes),
    ('hours', TimeUnits.hours),
    ('days', TimeUnits.days),
    ('weeks', TimeUnits.weeks)
])
def test_time_units_from_string(test_input, expected):
    assert TimeUnits.from_string(test_input) == expected


def test_time_units_from_string_invalid_str():
    with pytest.raises(ValueError):
        TimeUnits.from_string('invalid')


def test_duration_pdf(mocker):
    mocker.patch.object(GaussianPdf, 'sample', return_value=0.25)
    pdf = DurationPdf(GaussianPdf(0, 1))
    assert pdf.sample() == timedelta(seconds=0.25)


def test_duration_pdf_with_timeunit(mocker):
    mocker.patch.object(GaussianPdf, 'sample', return_value=0.25)
    pdf = DurationPdf(GaussianPdf(0, 1), units=TimeUnits.days)
    assert pdf.sample() == timedelta(days=0.25)


def test_duration_sample_with_minimum(mocker):
    class MockPdf(object):
        def __init__(self):
            self.first = False

        def sample(self):
            if self.first:
                self.first = False
                return 2.5
            else:
                return 10

    pdf = DurationPdf(MockPdf(), units=TimeUnits.days)
    assert pdf.sample(minimum=timedelta(days=5)) == timedelta(days=10)


def test_duration_pdf_equals():
    assert DurationPdf(GaussianPdf(10, 4)) == DurationPdf(GaussianPdf(10, 4))
    assert DurationPdf(GaussianPdf(10, 4), units=TimeUnits.seconds) != DurationPdf(GaussianPdf(10, 4),
                                                                                   units=TimeUnits.days)
    assert DurationPdf(GaussianPdf(10, 4)) != DurationPdf(GaussianPdf(10, 3))


def test_date_pdf(mocker):
    mocker.patch.object(GaussianPdf, 'sample', return_value=0.25)
    date = datetime(year=2018, month=5, day=12)
    pdf = DatePdf(date, GaussianPdf(0, 1))
    assert pdf.sample() == date + timedelta(seconds=0.25)


def test_date_pdf_with_units(mocker):
    mocker.patch.object(GaussianPdf, 'sample', return_value=0.25)
    date = datetime(year=2018, month=5, day=12, hour=12)
    pdf = DatePdf(date, GaussianPdf(0, 1), units=TimeUnits.days)
    assert pdf.sample() == date + timedelta(days=0.25)


def test_date_pdf_equals():
    date = datetime(year=2018, month=5, day=12, hour=12)
    assert DatePdf(date, GaussianPdf(0, 1), units=TimeUnits.days) == DatePdf(date, GaussianPdf(0, 1),
                                                                             units=TimeUnits.days)
    date2 = datetime(year=2017, month=5, day=12, hour=12)
    assert DatePdf(date, GaussianPdf(0, 1), units=TimeUnits.days) != DatePdf(date2, GaussianPdf(0, 1),
                                                                             units=TimeUnits.days)
    assert DatePdf(date, GaussianPdf(0, 1), units=TimeUnits.days) != DatePdf(date, GaussianPdf(10, 1),
                                                                             units=TimeUnits.days)
    assert DatePdf(date, GaussianPdf(0, 1), units=TimeUnits.days) != DatePdf(date, GaussianPdf(10, 1),
                                                                             units=TimeUnits.seconds)
