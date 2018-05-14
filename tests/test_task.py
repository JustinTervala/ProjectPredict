import mock
from pytest import fixture

from projectpredict.task import *


@fixture
def task():
    pdf = DurationPdf(GaussianPdf(0, 1))
    return Task('task1', duration_pdf=pdf)


@fixture
def task_with_earliest_start():
    duration_pdf = DurationPdf(GaussianPdf(0, 1))
    earliest_start_pdf = DatePdf(datetime(year=2018, month=5, day=1), GaussianPdf(0, 1), units=TimeUnits.days)
    return Task('task1', duration_pdf=duration_pdf, earliest_start_date_pdf=earliest_start_pdf)


def test_task_start(task, mocker):
    default_datetime = datetime(year=2018, month=5, day=12,  hour=12)
    mock_datetime = mocker.patch('projectpredict.task.datetime')
    mock_datetime.utcnow = mock.Mock(return_value=default_datetime)
    assert not task.is_started
    task.start()
    assert task.start_time == default_datetime
    assert task.is_started


def test_task_start_with_time(task):
    default_datetime = datetime(year=2018, month=5, day=12,  hour=12)
    task.start(start_time=default_datetime)
    assert task.start_time == default_datetime


def test_task_complete(task, mocker):
    default_datetime = datetime(year=2018, month=5, day=12,  hour=12)
    mock_datetime = mocker.patch('projectpredict.task.datetime')
    mock_datetime.utcnow = mock.Mock(return_value=default_datetime)
    assert not task.is_completed
    task.complete()
    assert task.completion_time == default_datetime
    assert task.is_completed


def test_task_complete_with_time(task):
    default_datetime = datetime(year=2018, month=5, day=12,  hour=12)
    task.complete(completion_time=default_datetime)
    assert task.completion_time == default_datetime


def test_set_duration_pdf(mocker):
    data = [1, 2, 3]
    task = Task('a', data=data)

    class MockModel:

        def predict(self, data_):
            return sum(data)

    model = MockModel()
    mocker.spy(model, 'predict')
    assert task.data == data
    task.set_duration_pdf(model)
    assert task.duration_pdf == sum(data)
    assert model.predict.call_count == 1


def test_task_set_latest_finish_pdf(task):
    date = datetime(year=2018, month=5, day=12,  hour=12)
    task.set_latest_finish_pdf(date, 10)
    assert task.latest_finish_date_pdf.mean_datetime == date
    assert task.latest_finish_date_pdf.units == TimeUnits.seconds
    assert task.latest_finish_date_pdf.pdf.pdf.mean() == 0
    assert task.latest_finish_date_pdf.pdf.pdf.std() == 10


def test_task_set_latest_finish_pdf_with_units(task):
    date = datetime(year=2018, month=5, day=12,  hour=12)
    task.set_latest_finish_pdf(date, 10, units=TimeUnits.days)
    assert task.latest_finish_date_pdf.mean_datetime == date
    assert task.latest_finish_date_pdf.units == TimeUnits.days
    assert task.latest_finish_date_pdf.pdf.pdf.mean() == 0
    assert task.latest_finish_date_pdf.pdf.pdf.std() == 10


def test_task_set_earliest_start_pdf(task):
    date = datetime(year=2018, month=5, day=12,  hour=12)
    task.set_earliest_start_pdf(date, 10)
    assert task.earliest_start_date_pdf.mean_datetime == date
    assert task.earliest_start_date_pdf.units == TimeUnits.seconds
    assert task.earliest_start_date_pdf.pdf.pdf.mean() == 0
    assert task.earliest_start_date_pdf.pdf.pdf.std() == 10


def test_task_set_earliest_start_pdf_with_units(task):
    date = datetime(year=2018, month=5, day=12,  hour=12)
    task.set_earliest_start_pdf(date, 10, units=TimeUnits.days)
    assert task.earliest_start_date_pdf.mean_datetime == date
    assert task.earliest_start_date_pdf.units == TimeUnits.days
    assert task.earliest_start_date_pdf.pdf.pdf.mean() == 0
    assert task.earliest_start_date_pdf.pdf.pdf.std() == 10


def test_task_sample_duration(task, mocker):
    sample = timedelta(seconds=13.0678)
    mocker.patch.object(task.duration_pdf, 'sample', return_value=sample)
    current_time = datetime(year=2018, month=3, day=11, hour=12, minute=42, second=36)
    assert task.get_duration_sample(current_time) == sample


def test_task_sample_duration_from_started_task():
    class MockPdf(object):
        def __init__(self):
            self.first = False

        def sample(self):
            if self.first:
                self.first = False
                return 2.5
            else:
                return 10

    task = Task('task1', duration_pdf=DurationPdf(MockPdf(), units=TimeUnits.days))

    start_time = datetime(year=2018, month=8, day=2)
    task.start(start_time=start_time)
    current_time = start_time + timedelta(days=5)
    assert task.get_duration_sample(current_time) == timedelta(days=10)


def test_task_sample_earliest_start_date(task_with_earliest_start, mocker):
    sample = timedelta(seconds=13.0678)
    mocker.patch.object(task_with_earliest_start.earliest_start_date_pdf, 'sample', return_value=sample)
    current_time = datetime(year=2018, month=3, day=11, hour=12, minute=42, second=36)
    assert task_with_earliest_start.get_earliest_start_sample(current_time) == sample


def test_task_sample_earliest_start_date_from_started(task_with_earliest_start):
    start_time = datetime.utcnow()
    task_with_earliest_start.start(start_time=start_time)
    assert task_with_earliest_start.get_earliest_start_sample(datetime.utcnow()) == start_time


def test_task_sample_earliest_start_date_no_pdf(task):
    current_time = datetime.utcnow()
    assert task.get_earliest_start_sample(current_time) == current_time


def test_task_from_pert():
    best_case = 1
    estimated = 5
    worst_case = 10
    mean = (best_case + 4 * estimated + worst_case) / 6
    std = (worst_case - best_case) / 6
    expected_duration = DurationPdf(GaussianPdf(mean, std**2))
    task = Task.from_pert('task1', best_case, estimated, worst_case)
    assert task.name == 'task1'
    assert task.duration_pdf == expected_duration

