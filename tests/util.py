from projectpredict import DatePdf, TimeUnits, Task, DurationPdf
from projectpredict.pdf import DeterministicPdf
from projectpredict.project import TaskStatistics
from datetime import datetime, timedelta


def make_deterministic_date_pdf(date, value):
    return DatePdf(date, DeterministicPdf(value), units=TimeUnits.days)


def task_factory(name, value):
    return Task(name, duration_pdf=DurationPdf(DeterministicPdf(value), units=TimeUnits.days))


def make_task_stat(current_date, means, variances, units):
    converted_means = [current_date + TimeUnits.to_timedelta(units, mean) for mean in means]
    converted_means[-1] = TimeUnits.to_timedelta(units, means[-1])
    variances = [TimeUnits.to_timedelta(units, variance) for variance in variances]
    stats = [{'mean': mean, 'variance': variance} for mean, variance in zip(converted_means, variances)]
    return TaskStatistics(*stats)


class MockModel:
    def predict(self, data):
        return DurationPdf(DeterministicPdf(data['a']))