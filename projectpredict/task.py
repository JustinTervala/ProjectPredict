from __future__ import division

from datetime import datetime, timedelta
from uuid import uuid4

from projectpredict.pdf import GaussianPdf, TimeUnits, DurationPdf, DatePdf


class Entity(object):
    """Base class for entities which provides a UUID and a hashability to the child classes

    Attributes:
        uid (UUID): The UUID of the object
        name (str): The name of the object

    Args:
        uid (UUID, optional): The UUID of the object
        name (str, optional): The name of the object
    """

    def __init__(self, uid=None, name=''):
        self.uid = uid or uuid4()
        self.name = name

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(str(self.uid))

    def __repr__(self):
        return '<{} name={}>'.format(
            self.__class__.__name__,
            self.name)


class Task(Entity):
    """A task in the project or overall process

    Attributes:
        project_uid (UUID): The UUID of the project containing this task
        duration_pdf (projectpredict.pdf.DurationPdf): A pdf to use to sample the duration of the task
        earliest_start_date_pdf (projectpredict.pdf.DatePdf): A pdf to use to sample the earliest start date of of the task
        latest_finish_date_pdf (projectpredict.pdf.DatePdf): A pdf to use to sample the latest finish date of the task
        start_time (datetime): The datetime the task was started
        completion_time (datetime): The datetime the task was completed
        data: Any data associated with this task.
        deadline_weight: The weight attached to the deadline for this task.

    Args:
        name (str): The name of the task
        uid (UUID, optional): The UUID of the task. If none is provided, one will be generated.
        project_uid (UUID, optional): The UUID of the project containing this task
        duration_pdf (projectpredict.pdf.DurationPdf): A pdf to use to sample the duration of the task
        earliest_start_date_pdf (DatePdf, optional): A pdf to use to sample the earliest start date of of the task
        latest_finish_date_pdf (DatePdf, optional): A pdf to use to sample the latest finish date of the task
        data (optional): Any data associated with this task.
        deadline_weight (int, optional): The weight attached to meeting this task's deadline
    """

    def __init__(
            self,
            name,
            uid=None,
            project_uid=None,
            duration_pdf=None,
            earliest_start_date_pdf=None,
            latest_finish_date_pdf=None,
            data=None,
            deadline_weight=1):
        super(Task, self).__init__(name=name, uid=uid)
        self.project_uid = project_uid
        self.duration_pdf = duration_pdf
        self.earliest_start_date_pdf = earliest_start_date_pdf
        self.latest_finish_date_pdf = latest_finish_date_pdf
        self.data = data
        self.deadline_weight = deadline_weight
        self.start_time = None
        self.completion_time = None

    def start(self, start_time=None):
        """Marks the task as started

        Args:
            start_time (datetime, optional): The datetime the task was started. Defaults to the current UTC timestamp
        """
        self.start_time = start_time if start_time is not None else datetime.utcnow()

    def complete(self, completion_time=None):
        """Completes the task

        Args:
            completion_time (datetime, optional): The datetime the task was completed. Defaults to the current UTC
                timestamp
        """
        self.completion_time = completion_time if completion_time is not None else datetime.utcnow()

    @property
    def is_completed(self):
        """bool: Is the task completed?
        """
        return self.completion_time is not None

    @property
    def is_started(self):
        """bool: Has the task been started?
        """
        return self.start_time is not None

    @property
    def mean_duration(self):
        """timedelta: Gets the mean of the duration pdf
        """
        return self.duration_pdf.mean

    def set_duration_pdf(self, model):
        """Sets the duration PDF from a model

        Args:
            model: The model to use to predict the duration of the task
        """
        self.duration_pdf = model.predict(self.data)

    def set_earliest_start_pdf(self, mean_datetime, std, units=TimeUnits.seconds):
        """Sets the earliest start date pdf as a normal distributirequired=Trueon about a mean date.

        Args:
            mean_datetime (datetime): The mean datetime of the earliest time a task can start
            std (float): The standard deviation of the distribution
            units (TimeUnits, optional): The units of time of the variance. Defaults to TimeUnits.seconds
        """
        self.earliest_start_date_pdf = DatePdf(mean_datetime, GaussianPdf(0, std ** 2), units=units)

    def set_latest_finish_pdf(self, mean_datetime, std, units=TimeUnits.seconds):
        """Sets the latest finish date pdf as a normal distribution about a mean date.

        Args:
            mean_datetime (datetime): The mean datetime of the latest time a task can finish
            std (float): The standard deviation of the distribution
            units (TimeUnits, optional): The units of time of the variance. Defaults to TimeUnits.seconds
        """
        self.latest_finish_date_pdf = DatePdf(mean_datetime, GaussianPdf(0, std ** 2), units=units)

    def get_duration_sample(self, current_time):
        """Gets a sample of the duration.

        If the task has already started, then only durations greater than current_time - start_time will be valid, and
        samples will be drawn until a valid duration is picked.

        Args:
            current_time (datetime): The current time at which the sample should be drawn from.

        Returns:
            timedelta: A sample of the duration pdf
        """
        minimum_duration = current_time - self.start_time if self.start_time else None
        return self.duration_pdf.sample(minimum=minimum_duration)

    def get_earliest_start_sample(self, current_time):
        """Gets a sample of the earliest start date pdf

        If a task has been started, this will always return the start time. Else if an earliest start date pdf has been
        provided, a sample is drawn from that distribution. If no distribution has ben provided, the current time is
        returned.

        Args:
            current_time (datetime): The current time at which the sample should be drawn from.

        Returns:
            datetime: A sample from the earliest start date pdf.
        """
        if self.is_started:
            return self.start_time
        elif self.earliest_start_date_pdf is not None:
            return self.earliest_start_date_pdf.sample()
        else:
            return current_time

    def get_latest_finish_sample(self):
        """Gets a sample of the latest finish date pdf

        If an latest finish date pdf has been provided, a sample is drawn from that distribution. else, this function
        will return None

        Returns:
            datetime: A sample from the latest start date pdf
        """
        if self.latest_finish_date_pdf is not None:
            return self.latest_finish_date_pdf.sample()
        else:
            return None

    @classmethod
    def from_pert(cls, name, best_case, estimated, worst_case, units=TimeUnits.seconds, **kwargs):
        """Constructs a Task from three-point (PERT) estimations.

        Args:
            name (str): The name of the task
            best_case (float): The estimated best case duration of the task
            estimated (float): The estimated duration of the task
            worst_case (float): The estimated worst case duration of the task
            units (TimeUnits, optional): The units of time used in the estimation. Defaults to TimeUnits.seconds
            **kwargs: Arguments to be passed into Task constructor

        Returns:
            Task: A task constructed from the provided arguments
        """
        mean = (best_case + 4 * estimated + worst_case) / 6
        std = (worst_case - best_case) / 6
        return cls(name, duration_pdf=DurationPdf(GaussianPdf(mean, std ** 2), units=units), **kwargs)
