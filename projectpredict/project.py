from __future__ import division

from collections import namedtuple
from datetime import datetime, timedelta
from itertools import combinations
from uuid import uuid4

import networkx as nx
import numpy as np

from projectpredict.exceptions import InvalidProject

ExpectedCompletionTimeSamples = namedtuple('ExpectedCompletionTimeSamples', ['samples', 'is_final_task'])
Statistic = namedtuple('Statistic', ['mean', 'var'])


def datetime_stats(datetimes):
    """Gets the mean and variance of a collection of datetimes

    Args:
        datetimes (iterable(datetime)): The datetimes to compute the statistics on.

    Returns:
        dict: A dictionary containing keys for the mean and variance. The mean is a datetime, and the variance is a
            timedelta.
    """
    seconds_timestamp = [(datetime_ - datetime.utcfromtimestamp(0)).total_seconds() for datetime_ in datetimes]
    return {'mean': datetime.utcfromtimestamp(np.mean(seconds_timestamp)),
            'variance': timedelta(seconds=np.var(seconds_timestamp))}


def timedelta_stats(timedeltas):
    """Gets the mean and variance of a collection of timedeltas

    Args:
        timedeltas (iterable(timedelta)): The timedeltas to compute the statistics on.

    Returns:
        dict: A dictionary containing keys for the mean and variance. both the mean and variance are datetimes.
    """
    seconds_timestamp = [timedelta_.total_seconds() for timedelta_ in timedeltas]
    return {'mean': timedelta(seconds=np.mean(seconds_timestamp)),
            'variance': timedelta(seconds=np.var(seconds_timestamp))}


class TaskSample(object):
    """A wrapper for a sample of the derived statistics for a Task

    Attributes:
        duration (timedelta): The sampled duration of the task
        earliest_start (datetime): The sampled earliest start date of the task
        latest_finish(datetime): The sampled latest finish date of the task
        latest_start (datetime): The latest start date of the task. Must be set independently of the constructor
        earliest_finish (datetime): The earliest finish date of the task. Must be set independently of the constructor.

    Args:
        duration (timedelta): The sampled duration of the task
        earliest_start (datetime): The sampled earliest start date of the task
        latest_finish(datetime): The sampled latest finish date of the task
    """
    __slots__ = ['duration', 'earliest_start', 'latest_finish', 'latest_start', 'earliest_finish', '_total_float']

    def __init__(self, duration, earliest_start, latest_finish):
        self.duration = duration
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        self.latest_start = None
        self.earliest_finish = None
        self._total_float = None

    @property
    def total_float(self):
        """timedelta: The total float of the task. Earliest finish mst be set before calculation.
        """
        if self._total_float:
            return self._total_float
        elif self.latest_finish and self.earliest_finish:
            self._total_float = self.latest_finish - self.earliest_finish
        return self._total_float

    @classmethod
    def from_task(cls, task, current_time):
        """Constructs a TaskSample from a task

        Args:
            task (Task): The task to sample
            current_time (datetime): The current datetime used to sample the task

        Returns:
            TaskSample: The constructed sample
        """
        return cls(
            task.get_duration_sample(current_time),
            task.get_earliest_start_sample(current_time),
            task.get_latest_finish_sample())


class TaskStatistics(object):
    """A container for the relevant derived statistics for a Task

    Attributes:
        latest_start (dict): A dict containing the mean and variance of the latest start date of the task in 'mean' and
            'variance' keys respectively.
        earliest_finish (dict): A dict containing the mean and variance of the earliest finish date of the task in
            'mean' and 'variance' keys respectively.
        total_float (dict): A dict containing the mean and variance of the total float date of the task in 'mean' and
            'variance' keys respectively.

    Args:
        latest_start (dict): A dict containing the mean and variance of the latest start date of the task in 'mean' and
            'variance' keys respectively.
        earliest_finish (dict): A dict containing the mean and variance of the earliest finish date of the task in
            'mean' and 'variance' keys respectively.
        total_float (dict): A dict containing the mean and variance of the total float date of the task in 'mean' and
            'variance' keys respectively.

    """
    __slots__ = ['latest_start', 'earliest_finish', 'total_float']

    def __init__(self, latest_start, earliest_finish, total_float):
        self.latest_start = latest_start
        self.earliest_finish = earliest_finish
        self.total_float = total_float

    @classmethod
    def from_samples(cls, samples):
        """Construct a TaskStatistics object from samples

        Args:
            samples (iterable(TaskSample)): The samples to compute the statistics from.

        Returns:
            TaskStatistics: The constructed TaskStatistics
        """
        return cls(
            datetime_stats([sample.latest_start for sample in samples]),
            datetime_stats([sample.earliest_finish for sample in samples]),
            timedelta_stats(sample.total_float for sample in samples)
        )

    def __repr__(self):
        return str({
            'latest_start': self.latest_start,
            'earliest-finish': self.earliest_finish,
            'total_float': self.total_float
        })


DependencySummary = namedtuple('DependencySummary', ['source', 'destination'])


class Project(nx.DiGraph):
    """A project

    Note:
        This must be an acyclic graph.

    Attributes:
        name (str): The name of the project
        uid (UUID): The UUID of the project
        model: A model used to predict the duration of tasks from their data

    Args:
        name (str): The name of the project
        model (optional): A model used to predict the duration of tasks from their data
        uid (UUID, optional): The UUID of the project
        tasks (iterable(Task), optional): A collections of Tasks associated with this project
        dependencies (iterable(dict), optional): The dependencies associated with the project in the form of dicts of
            'source' and 'destination' keys.
    """

    def __init__(self, name, model=None, uid=None, tasks=None, dependencies=None):
        super(Project, self).__init__()
        self.uid = uid or uuid4()
        self.name = name
        self.model = model
        for task in (tasks or []):
            self.add_task(task)
        for dependency in (dependencies or []):
            source = self.get_task_from_id(dependency['source'])
            target = self.get_task_from_id(dependency['destination'])
            self.add_edge(source, target)

    def validate(self):
        """Validates the Project meets the requirements to do inference

        Checks:
        * The Project is a directed acyclic graph
        * Every terminal Task (one without successors) has a latest start date PDF

        Raises:
            InvalidProject: If the project does not conform to the requirements.
        """
        errors = []
        if not nx.is_directed_acyclic_graph(self):
            errors.append('Project is not acyclic')
        for task in self.tasks:
            if not list(self.successors(task)) and task.latest_finish_date_pdf is None:
                errors.append('Task {} (id={}) requires a latest finish date pdf'.format(task.name, task.uid))
        if errors:
            raise InvalidProject(errors)

    @property
    def dependencies(self):
        """list[tuple(Task, Task)]: The dependencies in the project where the first element of the tuple is the source
        task and the second element of the tuple is the dependent task.
        """
        return self.edges

    @property
    def tasks(self):
        """iterable(Task): The tasks of this project
        """
        return self.nodes

    @property
    def dependencies_summary(self):
        """list[DependencySummary]: The dependencies of this project
        """
        return [DependencySummary(source.uid, target.uid) for source, target in self.edges]

    def get_task_from_id(self, id_):
        """Gets a task from an id

        Args:
            id_ (UUID): The UUID of the project to get

        Returns:
            Task|None: The task with the associated with the id or None if task is not found
        """
        return next((task for task in self.tasks if task.uid == id_), None)

    def add_task(self, task):
        """Adds a Task to this Project and determines the duration PDF of the task from the model if not previously specified.

        Args:
            task (Task): The Task to add to the project
        """
        task.project_uid = self.uid
        if not task.duration_pdf:
            if not self.model:
                msg = ('Could not add task. Either a duration pdf must be specified or a model must be specified for '
                       'the project')
                raise InvalidProject([msg])
            task.set_duration_pdf(self.model)
        self.add_node(task)

    def add_tasks(self, tasks):
        """Adds multiple Tasks to this Project and determines the duration PDF of the task from the model if not
            previously specified.

        Args:
            tasks (iterable(Task)): The Task to add to the project
        """
        for task in tasks:
            self.add_task(task)

    def add_dependency(self, parent, child):
        """Adds a Task dependency to this Project

        Args:
            parent (Task): The parent task
            child (Task): The child task, i.e. the Task which depends on the parent
        """
        if parent.uid == child.uid:
            raise InvalidProject('Parent and child Tasks must be different')
        if child not in self.tasks:
            self.add_task(child)
        if parent not in self.tasks:
            self.add_task(parent)
        self.add_edge(parent, child)

    def add_dependencies(self, dependencies):
        """Adds multiple Task dependencies to this Project

        Args:
            dependencies (list[tuple(Task, Task)]): A list of tuples of Task dependencies in the form of
                (parent task, child task)
        """
        for dependency in dependencies:
            self.add_dependency(*dependency)

    def calculate_earliest_finish_times(self, current_time=None, iterations=1000):
        """Generates samples of the earliest finish times for each uncompleted node in the project.

        Args:
            current_time (datetime): the time at which to take the samples
            iterations (int, optional): The number of samples to generate. Defaults to 1000

        Returns:
            dict{Task: [datetime]}: A dictionary of the samples for each task.
        """
        self.validate()
        samples = self._get_samples(
            forward_sample_func=self.earliest_finish_sample_func,
            iterations=iterations,
            current_time=current_time)
        return {task: [task_sample.earliest_finish for task_sample in task_samples]
                for task, task_samples in samples.items()}

    @staticmethod
    def earliest_finish_sample_func(task, parents, children, samples, **kwargs):
        if not parents:
            earliest_finish = samples[task].earliest_start + samples[task].duration
        else:
            earliest_start = max(samples[parent_task].earliest_finish for parent_task in parents)
            samples[task].earliest_start = earliest_start
            earliest_finish = earliest_start + samples[task].duration

        samples[task].earliest_finish = earliest_finish

    def calculate_latest_start_times(self, iterations=1000):
        """Generates samples of the latest start times for each uncompleted node in the project.

        Args:
            iterations (int, optional): The number of samples to generate. Defaults to 1000

        Returns:
            dict{Task: [datetime]}: A dictionary of the samples for each task.
        """
        self.validate()
        samples = self._get_samples(
            backward_sample_func=self.latest_start_sample_func,
            iterations=iterations)
        return {task: [task_sample.latest_start for task_sample in task_samples]
                for task, task_samples in samples.items()}

    @staticmethod
    def latest_start_sample_func(task, parents, children, samples, **kwargs):
        if not children:
            latest_start = samples[task].latest_finish - samples[task].duration
        else:
            latest_finish = min(samples[child_task].latest_start for child_task in children)
            samples[task].latest_finish = latest_finish
            latest_start = latest_finish - samples[task].duration
        samples[task].latest_start = latest_start

    def _get_samples(
            self,
            forward_sample_func=None,
            backward_sample_func=None,
            iterations=1000,
            current_time=None,
            **kwargs):
        current_time = current_time or datetime.utcnow()
        sample_collection = {node: [] for node in self.nodes}
        for _ in range(iterations):
            samples = {task: TaskSample.from_task(task, current_time) for task in self.nodes}
            if forward_sample_func:
                for task in nx.topological_sort(self):
                    parents = list(parent for parent in self.predecessors(task) if not parent.is_completed)
                    children = list(child for child in self.successors(task) if not child.is_completed)
                    forward_sample_func(task, parents, children, samples, **kwargs)
            if backward_sample_func:
                for task in reversed(list(nx.topological_sort(self))):
                    parents = list(parent for parent in self.predecessors(task) if not parent.is_completed)
                    children = list(child for child in self.successors(task) if not child.is_completed)
                    backward_sample_func(task, parents, children, samples, **kwargs)
            for task, sample in samples.items():
                sample_collection[task].append(sample)
        return sample_collection

    def calculate_task_statistics(self, current_time=None, iterations=1000):
        self.validate()
        samples = self._get_samples(
            forward_sample_func=self.earliest_finish_sample_func,
            backward_sample_func=self.latest_start_sample_func,
            iterations=iterations,
            current_time=current_time)
        return {task: TaskStatistics.from_samples(task_samples) for task, task_samples in samples.items()}

    def recommend_next(
            self,
            current_time=None,
            constraints=None,
            iterations=1000,
            score_func=None,
            selection_func=None,
            min_number=1,
            max_number=1,
            batch_wait=False,
            selection_func_arguments=None,
            **score_func_arguments):
        """Get the recommended next tasks

        Args:
            current_time (datetime, optional): The current time (in UTC) to query the project.
                Defaults to the current time.
            constraints (iterable(callable)): A list of constraints to apply to the selected tasks. These must be
                functions which task in two parameters -- the project (self) and the set of Tasks under consideration.
            iterations (int, optional): The number of iterations to query the project for each considered set of Tasks.
                Defaults to 1000.
            score_func (func, optional): The function used to score the results of a Task set. Defaults to a function
                which returns a dict containing the mean and precision (inverse variance) of the total float of each
                task weighted by the Tasks' deadline weight. The function must take keyword arguments which can be
                specified as keyword arguments to this function (see score_func_arguments).
            selection_func (func, optional): The function used to select which task set is best from the results
                returned from the score_func. Defaults to a function which scales the total float and precision each
                between 0 and 1 and sums them according to a weighting parameter (see selection_func_arguments). The
                function must accept a dict of Task set to score and keyword arguments which can be specified by the
                selection_func_arguments parameter of this function.
            min_number (int, optional): The minimum number of tasks which can can be recommended. Defaults to 1.
            max_number (int, optional): The maximum number of tasks which can be recommended. Defaults to 1.
            batch_wait (bool, optional): Do all tasks for a proposed tuple of Tasks need to be completed before the next
                tasks can begin? Defaults to False.
            selection_func_arguments (dict, optional): The arguments to be passed to the selection_func.
            **score_func_arguments: The arguments to pass to the score_func

        Returns:
            tuple(Task): The recommended tasks to complete next
        """
        self.validate()
        score_func = score_func or self._default_recommendation_score_func
        selection_func = selection_func or self._default_recommendation_selection_func
        selection_func_arguments = selection_func_arguments or {}
        current_time = current_time or datetime.utcnow()
        if not constraints:
            constraints = []

        def recommendation_sample_func(task, parents, children, samples, **kwargs):
            tasks_under_sample = kwargs['tasks_under_sample']
            if task in kwargs['tasks_under_sample']:
                earliest_finish = samples[task].earliest_start + samples[task].duration
            else:
                if not parents:
                    if kwargs['batch_wait']:
                        additional_wait = min(samples[task_].duration for task_ in tasks_under_sample)
                    else:
                        additional_wait = max(samples[task_].duration for task_ in tasks_under_sample)
                    earliest_finish = samples[task].earliest_start + samples[task].duration + additional_wait
                else:
                    earliest_start = (max(samples[parent_task].earliest_finish for parent_task in parents))
                    samples[task].earliest_start = earliest_start
                    earliest_finish = earliest_start + samples[task].duration
            samples[task].earliest_finish = earliest_finish

        possible_tasks = [node for node in self.nodes
                          if not node.is_completed
                          and not [parent_node for parent_node in self.predecessors(node)
                                   if not parent_node.is_completed]]
        if not max_number:
            max_number = len(possible_tasks)
        scores = {}
        for n in reversed(range(min_number, max_number + 1)):
            for task_set in (task_set for task_set in combinations(possible_tasks, n)
                             if all(constraint(self, task_set) for constraint in constraints)):
                mean_durations = {task: task.mean_duration for task in task_set}
                samples = self._get_samples(
                    forward_sample_func=recommendation_sample_func,
                    backward_sample_func=self.latest_start_sample_func,
                    current_time=current_time,
                    iterations=iterations,
                    mean_duration_lookup=mean_durations,
                    tasks_under_sample=task_set,
                    batch_wait=batch_wait)
                scores[tuple(task_set)] = score_func(samples, **score_func_arguments)
        if not scores:
            raise ValueError('No tasks found which meet constraints')
        return selection_func(scores, **selection_func_arguments)

    @staticmethod
    def _default_recommendation_score_func(samples, **kwargs):
        task_float_stats = {task: timedelta_stats([sample.total_float for sample in task_samples])
                            for task, task_samples in samples.items()}
        float_score = sum(
            score['mean'].total_seconds() * task.deadline_weight for task, score in task_float_stats.items())
        precision = sum(1 / max(score['variance'].total_seconds(), 1e-6) * task.deadline_weight
                        for task, score in task_float_stats.items())
        return {'total_float_score': float_score, 'precision': precision}

    @staticmethod
    def _default_recommendation_selection_func(scores, **kwargs):

        def scale_scores(field):
            min_score = min(score[field] for score in scores.values())
            max_score = max(score[field] for score in scores.values())
            score_diff = max_score - min_score
            return {task_set: (score[field] - min_score) / max(score_diff, 1e-6) for task_set, score in scores.items()}

        float_scores = scale_scores('total_float_score')
        precision_scores = scale_scores('precision')
        risk_tolerance = kwargs.get('risk_tolerance', 0.5)
        combined_scores = {
            task_set: risk_tolerance * float_scores[task_set] + (1 - risk_tolerance) * precision_scores[task_set]
            for task_set in scores}
        return max(combined_scores, key=combined_scores.get)

    def get_starting_and_terminal_tasks(self):
        """Gets the starting tasks (ones without predecessors) and terminal tasks (ones without successors)

        Returns:
            tuple(list[Task], list[Task]): The starting and terminal tasks in the form of
                (starting tasks, terminal tasks)
        """
        start_tasks = []
        terminal_tasks = []
        for task in self.tasks:
            if len(list(self.predecessors(task))) == 0:
                start_tasks.append(task)
            if len(list(self.successors(task))) == 0:
                terminal_tasks.append(task)
        return start_tasks, terminal_tasks

    def update_from_dict(self, data):
        """Updates the Project using a dictionary of new values

        Args:
            data (dict): The new values
        """
        if 'name' in data:
            self.name = data['name']
        new_tasks = data.get('tasks', [])
        new_dependencies = [(dependency['source'], dependency['destination']) for dependency in
                            data.get('dependencies', [])]
        for task in new_tasks:
            if task not in self.tasks:
                self.add_task(task)
        removed_tasks = [task for task in self.tasks if task not in new_tasks]
        for task in removed_tasks:
            self.remove_node(task)
        edges = [(dependency[0].uid, dependency[1].uid) for dependency in self.dependencies]
        for dependency in new_dependencies:
            if dependency not in edges:
                dependency = (self.get_task_from_id(dependency[0]), self.get_task_from_id(dependency[1]))
                self.add_edge(*dependency)

        removed_dependencies = [dependency for dependency in edges if dependency not in new_dependencies]
        for dependency in removed_dependencies:
            try:
                self.remove_edge(self.get_task_from_id(dependency[0]), self.get_task_from_id(dependency[1]))
            except nx.NetworkXError:
                pass

    @classmethod
    def from_dict(cls, data_in, model):
        """Constructs a Project from a dictionary of values and a model

        Args:
            data_in (dict): The data to construct the Project from
            model: The model used to predict the durations of tasks

        Returns:
            Project: The constructed project
        """
        return cls(
            data_in['name'],
            model,
            uid=data_in.get('uid', None),
            tasks=data_in.get('tasks', []),
            dependencies=data_in.get('dependencies', [])
        )
