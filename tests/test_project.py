from projectpredict.pdf import DeterministicPdf
from projectpredict.project import *
from projectpredict.task import *
import pytest
from datetime import datetime, timedelta
import numpy as np


def make_deterministic_date_pdf(date, value):
    return DatePdf(date, DeterministicPdf(value), units=TimeUnits.days)


def task_factory(name, value):
    return Task(name, duration_pdf=DurationPdf(DeterministicPdf(value), units=TimeUnits.days))


@pytest.fixture
def project():
    '''
    1 ->           5
        |-> 3 -> |
    2 ->           6
         | 4
    '''
    task1 = task_factory('1', 1)
    task2 = task_factory('2', 2)
    task3 = task_factory('3', 3)
    task4 = task_factory('4', 2)
    task5 = task_factory('5', 1)
    task6 = task_factory('6', 2)
    current_time = datetime(year=2018, month=5, day=14)
    task4.latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 5)
    task5.latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 9)
    task6.latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 7)

    class MockModel:
        def predict(self, data):
            return DurationPdf(DeterministicPdf(data['a']))

    proj = Project('proj', MockModel())
    proj.add_dependencies([
        (task1, task3),
        (task2, task3),
        (task2, task4),
        (task3, task5),
        (task3, task6)])
    return proj, {task.name: task for task in [task1, task2, task3, task4, task5, task6]}


@pytest.fixture
def simple_project():
    '''
    1 ->
        |-> 3
    2 ->
    '''
    task1 = task_factory('1', 1)
    task2 = task_factory('2', 2)
    task3 = task_factory('3', 3)
    current_time = datetime(year=2018, month=5, day=14)
    task3.latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 5)

    class MockModel:
        def predict(self, data):
            return DurationPdf(DeterministicPdf(data['a']))

    proj = Project('proj', MockModel())
    proj.add_dependencies([
        (task1, task3),
        (task2, task3)])
    return proj, {task.name: task for task in [task1, task2, task3]}


def test_datetime_stats():
    current_date = datetime(year=2019, month=4, day=13)
    days = [1,2,3]
    datetimes = [current_date + timedelta(days=day) for day in days]
    expected_variance = np.var([(date - datetime.utcfromtimestamp(0)).total_seconds() for date in datetimes])
    stats = datetime_stats(datetimes)
    assert stats == {'mean': current_date + timedelta(days=np.mean(days)),'variance': timedelta(seconds=expected_variance)}


def test_timedelta_variance():
    timedeltas = [timedelta(days=day) for day in range(1, 4)]
    seconds_timestamp = [timedelta_.total_seconds() for timedelta_ in timedeltas]
    expected_mean = timedelta(seconds=np.mean(seconds_timestamp))
    expected_variance = timedelta(seconds=np.var(seconds_timestamp))
    assert timedelta_stats(timedeltas) == {'mean': expected_mean, 'variance': expected_variance}


def test_task_sample_total_float():
    current_date = datetime(year=2019, month=4, day=13)
    sample = TaskSample(timedelta(days=1), current_date + timedelta(days=2), current_date + timedelta(days=3))
    assert sample.total_float is None
    sample.earliest_finish = current_date + timedelta(days=2)
    assert sample.total_float == timedelta(days=1)
    assert sample.total_float == timedelta(days=1)


def test_task_sample_from_tasj():
    current_date = datetime(year=2019, month=4, day=13)
    task = task_factory('1', 1)
    task.earliest_start_date_pdf = make_deterministic_date_pdf(current_date, 5)
    task.latest_finish_date_pdf = make_deterministic_date_pdf(current_date, 7)
    sample = TaskSample.from_task(task, current_date)
    assert sample.duration == timedelta(days=1)
    assert sample.earliest_start == current_date + timedelta(days=5)
    assert sample.latest_finish == current_date + timedelta(days=7)
    assert sample.latest_start is None
    assert sample.earliest_finish is None


def test_task_statistics_from_samples():
    current_date = datetime(year=2019, month=4, day=13)
    days = [1, 2, 3]
    datetimes = [current_date + timedelta(days=day) for day in days]
    timedeltas = [timedelta(days=day) for day in range(1, 4)]
    expected_latest_start = datetime_stats(datetimes)
    expected_earliest_finish = datetime_stats(datetimes)
    expected_total_float = timedelta_stats(timedeltas)
    samples = [TaskSample(1, 1, 1) for _ in range(len(datetimes))]  # dummy objects
    for sample, datetime_, timedelta_ in zip(samples, datetimes, timedeltas):
        sample.latest_start = datetime_
        sample.earliest_finish = datetime_
        sample._total_float = timedelta_
    task_stat = TaskStatistics.from_samples(samples)
    assert task_stat.latest_start == expected_latest_start
    assert task_stat.earliest_finish == expected_earliest_finish
    assert task_stat.total_float == expected_total_float


def test_invalid_project():
    err = ['something occurred']
    exc = InvalidProject(err)
    assert isinstance(exc, Exception)
    assert exc.errors == err


def test_init():
    class MockModel: pass
    model = MockModel()
    project = Project('important_proc', model=model)
    assert project.name == 'important_proc'
    assert project.model == model


def test_add_task():
    project = Project('proj')
    task = task_factory('task1', 3)
    project.add_task(task)
    assert task in project.tasks
    assert task.project_uid == project.uid


def test_add_task_no_duration_no_model():
    project = Project('proj')
    task = Task('task1')
    with pytest.raises(InvalidProject):
        project.add_task(task)


def test_add_task_no_duration_with_model(project):
    project = project[0]
    task = Task('task1', data={'a': 3})
    project.add_task(task)
    assert task.duration_pdf == DurationPdf(DeterministicPdf(3))


def test_add_tasks_some_no_duration_no_model():
    project = Project('proj')
    task1 = Task('task1')
    task2 = task_factory('task2', 4)
    with pytest.raises(InvalidProject):
        project.add_tasks([task2, task1])
    assert task1 not in project.tasks
    assert task2 in project.tasks


def test_add_tasks_some_mixed_no_duration(project):
    project = project[0]
    task1 = Task('task1', data={'a': 3})
    task2 = task_factory('task2', 4)
    project.add_tasks([task1, task2])
    assert task1.duration_pdf == DurationPdf(DeterministicPdf(3))
    assert task1 in project.tasks
    assert task2 in project.tasks


def test_add_dependency_to_self():
    project = Project('proj')
    task1 = Task('task1')
    with pytest.raises(InvalidProject):
        project.add_dependency(task1, task1)


def test_add_dependency_already_added(project):
    project = project[0]
    task1 = Task('task1', data={'a': 3})
    task2 = task_factory('task2', 4)
    project.add_tasks([task1, task2])
    project.add_dependencies([(task1, task2)])
    assert list(project.predecessors(task2)) == [task1]


def test_add_dependency_not_already_added(project):
    project = project[0]
    task1 = Task('task1', data={'a': 3})
    task2 = task_factory('task3', 4)
    project.add_dependencies([(task1, task2)])
    assert list(project.predecessors(task2)) == [task1]


def test_add_dependencies_some_invalid():
    project = Project('proj')
    task1 = task_factory('task1', 4)
    task2 = task_factory('task2', 4)
    task3 = task_factory('task3', 4)
    with pytest.raises(InvalidProject):
        project.add_dependencies([
            (task1, task2),
            (task3, task3)
        ])
    assert list(project.predecessors(task2)) == [task1]
    assert task3 not in project.tasks


def test_add_dependencies_some_added():
    project = Project('proj')
    task1 = task_factory('task1', 1)
    task2 = task_factory('task2', 2)
    task3 = task_factory('task3', 3)
    task4 = task_factory('task4', 4)
    project.add_tasks([task1, task2])
    project.add_dependencies([
        (task1, task2),
        (task3, task4)
    ])
    for task in [task1, task2, task3, task4]:
        assert task in project.tasks
    assert list(project.predecessors(task2)) == [task1]
    assert list(project.predecessors(task4)) == [task3]


def test_validate_cyclic():
    project = Project('proj')
    task1 = task_factory('task1', 1)
    task2 = task_factory('task2', 2)
    project.add_dependencies([
        (task1, task2),
        (task2, task1)
    ])
    with pytest.raises(InvalidProject) as e:
        project.validate()
        assert len(e.errors) == 2


def test_validate_terminal_task_no_pdf():
    project = Project('proj')
    task1 = task_factory('task1', 1)
    task2 = task_factory('task2', 2)
    project.add_dependency(task1, task2)
    with pytest.raises(InvalidProject) as e:
        project.validate()
        assert len(e.errors) == 1


def test_validate(project):
    project = project[0]
    project.validate()


def test_calculate_earliest_finish_times(project):
    tasks = project[1]
    project = project[0]
    current_time = datetime(year=2017, month=11, day=1)
    iterations = 1
    times = project.calculate_earliest_finish_times(current_time=current_time, iterations=iterations)
    expected_times = {
        tasks['1']: 1,
        tasks['2']: 2,
        tasks['3']: 5,
        tasks['4']: 4,
        tasks['5']: 6,
        tasks['6']: 7
    }
    expected_times = {task: [current_time + timedelta(days=days)]*iterations
                      for task, days in expected_times.items()}
    assert times == expected_times


def test_calculate_earliest_finish_times_invalid_project(project):
    tasks = project[1]
    project = project[0]
    tasks['4'].latest_finish_date_pdf = None
    current_time = datetime(year=2017, month=11, day=1)
    iterations = 1
    with pytest.raises(InvalidProject):
        project.calculate_earliest_finish_times(current_time=current_time, iterations=iterations)


def test_calculate_latest_start_times(project):
    tasks = project[1]
    project = project[0]
    current_time = datetime(year=2017, month=11, day=3)
    tasks['4'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 5)
    tasks['5'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 9)
    tasks['6'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 7)
    iterations = 10
    times = project.calculate_latest_start_times(iterations=iterations)
    expected_times = {
        tasks['1']: 1,
        tasks['2']: 0,
        tasks['3']: 2,
        tasks['4']: 3,
        tasks['5']: 8,
        tasks['6']: 5
    }
    expected_times = {task: [current_time + timedelta(days=days)] * iterations
                      for task, days in expected_times.items()}
    assert times == expected_times


def test_calculate_latest_start_times_invalid_project(project):
    tasks = project[1]
    project = project[0]
    tasks['4'].latest_finish_date_pdf = None
    iterations = 10
    with pytest.raises(InvalidProject):
        project.calculate_latest_start_times(iterations=iterations)


def test_calculate_task_statistics(project):
    tasks = project[1]
    project = project[0]
    current_time = datetime(year=2017, month=11, day=3)
    tasks['4'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 5)
    tasks['5'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 9)
    tasks['6'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 7)
    iterations = 10
    stats = project.calculate_task_statistics(current_time=current_time, iterations=iterations)
    expected_earliest_finish = {
        tasks['1']: 1,
        tasks['2']: 2,
        tasks['3']: 5,
        tasks['4']: 4,
        tasks['5']: 6,
        tasks['6']: 7
    }
    expected_latest_start = {
        tasks['1']: 1,
        tasks['2']: 0,
        tasks['3']: 2,
        tasks['4']: 3,
        tasks['5']: 8,
        tasks['6']: 5
    }

    expected_total_float = {
        tasks['1']: 1,
        tasks['2']: 0,
        tasks['3']: 0,
        tasks['4']: 1,
        tasks['5']: 3,
        tasks['6']: 0
    }

    expected_earliest_finish = {task: {'mean': current_time + timedelta(days=days), 'variance': timedelta(days=0)}
                                for task, days in expected_earliest_finish.items()}
    expected_latest_start = {task: {'mean': current_time + timedelta(days=days), 'variance': timedelta(days=0)}
                                for task, days in expected_latest_start.items()}
    expected_total_float = {task: {'mean': timedelta(days=days), 'variance': timedelta(days=0)}
                             for task, days in expected_total_float.items()}
    assert {task: stat.earliest_finish for task, stat in stats.items()} == expected_earliest_finish
    assert {task: stat.latest_start for task, stat in stats.items()} == expected_latest_start
    assert {task: stat.total_float for task, stat in stats.items()} == expected_total_float


def test_calculate_task_statistics_invalid_project(project):
    tasks = project[1]
    project = project[0]
    current_time = datetime(year=2017, month=11, day=3)
    tasks['4'].latest_finish_date_pdf = None
    with pytest.raises(InvalidProject):
        project.calculate_task_statistics(current_time=current_time, iterations=10)


def test_project_recommendation(project):
    tasks = project[1]
    project = project[0]
    current_time = datetime(year=2017, month=11, day=3)
    tasks['4'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 5)
    tasks['5'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 9)
    tasks['6'].latest_finish_date_pdf = make_deterministic_date_pdf(current_time, 7)

    assert project.recommend_next(iterations=5)[0] in (tasks['1'], tasks['2'])

    def constraint(project, tasks_):
        return tasks['1'] not in tasks_

    assert project.recommend_next(iterations=3, constraints=[constraint]) == (tasks['2'],)

    def constraint(project, tasks_):
        return False

    with pytest.raises(ValueError):
        project.recommend_next(iterations=1, constraints=[constraint])

    assert set(project.recommend_next(
        iterations=1,
        current_time=current_time,
        max_number=2,
        selection_func_arguments={'risk_tolerance': 1})) == {tasks['1'], tasks['2']}


def test_project_recommendation_invalid_graph(project):
    tasks = project[1]
    project = project[0]
    tasks['4'].latest_finish_date_pdf = None
    with pytest.raises(InvalidProject):
        project.recommend_next(iterations=5)


def test_project_from_dict_no_tasks():
    dict_in = {'name': 'proj1'}

    class MockModel: pass

    model = MockModel()
    proj = Project.from_dict(dict_in, model)
    assert proj.uid is not None
    assert proj.name == 'proj1'
    assert proj.model is model


def test_project_from_dict_with_tasks_with_durations():
    task1 = task_factory('1', 1)
    task2 = task_factory('2', 2)
    task3 = task_factory('3', 3)
    task4 = task_factory('4', 2)
    task5 = task_factory('5', 1)
    task6 = task_factory('6', 2)
    tasks = [task1, task2, task3, task4, task5, task6]
    data_in = {
        'name': 'proj',
        'tasks': tasks,
        'dependencies': [
            {'source': source.uid, 'destination': destination.uid}
            for source, destination in [
                (task1, task3),
                (task2, task3),
                (task2, task4),
                (task3, task5),
                (task3, task6)]]
    }

    class MockModel: pass
    model = MockModel()
    proj = Project.from_dict(data_in, model)
    assert proj.uid is not None
    assert proj.name == 'proj'
    assert proj.model is model
    for task in tasks:
        assert task in proj.tasks
    assert set(proj.successors(task1)) == {task3}
    assert set(proj.successors(task2)) == {task4, task3}
    assert set(proj.successors(task3)) == {task5, task6}
    for task in {task5, task6}:
        assert set(proj.successors(task)) == set()
    for task in {task1, task2}:
        assert set(proj.predecessors(task)) == set()
    assert set(proj.predecessors(task3)) == {task1, task2}
    assert set(proj.predecessors(task4)) == {task2}
    assert set(proj.predecessors(task5)) == {task3}
    assert set(proj.predecessors(task6)) == {task3}


def test_project_from_dict_with_tasks_without_durations():
    task1 = Task('1', data={'a': 1})
    task2 = Task('2', data={'a': 2})
    task3 = Task('3', data={'a': 3})
    tasks = [task1, task2, task3]
    data_in = {
        'name': 'proj',
        'tasks': tasks,
        'dependencies': [
            {'source': source.uid, 'destination': destination.uid}
            for source, destination in [
                (task1, task3),
                (task2, task3)]]
    }

    class MockModel:
        def predict(self, data):
            return DeterministicPdf(data['a'])

    model = MockModel()
    proj = Project.from_dict(data_in, model)

    assert proj.uid is not None
    assert proj.name == 'proj'
    assert proj.model is model
    for task in tasks:
        assert task in proj.tasks
        assert task.project_uid == proj.uid
    for task in {task1, task2}:
        assert set(proj.successors(task)) == {task3}
    assert set(proj.successors(task3)) == set()
    for task in {task1, task2}:
        assert set(proj.predecessors(task)) == set()
    assert set(proj.predecessors(task3)) == {task1, task2}

    for task in tasks:
        assert task.duration_pdf.mean == task.data['a']


def test_project_update_from_dict_remove_task_add_task(simple_project):
    task_lookup = simple_project[1]
    task3 = task_lookup['3']
    project = simple_project[0]
    task4 = Task('4', data={'a': 4})
    tasks = [task for task in task_lookup.values() if task.name != '2']
    tasks.append(task4)
    edges =[edge for edge in project.edges if edge[0].name != '2']
    edges.append((task3, task4))
    dependencies_json = [{'source': edge[0].uid, 'destination': edge[1].uid} for edge in edges]
    update_data = {
        'uid': project.uid,
        'name': 'new_name',
        'tasks': tasks,
        'dependencies': dependencies_json
    }
    project.update_from_dict(update_data)
    assert project.name == 'new_name'
    assert task_lookup['2'] not in project.tasks
    for task in tasks:
        assert task in project.tasks
    assert set(project.successors(task_lookup['1'])) == {task_lookup['3']}
    assert set(project.predecessors(task_lookup['1'])) == set()
    assert set(project.successors(task_lookup['3'])) == {task4}
    assert set(project.predecessors(task_lookup['3'])) == {task_lookup['1']}
    assert set(project.successors(task4)) == set()
    assert set(project.predecessors(task4)) == {task_lookup['3']}


def test_project_update_from_dict_remove_edge(simple_project):
    task_lookup = simple_project[1]
    project = simple_project[0]
    edges =[edge for edge in project.edges if edge[0].name != '2']
    dependencies_json = [{'source': edge[0].uid, 'destination': edge[1].uid} for edge in edges]
    update_data = {
        'uid': project.uid,
        'name': 'new_name',
        'tasks': list(task_lookup.values()),
        'dependencies': dependencies_json
    }
    project.update_from_dict(update_data)
    assert project.name == 'new_name'
    for task in task_lookup.values():
        assert task in project.tasks
    assert set(project.successors(task_lookup['1'])) == {task_lookup['3']}
    assert set(project.predecessors(task_lookup['1'])) == set()
    assert set(project.successors(task_lookup['2'])) == set()
    assert set(project.predecessors(task_lookup['2'])) == set()
    assert set(project.successors(task_lookup['3'])) == set()
    assert set(project.predecessors(task_lookup['3'])) == {task_lookup['1']}


def test_project_dependencies_property(project):
    project = project[0]
    dependencies = project.dependencies_summary
    for source, target in project.edges:
        assert DependencySummary(source.uid, target.uid) in dependencies
