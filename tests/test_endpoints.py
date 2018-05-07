import pytest
from datetime import datetime

from projectpredict.app import *
from projectpredict.context import context_factory
from projectpredict.pdf import *
from projectpredict.services.marshmallow import *


class TestConfig:
    serialization = 'marshmallow'
    repository = 'memory'


class MockModel:

    def predict(self, data):
        return DurationPdf(DeterministicPdf(sum(data['a'])))


def task_factory(name, value):
    return Task(name, duration_pdf=DurationPdf(DeterministicPdf(value), units=TimeUnits.days))


@pytest.fixture()
def app(request):
    """Session-wide test `Flask` application."""
    app_ = create_app(TestConfig, MockModel())
    context = app_.running_context
    app_ = app_.test_client()
    app_.running_context = context
    return app_


def test_create_app():
    context = context_factory(TestConfig)
    model = MockModel()
    app = create_app(TestConfig, model)
    assert app.name == 'projectpredict'
    assert isinstance(app.running_context.serialization_service, type(context.serialization_service))
    assert isinstance(app.running_context.project_repository, type(context.project_repository))
    assert isinstance(app.running_context.task_repository, type(context.task_repository))
    assert app.running_context.learning_model is model


def test_get_all_projects(app):
    response = app.get('/projects')
    assert response.status_code == ReturnCodes.success
    assert response.get_json() == {'projects': []}

    project1 = Project('a', MockModel())
    project2 = Project('b', MockModel())
    task1 = Task('t1', data={'a': [1, 2, 3]})
    task2 = Task('t2', data={'a': [2, 3, 4]})
    task3 = Task('t3', data={'a': [3, 4, 5]})
    project2.add_edges_from([(task1, task3), (task2, task3)])
    app.running_context.project_repository.create(project1)
    app.running_context.project_repository.create(project2)
    expected = {'projects': [app.running_context.serialization_service.dump(project)
                             for project in app.running_context.project_repository.get_all().values()]}
    response = app.get('/projects')
    assert response.status_code == ReturnCodes.success
    assert response.get_json() == expected


def test_create_project(app):
    project = Project('b', MockModel())
    current_date = datetime(year=2016, month=10, day=23)
    deadline_date = datetime(year=2016, month=11, day=17)
    task1 = Task(
        't1',
        data={'a': [1, 2, 3]},
        project_uid=project.uid,
        earliest_start_date_pdf=DatePdf(current_date, DeterministicPdf(2)))
    task2 = Task('t2', data={'a': [2, 3, 4]}, project_uid=project.uid)
    task3 = Task(
        't3',
        data={'a': [3, 4, 5]},
        project_uid=project.uid,
        latest_finish_date_pdf=DatePdf(deadline_date, DeterministicPdf(3)))
    project.add_edges_from([(task1, task2), (task2, task3)])
    project_data = app.running_context.serialization_service.dump(project)
    response = app.post('/projects', json=project_data)
    assert response.status_code == ReturnCodes.created
    project_returned = app.running_context.serialization_service.load(Project, response.get_json())
    assert project_returned['name'] == 'b'
    assert set(project_returned['tasks']) == {task1, task2, task3}
    assert project_returned['uid'] == project.uid
    returned_edges = [(edge['source'], edge['destination']) for edge in project_returned['dependencies']]
    previous_edges = [(source.uid, target.uid) for source, target in project.edges]
    for edge in returned_edges:
        assert edge in previous_edges


def test_create_project_invalid_data(app):
    project = Project('b', MockModel())
    current_date = datetime(year=2016, month=10, day=23)
    task1 = Task(
        't1',
        data={'a': [1, 2, 3]},
        project_uid=project.uid,
        earliest_start_date_pdf=DatePdf(current_date, DeterministicPdf(2)))
    task2 = Task('t2', data={'a': [2, 3, 4]}, project_uid=project.uid)
    project.add_edges_from([(task1, task2)])
    project_data = app.running_context.serialization_service.dump(project)
    project_data['uid'] = 'invalid'
    response = app.post('/projects', json=project_data)
    assert response.status_code == ReturnCodes.bad_request
    assert 'error' in response.get_json()


def test_update_project(app):
    project = Project('b', MockModel())
    current_date = datetime(year=2016, month=10, day=23)
    deadline_date = datetime(year=2016, month=11, day=17)
    task1 = Task(
        't1',
        data={'a': [1, 2, 3]},
        project_uid=project.uid,
        earliest_start_date_pdf=DatePdf(current_date, DeterministicPdf(2)))
    task2 = Task('t2', data={'a': [2, 3, 4]}, project_uid=project.uid)
    task3 = Task(
        't3',
        data={'a': [3, 4, 5]},
        project_uid=project.uid,
        latest_finish_date_pdf=DatePdf(deadline_date, DeterministicPdf(3)))
    project.add_edges_from([(task1, task2), (task2, task3)])
    project_data = app.running_context.serialization_service.dump(project)
    print(project_data)
    app.post('/projects', json=project_data)

    project_data['name'] = 'new_name'
    modified_task_uid = project_data['tasks'][0]
    project_data['tasks'][0]['name'] = 'new_task'
    project_data['dependencies'] = [{'source': str(task1.uid), 'destination': str(task2.uid)}]
    response = app.put('/projects', json=project_data)
    assert response.status_code == ReturnCodes.success
    resp_data = response.get_json()
    print(response.get_json())
    assert resp_data['name'] == 'new_name'
    modified_task = next(task for task in resp_data['tasks'] if task['uid' == modified_task_uid])
    assert modified_task['name'] == 'new_task'