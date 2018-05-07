from projectpredict.project import Project
from projectpredict.task import Task
from projectpredict.services.memorystore import *
import pytest


@pytest.fixture
def task_repo():
    proj_repo = ProjectRepository()

    class MockModel:
        def predict(self, data):
            return data

    project = Project('proj', MockModel())
    proj_repo.create(project)
    return  TaskRepository(proj_repo)


@pytest.fixture
def sample_project():
    class MockModel:
        def predict(self, data):
            return data

    return Project('proj', MockModel())


def test_base_init():
    db = MemoryStoreBase()
    assert db.entities == {}


def test_base_create():
    db = MemoryStoreBase()
    class A:
        uid = 14
    a = A()
    db.create(a)
    assert db.entities[14] == a

    class B:
        key = 12

    b = B()
    with pytest.raises(AttributeError):
        db.create(b)

    db.create(b, key='key')
    assert db.entities[12] == b

    class C:
        uid = B.key

    c = C()

    db.create(c)
    assert db.entities[B.key] == c


def test_base_get_all():
    db = MemoryStoreBase()

    class A:
        uid = 14

    a = A()
    a2 = A()
    a2.uid = 12
    db.create(a)
    db.create(a2)
    assert db.get_all() == {12: a2, 14: a}


def test_base_get():
    db = MemoryStoreBase()

    class A:
        uid = 14

    a = A()
    db.create(a)
    assert db.get(14) == a
    with pytest.raises(KeyError):
        db.get(404)


def test_base_delete():
    db = MemoryStoreBase()

    class A:
        uid = 14

    a = A()
    db.create(a)
    db.delete(14)
    assert 14 not in db.entities

    with pytest.raises(KeyError):
        db.delete(404)


def test_base_update():
    class A:
        def __init__(self, uid, a, b, c):
            self.__dict__.update(locals())

    db = MemoryStoreBase()
    a = A(1, 2, 3, 4)
    db.create(a)
    db.update({'uid': 1, 'a': 45})
    assert a.a == 45

    with pytest.raises(KeyError):
        db.update({'uid': 404, 'b': 12})
    assert a.b != 12

    with pytest.raises(KeyError):
        db.update({'id': 404, 'b': 12})
    assert a.b != 12


def test_project_repo_init():
    repo = ProjectRepository()
    assert repo.task_repository is None


def test_project_repo_create(sample_project):
    repo = ProjectRepository()
    repo.create(sample_project)
    assert sample_project in repo.entities.values()


def test_project_repo_get(sample_project):
    repo = ProjectRepository()
    repo.create(sample_project)
    assert repo.get(sample_project.uid) is sample_project
    with pytest.raises(KeyError):
        assert repo.get('invalid')


def test_project_repo_delete(sample_project):
    repo = ProjectRepository()
    repo.create(sample_project)
    repo.delete(sample_project.uid)
    assert sample_project not in repo.entities.values()
    with pytest.raises(KeyError):
        repo.delete('invalid')
    with pytest.raises(KeyError):
        repo.delete(sample_project.uid)


def test_project_repo_get_tasks_for_project():
    class MockProject(object):
        def __init__(self):
            self.uid = 12
            self.tasks = [1, 2, 3]
    proj = MockProject()
    repo = ProjectRepository()
    repo.create(proj)
    assert repo.get_tasks_for_project(12) == [1, 2, 3]


def test_task_repo_init():
    proj_repo = ProjectRepository()
    repo = TaskRepository(proj_repo)
    assert repo.project_repository == proj_repo


def get_project_repo_from_task_repo(task_repo):
    return list(task_repo.project_repository.entities.values())[0]


def test_task_repo_create(task_repo):
    project = get_project_repo_from_task_repo(task_repo)
    task = Task('a', project_uid=project.uid)
    task_repo.create(task)
    project = get_project_repo_from_task_repo(task_repo)
    assert task.project_uid == project.uid
    assert task in project.tasks


def test_task_repo_get(task_repo):
    project = get_project_repo_from_task_repo(task_repo)
    task = Task('a', project_uid=project.uid)
    task_repo.create(task)
    assert task_repo.get(task.uid)
    with pytest.raises(KeyError):
        task_repo.get('invalid')


def test_task_repo_delete(task_repo):
    project = get_project_repo_from_task_repo(task_repo)
    task = Task('a', project_uid=project.uid)
    task_repo.create(task)
    task_repo.delete(task.uid)
    assert task not in project.tasks
    assert task.uid not in task_repo.entities
    with pytest.raises(KeyError):
        task_repo.get(task.uid)
    with pytest.raises(KeyError):
        task_repo.delete(task.uid)


def test_project_repo_update(task_repo):
    project = get_project_repo_from_task_repo(task_repo)
    task = Task('a', project_uid=project.uid)
    task_repo.create(task)
    update_dict = {'uid': task.uid, 'name': 'b'}
    task_repo.update(update_dict)
    assert task.name == 'b'
    with pytest.raises(KeyError):
        update_dict = {'uid': 'invalid', 'name': 'c'}
        task_repo.update(update_dict)