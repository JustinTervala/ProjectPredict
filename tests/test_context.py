from projectpredict.context import *
import pytest


def test_context_factory():
    from projectpredict.services.marshmallow import MarshmallowSerializationService
    from projectpredict.services.memorystore import ProjectRepository, TaskRepository
    class Config:
        serialization = 'marshmallow'
        repository = 'memory'
    context = context_factory(Config)
    assert isinstance(context.serialization_service, MarshmallowSerializationService)
    assert isinstance(context.project_repository, ProjectRepository)
    assert isinstance(context.task_repository, TaskRepository)
    assert context.task_repository.project_repository is context.project_repository


def test_context_factory_invalid_serialization():
    class Config:
        serialization = 'invalid'
        repository = 'memory'
    with pytest.raises(ValueError):
        context_factory(Config)


def test_context_factory_invalid_repository():
    class Config:
        serialization = 'marshmallow'
        repository = 'invalid'
    with pytest.raises(ValueError):
        context = context_factory(Config)

