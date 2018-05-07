

class Context(object):
    def __init__(self, task_repository, project_repository, serialization_service):
        self.task_repository = task_repository
        self.project_repository = project_repository
        self.serialization_service = serialization_service


def context_factory(config):
    if config.serialization.lower() == 'marshmallow':
        from .services.marshmallow import MarshmallowSerializationService
        serialization_service = MarshmallowSerializationService()
    else:
        raise ValueError('Unknown serialization service configuration {}'.format(config.serialization))

    if config.repository.lower() == 'memory':
        from .services.memorystore import TaskRepository, ProjectRepository
        project_repository = ProjectRepository()
        task_repository = TaskRepository(project_repository)
        project_repository.task_repository = task_repository
    else:
        raise ValueError('Unknown repository type {}'.format(config.repository))

    return Context(task_repository, project_repository, serialization_service)