

class MemoryStoreBase(object):
    default_key = 'uid'

    def __init__(self):
        self.entities = {}

    def get_all(self):
        return self.entities

    def get(self, uid):
        return self.entities[uid]

    def create(self, entity, key=None):
        key = key or self.default_key
        self.entities[getattr(entity, key)] = entity

    def delete(self, uid):
        self.entities.pop(uid)

    def update(self, data, key=None):
        key = key or self.default_key
        entity = self.get(data[key])
        entity.__dict__.update(data)


class ProjectRepository(MemoryStoreBase):
    def __init__(self):
        super(ProjectRepository, self).__init__()
        self.task_repository = None

    def get_tasks_for_project(self, project_uid):
        return self.get(project_uid).tasks

    def update(self, data, key='uid'):
        project = self.get(data[key])
        project.update_from_dict(data)


class TaskRepository(MemoryStoreBase):
    def __init__(self, project_db):
        super(TaskRepository, self).__init__()
        self.project_repository = project_db

    def _get_project_for_task(self, project_uid):
        return self.project_repository.get(project_uid)

    def create(self, task, key=None):
        project = self._get_project_for_task(task.project_uid)
        project.add_task(task)
        super(TaskRepository, self).create(task)

    def delete(self, uid):
        task = self.get(uid)
        project = self._get_project_for_task(task.project_uid)
        project.remove_node(task)
        super(TaskRepository, self).delete(uid)