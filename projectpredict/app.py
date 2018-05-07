from flask import Flask, jsonify, request, current_app, Blueprint
from projectpredict.context import context_factory
from projectpredict.config import Config
from enum import IntEnum
from projectpredict.project import Project
from projectpredict.exceptions import SchemaValidationError


project_blueprint = Blueprint('project_page', __name__)


class ReturnCodes(IntEnum):
    success = 200
    created = 201
    no_content = 204
    bad_request = 400
    not_found = 404


def create_app(config, model):
    app = Flask('projectpredict')
    context = context_factory(config)
    context.learning_model = model
    app.running_context = context
    app.register_blueprint(project_blueprint, url_prefix='/')
    return app


@project_blueprint.route('/projects', methods=['GET'])
def get_all_projects():
    projects = current_app.running_context.project_repository.get_all()
    return jsonify(
        {'projects': [current_app.running_context.serialization_service.dump(project) for project in
                      projects.values()]}), ReturnCodes.success


@project_blueprint.route('/projects', methods=['POST'])
def create_project():
    project_data = request.get_json()
    try:
        project_data = current_app.running_context.serialization_service.load(Project, project_data)
        for task in project_data['tasks']:
            task.set_duration_pdf(current_app.running_context.learning_model)
        project = Project.from_dict(project_data, current_app.running_context.learning_model)
        current_app.running_context.project_repository.create(project)
        return jsonify(current_app.running_context.serialization_service.dump(project)), ReturnCodes.created
    except SchemaValidationError as e:
        return jsonify({'error': e.data}), ReturnCodes.bad_request


@project_blueprint.route('/projects', methods=['PUT'])
def update_project():
    project_data = request.get_json()
    try:
        project_data = current_app.running_context.serialization_service.load(project_data)
    except SchemaValidationError as e:
        return {'error': e.data}, ReturnCodes.bad_request
    project = current_app.running_context.project_repository.update(project_data)
    return jsonify(current_app.running_context.serialization_service.dump(project)), ReturnCodes.success


@project_blueprint.route('/projects/{uid}', methods=['GET'])
def get_project(uid):
    try:
        return jsonify(
            current_app.running_context.serialization_service.dump(
                current_app.running_context.project_repository.get(uid))), ReturnCodes.success
    except KeyError:
        return 'Project not found', ReturnCodes.not_found


@project_blueprint.route('/projects/{uid}', methods=['DELETE'])
def delete_project(uid):
    try:
        current_app.running_context.project_repository.delete(uid)
        return None, ReturnCodes.no_content
    except KeyError:
        return 'Project not found', ReturnCodes.not_found


@project_blueprint.route('/projects/{uid}/recommendation', methods=['GET'])
def get_next_task_recommendation(uid):
    recommendation_data = request.get_data()
    try:
        recommendation = current_app.running_context.project_repository.get(uid).recommend_next(**recommendation_data)
        return jsonify(
            {'recommended_tasks': [
                current_app.running_context.serialization_service.dump(task) for task in
                recommendation]}), ReturnCodes.success
    except KeyError:
        return 'Project not found', ReturnCodes.not_found


@project_blueprint.route('/projects/{uid}/tasks', methods=['GET'])
def get_all_tasks_for_projects(uid):
    try:
        tasks = current_app.running_context.project_repository.get_tasks_for_project(uid)
        return jsonify(
            {'tasks': [current_app.running_context.serialization_service(task) for task in tasks]}), ReturnCodes.success
    except KeyError:
        return 'Project not found', ReturnCodes.not_found


@project_blueprint.route('/tasks', methods=['POST'])
def create_task():
    task_data = request.get_json()
    task = current_app.running_context.serialization_service.load(task_data)
    current_app.running_context.task_repository.create(task)
    return jsonify(current_app.running_context.serialization_service.dump(task)), ReturnCodes.created


@project_blueprint.route('/tasks', methods=['PUT', 'PATCH'])
def update_task():
    task_data = request.get_json()
    task = current_app.running_context.project_repository.update(task_data)
    return jsonify(current_app.running_context.serialization_service.dump(task)), ReturnCodes.success


@project_blueprint.route('/tasks/{uid}', methods=['GET'])
def get_task(uid):
    try:
        return jsonify(
            current_app.running_context.serialization_service.dump(
                current_app.running_context.task_repository.get(uid))), ReturnCodes.success
    except KeyError:
        return 'Task not found', ReturnCodes.not_found


@project_blueprint.route('/tasks/{uid}', methods=['DELETE'])
def delete_task(uid):
    try:
        current_app.running_context.task_repository_repository.delete(uid)
        return None, ReturnCodes.no_content
    except KeyError:
        return 'Task not found', ReturnCodes.not_found


@project_blueprint.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    from projectpredict.learningmodels import GaussianProcessRegressiorModel

    learning_model = GaussianProcessRegressiorModel()
    with open('training_data.csv') as f:
        # format data
        data = []
        learning_model.train(data, [])
    app = create_app(Config, learning_model)
    app.run()
