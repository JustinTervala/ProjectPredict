

class Config(object):
    repository = 'mongodb'
    serialization = 'marshmallow'
    model_serialization = 'dill'
    model_path = 'projectpredict.learningmodels.gpy.Model'
    recommendation_score_func = 'projectpredict.models.project.Project._default_recommendation_score_func'
