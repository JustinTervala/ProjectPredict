
class InvalidProject(Exception):
    """Exception thrown when a project is determined to be invalid

    Attributes:
        errors(list[str]|str): The errors found with the Project

    Args:
        errors(list[str]|str): The errors found with the Project
    """
    def __init__(self, errors):
        super(InvalidProject, self).__init__()
        self.errors = errors

    def __repr__(self):
        return 'Project is invalid. Errors: {}'.format(self.errors)
