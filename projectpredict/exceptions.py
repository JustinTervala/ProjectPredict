class SchemaValidationError(Exception):
    def __init__(self, data):
        super(SchemaValidationError, self).__init__()
        self.data = data