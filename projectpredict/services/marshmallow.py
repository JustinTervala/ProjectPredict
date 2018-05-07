from marshmallow import Schema, fields, ValidationError, post_load, post_dump

from projectpredict.exceptions import SchemaValidationError
from projectpredict.task import *
from projectpredict.project import Project
from projectpredict.pdf import PdfFactory, TimeUnits, DurationPdf, DatePdf


class BaseSchema(Schema):
    """Base schema for the execution database.
    This base class adds functionality to strip null fields from serialized objects and attaches the
    execution_db.session on load
    """
    __skipvalues = (None, [], [{}])

    @post_dump
    def _do_post_dump(self, data):
        return self.remove_skip_values(data)

    def remove_skip_values(self, data):
        """Removes fields with empty values from data
        Args:
            data (dict): The data passed in
        Returns:
            (dict): The data with forbidden fields removed
        """
        return {
            key: value for key, value in data.items()
            if value not in self.__skipvalues
        }


class TimeUnitField(fields.Str):
    def _serialize(self, value, attr, obj):
        return value.name

    def _deserialize(self, value, attr, data):
        try:
            return TimeUnits.from_string(value)
        except ValueError:
            raise ValidationError('Unknown time unit {}'.format(value))


class PdfSchema(BaseSchema):
    type = fields.Str(required=True, attribute='type')
    parameters = fields.Function(lambda obj: obj.to_dict())

    @post_load
    def make_pdf(self, data):
        try:
            return PdfFactory.create(data['type'], data['parameters'])
        except KeyError:
            raise ValidationError('Unknown pdf type')


class DurationPdfSchema(BaseSchema):
    pdf = fields.Nested(PdfSchema(), required=True)
    units = TimeUnitField()

    @post_load
    def make_duration_pdf(self, data):
        return DurationPdf(**data)


class DatePdfSchema(BaseSchema):
    mean_datetime = fields.DateTime(required=True)
    pdf = fields.Nested(PdfSchema(), required=True)
    units = TimeUnitField()

    @post_load
    def make_duration_pdf(self, data):
        return DatePdf(**data)


class TaskSchema(BaseSchema):
    uid = fields.UUID(required=True)
    name = fields.Str(required=True)
    duration_pdf = fields.Nested(DurationPdfSchema())
    earliest_start_date_pdf = fields.Nested(DatePdfSchema, required=False)
    latest_finish_date_pdf = fields.Nested(DatePdfSchema, required=False)
    data = fields.Raw()
    deadline_priority = fields.Integer()
    start_time = fields.DateTime(required=False)
    completion_time = fields.DateTime(required=False)
    project_uid = fields.UUID(required=True)

    @post_load
    def make_task(self, data):
        return Task(**data)


class DependencySchema(BaseSchema):
    source = fields.UUID(required=True)
    destination = fields.UUID(required=True)


class ProjectSchema(BaseSchema):
    uid = fields.UUID(required=True)
    name = fields.Str(required=True)
    dependencies = fields.Nested(DependencySchema, many=True, attribute='dependencies')
    tasks = fields.Nested(TaskSchema, many=True)


class MarshmallowSerializationService(object):
    _schema_lookup = {
        DurationPdf: DurationPdfSchema(),
        DatePdf: DatePdfSchema(),
        Task: TaskSchema(),
        Project: ProjectSchema()
    }

    @classmethod
    def dump(cls, model):
        return cls._schema_lookup[model.__class__].dump(model).data

    @classmethod
    def load(cls, model, data):
        loaded = cls._schema_lookup[model].load(data, partial=True)
        if loaded.errors:
            raise SchemaValidationError(loaded.errors)
        return loaded.data
