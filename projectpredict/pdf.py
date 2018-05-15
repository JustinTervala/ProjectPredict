from datetime import timedelta
from enum import Enum
from math import fabs, sqrt

from scipy.stats import norm


class SciPyPdf(object):
    def __init__(self, pdf):
        self.pdf = pdf

    def sample(self):
        """Get a sample from the PDF

        Returns:
            float: A sample from the PDF
        """
        return self.pdf.rvs()

    @property
    def mean(self):
        """float: The mean of the PDF
        """
        return self.pdf.mean()

    @property
    def variance(self):
        """float: The variance of the PDF
        """
        return self.pdf.var()

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and fabs(self.mean - other.mean) < 1e-6
                and fabs(self.variance - other.variance) < 1e-6)

    def __repr__(self):
        return repr(self.pdf)


class GaussianPdf(SciPyPdf):
    """A PDF representing a Gaussian distribution

    Attributes:
        pdf (norm): The Gaussian pdf object

    Args:
        pdf (norm): The Gaussian pdf object
    """
    type = 'gaussian'

    def __init__(self, mean, variance):
        super(GaussianPdf, self).__init__(norm(loc=mean, scale=sqrt(variance)))

    @classmethod
    def from_dict(cls, dict_in):
        """Creates a GaussianPdf from a dictionary

        Args:
            dict_in (dict): The dict to create the PDF from. Must contain keys for 'mean' and 'variance'

        Returns:
            GaussianPdf: The constructed Gaussian PDF
        """
        return cls(dict_in['mean'], dict_in['variance'])

    def to_dict(self):
        """Gets a dictionary representation of this PDF

        Returns:
            dict: The dictionary representation of this PDF
        """
        return {'mean': self.mean, 'variance': self.variance}


class DeterministicPdf(object):
    """A PDF representing a Gaussian distribution

    Attributes:
        pdf (float): The exact value to be returned by the sample() function

    Args:
        value (float): The exact value to be returned by the sample() function
    """
    type = 'deterministic'

    def __init__(self, value):
        self.pdf = value

    def sample(self):
        """Get a sample from the PDF. Will always return the value passed into the constructor.

        Returns:
            float: The value passed into the constructor
        """
        return self.pdf

    @property
    def mean(self):
        """float: The mean of the PDF. Always equal to the value passed into the constructor
        """
        return self.pdf

    @property
    def variance(self):
        """float: The variance of the PDF. Will always return 0
        """
        return 0

    def __eq__(self, other):
        return (isinstance(other, DeterministicPdf)
                and fabs(self.mean - other.mean) < 1e-6)

    @classmethod
    def from_dict(cls, dict_in):
        """Creates a DeterministicPdf from a dictionary

        Args:
            dict_in (dict): The dict to create the PDF from. Must contain keys for 'mean'

        Returns:
            DeterministicPdf: The constructed deterministic PDF
        """
        return cls(dict_in['mean'])

    def to_dict(self):
        """Gets a dictionary representation of this PDF

        Returns:
            dict: The dictionary representation of this PDF
        """
        return {'mean': self.mean}


class PdfFactory(object):
    """Factory to construct PDFs from dictionaries
    """
    pdf_registry = {DeterministicPdf.type: DeterministicPdf, GaussianPdf.type: GaussianPdf}

    @classmethod
    def create(cls, pdf_type, parameters):
        """Create a PDF

        Args:
            pdf_type (str): The type of PDF to construct. Must match an entry in the pdf_registry
            parameters (dict): The parameters from which to construct the PDF from.

        Returns:
            The constructed PDF
        """
        return cls.pdf_registry[pdf_type].from_dict(parameters)


class TimeUnits(Enum):
    """Enum representing possible units of time
    """
    milliseconds = 1
    seconds = 2
    minutes = 3
    hours = 4
    days = 5
    weeks = 6

    @staticmethod
    def to_timedelta(units, value):
        """Converts a TimeUnits and a value to a timedelta

        Args:
            units (TimeUnits): The units to use with the timedelta
            value (float): The value to use in the timedelta

        Returns:
            timedelta: The timedelta with the given units and value
        """
        return timedelta(**{units.name: value})

    @classmethod
    def from_string(cls, value):
        """Converts a string to a TimeUnits

        Args:
            value (str): The string to convert

        Returns:
            TimeUnits: The converted timeunit

        Raises:
            ValueError: If no matching string is found.
        """
        unit = next((unit for unit in cls if unit.name == value), None)
        if unit is None:
            raise ValueError('Unknown unit {}. Valid units are {}'.format(value, [unit.name for unit in cls]))
        return unit


class DurationPdf(object):
    """A probability density function over a time duration

    Attributes:
        pdf: A probability density function object which provides a mechanism for sampling via a sample() method
        units (TimeUnits): The units to use for the duration

    Args:
        pdf: A probability density function object
        units (TimeUnits, optional): The units to use for the duration. Defaults to TimeUnits.seconds
    """

    def __init__(self, pdf, units=TimeUnits.seconds):
        self.pdf = pdf
        self.units = units

    @property
    def mean(self):
        """timedelta: The mean value of this PDF
        """
        return TimeUnits.to_timedelta(self.units, self.pdf.mean)

    def sample(self, minimum=None):
        """Get a sample from the distribution

        Args:
            minimum (timedelta): The minimum duration

        Returns:
            timedelta: A sample from the distribution
        """

        sample = TimeUnits.to_timedelta(self.units, self.pdf.sample())
        if minimum:
            while sample < minimum:
                sample = TimeUnits.to_timedelta(self.units, self.pdf.sample)
        return sample

    def __eq__(self, other):
        return self.pdf == other.pdf and self.units == other.units


class DatePdf(object):
    """A probability density function over a datetime.

    Attributes:
        mean_datetime (datetime): A datetime to use as the mean value
        pdf: A probability density function object which provides a sampling mechanism via a sample() method
        units (TimeUnits): The units to use for the pdf samples

    Args:
        mean_datetime (datetime): A datetime to use as the mean value
        pdf: A probability density function object
        units (TimeUnits, optional): The units to use for pdf samples. Defaults to TimeUnits.seconds
    """

    def __init__(self, mean_datetime, pdf, units=TimeUnits.seconds):
        self.mean_datetime = mean_datetime
        self.pdf = pdf
        self.units = units

    @property
    def mean(self):
        """timedelta: The mean value of this PDF
        """
        return self.mean_datetime + TimeUnits.to_timedelta(self.units, self.pdf.mean)

    def sample(self):
        """Get a sample from the distribution

        Returns:
            datetime: A sample from the distribution
        """
        return self.mean_datetime + TimeUnits.to_timedelta(self.units, self.pdf.sample())

    def __eq__(self, other):
        return (self.mean_datetime - other.mean_datetime < timedelta(microseconds=100)
                and self.pdf == other.pdf
                and self.units == other.units)
