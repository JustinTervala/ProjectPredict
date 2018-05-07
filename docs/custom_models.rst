.. _custom_models:

Customized Learning Models
==========================
ProjectPredict comes with a Gaussian Process Regression model, however you may find this model unsuitable for your data.
To make your own model, you only need to follow a minimal interface -- the only requirement is that you have a method
named "predict" that accepts the dictionary of data associated with a task and returns a DurationPdf. For simplicity,
assume your tasks have a "points" value in their data, and your model simply returns a DurationPdf wrapping a
DeterministicPdf containing with the same value as the points passed into it. You could write this as

.. code-block:: python
   :linenos:

    class SimpleModel(object):
        def __init__(self, units=TimeUnits.hours):
            self.units = units

        def predict(self, input_data):
            return DurationPdf(DeterministicPdf(input_data['points']), units=self.units)
