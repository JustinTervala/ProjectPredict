.. _custompdfs:

Customized PDFs
===============
ProjectPredict only comes with two built in PDFs, the DeterministicPdf and the GaussianPdf, however, making a custom PDF
is straightforward, and requires only a minimal interface.

PDFs from Scipy
---------------
Generating custom PDFs from `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ distributions
requires only that you extend from the projectpredict.pdf.SciPyPdf base class and provide a constructor. For example,
to provide a half-normal distribution from `scipy.stats.halfnorm
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html#scipy.stats.halfnorm>`_, you could write
the following class

.. code-block:: python
   :linenos:

    from scipy.stats import halfnorm
    from math import sqrt

    class HalfNormalPdf(SciPyPdf):
        def __init__(mean, variance):
            super(HalfNormalPdf, self).__init__(halfnorm(loc=mean, scale=sqrt(variance)))

Fully Custom PDFs
-----------------
All PDFs must provide the following methods:

* A method called sample() which takes no parameters and return a random sample from the PDF in the form of a float
* A field or property called "mean" which holds the mean of the pdf
* A field or property called "variance" which holds the variance of the pdf

For example, a uniform PDF from Python's built-in random module could be written as

.. code-block:: python
   :linenos:

   from rand import uniform

   class UniformPdf(object):
       def __init__(low, high):
           self.low = low
           self.high = high
           self.mean = (high - low) / 2
           self.variance = 1/12 * (high - low)**2

       def sample();
         return uniform(low, high)

