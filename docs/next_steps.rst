.. _next_steps:

Next Steps
==========
ProjectPredict is still in development, and numerous improvement can be made. Amoung them are:

* The default learning algorithm, the `Gaussian Process Regressor
  <http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_ model from
  scikit-learn does not perform adequately for a wide variety of data sets. Some alternatives would be to use `GPFlow
  <http://gpflow.readthedocs.io/en/latest/>`_ or `pymc3 <https://docs.pymc.io>`_ to determine the distribution using
  non-parametric Bayesian methods.
* The visualization capabilities are admittedly somewhat primitive and lacks the ability to interact with the project
  graph. A much better solution would be to set up a small web server and use `cytoscape <http://js.cytoscape.org>`_ to
  view and interact with the model.
* Durations are internally represented as Python datetime.timedelta objects. It might be better to allow users to
  specify how long a working day is (8 hours) and define a day to be the length of the working hours.
* Completing a task should update the model so that it learns as the project progresses.

.. Link to branch with webserver under development