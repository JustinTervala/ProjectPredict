.. _recommendations:

The Recommendation Engine
=========================
ProjectPredict comes with a flexible recommendation engine which can be used to determine which tasks should be started
next. You can constrain the set of tasks both by a minimum and maximum number of tasks as well as by using custom
constraint functions. You can also specify if all tasks must be completed before the next tasks can begin or if a new
set or tasks can be started whenever any of the tasks in the recommended set completes. The default algorithm selects a
set of tasks which maximizes the sum of the total float across the project, weighted by the importance of some tasks'
deadlines and the risk tolerance.

The Default Algorithm
---------------------
The default algorithm iterates through all possible combinations of tasks which can be started (all tasks with no
uncompleted predecessors) and, for each combination infers the latest start date, earliest finish date, and total float
of each task in the project assuming that the combination of tasks is begun at the current time. For each combination it
creates two scores, the float score and the precision score as defined by

:math:`s_f = \sum_{\text{tasks}\; i} { w_i \mu_i}`

:math:`s_p = \sum_{\text{tasks}\; i} { w_i /\sigma_i}`

Where :math:`\mu_i` is the mean total float for task :math:`i`, :math:`\sigma_i` is the mean total float for task
:math:`i`, and :math:`w_i` is the weight of the deadline for task :math:`i` (defaults to 1 if unspecified).

These scores are then used to select the best combination of tasks. First each score is scaled linearly between 0 and 1
based on the minimum and maximum of both scores.

:math:`\bar{s_f} = \frac{s_f - \min_{\text{task set i}}{s_{f_i}}}{\max_{\text{task set i}}{s_{f_i}}}`

:math:`\bar{s_p} = \frac{s_p - \min_{\text{task set i}}{s_{p_i}}}{\max_{\text{task set i}}{s_{p_i}}}`

Where :math:`\bar{s_f}` and :math:`\bar{s_p}` are the scaled total float score and scaled precision respectively for a
task. These two are then combined with a risk tolerance factor, :math:`r`, a value from 0 to 1, to obtain the combined
score :math:`s`, using :math:`s = r \bar{s_f} + (1-r)\bar{s_p}`. The recommended task set is the set of tasks which has the maximum
combined score.

Customization
-------------
The recommendation algorithm can be customized by specifying a scoring function which will accept the earliest start
date, latest start date, earliest finish date, latest finish date, and total float samples generated for a task set as
well as some optional keyword arguments. A recommendation selection function must also be supplied which accepts the
generated scores and some optional keyword arguments. A list of constraints can be specified by supplying a list of
functions which accept the project and a proposed set of tasks and returns a boolean indicating if the set of task
satisfies the constraints. For examples see :ref:`Recommendations with Constraints`

.. link to actual api docs