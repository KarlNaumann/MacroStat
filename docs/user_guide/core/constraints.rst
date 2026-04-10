================
Constraints
================
.. currentmodule:: macrostat.core.constraints

Many SFC and behavioral models have *adding-up* relationships between
parameters. Tobin portfolio shares must sum to one, the corresponding
rate sensitivities must sum to zero, factor income shares must close
to total income, and so on. The :class:`LinearConstraint` system lets
a model declare these relationships once, and MacroStat enforces them
both at parameter initialization and inside every simulation step,
without breaking autograd.

Constructor
~~~~~~~~~~~
.. autosummary::

   LinearConstraint

Properties and methods
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   LinearConstraint.free_params
   LinearConstraint.derived_param
   LinearConstraint.enforce


Residual parameterization
=========================

A :class:`LinearConstraint` enforces

.. math::

   \sum_{i \in \text{group}} p_i = T

by treating the **last** parameter in ``param_names`` as the residual
of the others:

.. math::

   p_{\text{derived}} = T - \sum_{i \in \text{free}} p_i.

The free parameters remain independent. The derived parameter is
computed from them. Because the derivation is a linear combination
of tensors, the operation is fully differentiable: autograd records
the dependency

.. math::

   \frac{\partial p_{\text{derived}}}{\partial p_i} = -1
   \quad \text{for every free } p_i,

and the gradient of any downstream loss with respect to
``p_derived`` is zero by construction. This is the correct behavior:
``p_derived`` is no longer a free parameter, so its sensitivity is
absorbed into the free parameters of the same group. See
:doc:`../diff` for an "Expected disagreements" note that revisits
this from the Jacobian side.


Declaring constraints on a model
================================

Constraints are declared by overriding
:meth:`~macrostat.core.parameters.Parameters.get_constraints` on the
model's :class:`~macrostat.core.parameters.Parameters` subclass. The
method returns an ordered tuple of :class:`LinearConstraint`
instances. As an example, the GL06INSOUT model declares five
constraints for its 4-asset Tobin portfolio (M1, M2, Bills, Bonds),
with M1 cash as the residual asset in every group:

.. literalinclude:: ../../../src/macrostat/models/GL06INSOUT/parameters.py
   :language: python
   :pyobject: ParametersGL06INSOUT.get_constraints

Each :class:`LinearConstraint` accepts:

* ``param_names`` (``tuple[str, ...]``): all parameters in the group.
  The **last** entry is the derived (residual) parameter; everything
  before it is free.
* ``target`` (``float``): the value the group must sum to.

The order of constraints in the returned tuple is preserved. If a
parameter is the derived parameter in one constraint and a free
parameter in another, MacroStat does not topologically reorder for
you. The user is responsible for ordering such that each constraint
is well-defined when it runs.


Adding a constraint to your own model
=====================================

To add a new constraint group to a model:

1. Open the model's ``parameters.py`` and locate the
   :class:`~macrostat.core.parameters.Parameters` subclass.
2. Override or extend the ``get_constraints()`` method so that it
   returns a tuple of :class:`LinearConstraint` instances. If the
   parent already declares constraints, return ``super().get_constraints() + (...)``.
3. Choose which parameter in the group is the residual. This is
   typically the *buffer-stock* asset, the *anchor* category, or
   any parameter you do not want to estimate independently. List it
   **last** in ``param_names``.
4. Pick the ``target`` (1.0 for shares, 0.0 for sensitivities, or
   any other linear sum).
5. Re-instantiate the parameters object. ``verify_constraints()`` is
   called automatically and will raise if the constraint is
   ill-formed; ``enforce_constraints()`` then adjusts the residual
   parameter to satisfy the sum.

Existing models that do not declare any constraints need no
changes. Constraints are opt-in.


Verification and enforcement
============================

Constraints are checked and enforced at several points in the
parameter lifecycle.

**Init-time verification.**
:meth:`~macrostat.core.parameters.Parameters.verify_constraints` is
called once when a :class:`~macrostat.core.parameters.Parameters`
instance is constructed. It catches three classes of error:

* A constraint references a parameter name that does not exist in
  ``self.values``. The raised :class:`ValueError` lists the unknown
  name and the available parameters.
* The same parameter is declared as the derived (residual)
  parameter in two or more constraints. The raised
  :class:`ValueError` lists the duplicate names.
* A parameter is the derived parameter in one constraint and a
  free parameter in another. The raised :class:`ValueError`
  describes the circular dependency.

If the default values stored in ``self.values`` do not satisfy a
declared constraint, ``verify_constraints`` does not raise. It
instead emits a warning of the form
``"Constraint violation in defaults: sum(...) = ..., target = ...."``
so the user knows that the residual parameter will be adjusted on
the next call to ``enforce_constraints``.

**Init-time enforcement.**
:meth:`~macrostat.core.parameters.Parameters.enforce_constraints`
runs immediately after verification. For each constraint, it
overwrites the derived parameter's value with
``target - sum(free)``. This is a float-level operation that
modifies ``self.values`` in place.

**Step-time enforcement.** During simulation,
:meth:`~macrostat.core.behavior.Behavior.apply_parameter_shocks`
applies any scenario shocks to the parameter tensors and then runs
each constraint's :meth:`LinearConstraint.enforce` again on the
shocked tensors. This second pass keeps the adding-up identity
intact even when scenarios shock free parameters in the group, and
because it operates on tensors with ``requires_grad=True``, autograd
sees the residual relationship at every timestep.

**Helpers for callers.**
:meth:`~macrostat.core.parameters.Parameters.get_free_param_names`
returns the parameter names that are not derived by any
constraint. Use it whenever you want to iterate over the parameters
that are actually independent (for calibration, sampling, or
diagnostics).


See also
========

* :doc:`parameters` for the full Parameters API including the
  constraint hooks.
* :doc:`behavior` for the role of ``apply_parameter_shocks`` in
  step-time enforcement.
* :doc:`../diff` for how the residual parameterization shows up in
  numerical and autograd Jacobians.
* :doc:`../constraints_example` for a runnable notebook walkthrough.
