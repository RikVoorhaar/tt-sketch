r"""Implements sketching methods for given types of tensors.

Each sketching method consists of a method to sketch ``omega``
:math:`\Omega_\mu` and to sketch ``psi`` :math:`\Psi_\mu` (following the
notation of the paper), using the output from the respective ``DRM.sketch_``
methods. 

For example, the ``sketch_omega_tt`` method sketches the tensor
:math:`\Omega_\mu` using the output of ``some_drm.sketch_tt``, where
``some_drm`` is any DRM that supports the ``CanSketchTT`` interface.
"""