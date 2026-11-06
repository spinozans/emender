"""Opt-in experiment; never install in a training runner before certification."""
from . import c14_b_control_native as native
from .c14_phase2_staged_kernel import kernel
from .c14_launch_shape import install as install_shape


def install():
    native.kernel = kernel
    return install_shape()


def install_gluon():
    from .c14_phase2_gluon_kernel import kernel as explicit_kernel
    native.kernel = explicit_kernel
    return install_shape()


def install_rowkeys():
    from .c14_phase2_gluon_rowkeys_kernel import kernel as explicit_kernel
    native.kernel = explicit_kernel
    return install_shape()
