from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .utils import get_epsilon

_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class SO3:
    """Special orthogonal group for 3D rotations.

    Internal parameterization is (qw, qx, qy, qz). Tangent parameterization is
    (omega_x, omega_y, omega_z). Matches Mink's SO3.
    """

    wxyz: np.ndarray
    matrix_dim: int = 3
    parameters_dim: int = 4
    tangent_dim: int = 3
    space_dim: int = 3

    def __post_init__(self) -> None:
        if self.wxyz.shape != (self.parameters_dim,):
            raise ValueError(
                f"Expected wxyz to be a length 4 vector but got {self.wxyz.shape[0]}."
            )

    def __repr__(self) -> str:
        wxyz = np.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    def parameters(self) -> np.ndarray:
        return self.wxyz

    def copy(self) -> SO3:
        return SO3(wxyz=self.wxyz.copy())

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SO3:
        assert matrix.shape == (SO3.matrix_dim, SO3.matrix_dim)
        wxyz = np.empty(SO3.parameters_dim, dtype=np.float64)
        mujoco.mju_mat2Quat(wxyz, matrix.ravel())
        return SO3(wxyz=wxyz)

    @classmethod
    def identity(cls) -> SO3:
        return SO3(wxyz=_IDENTITY_WXYZ.copy())

    def as_matrix(self) -> np.ndarray:
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.wxyz)
        return mat.reshape(3, 3)

    def inverse(self) -> SO3:
        conjugate_wxyz = np.empty(4)
        mujoco.mju_negQuat(conjugate_wxyz, self.wxyz)
        return SO3(wxyz=conjugate_wxyz)

    def normalize(self) -> SO3:
        normalized_wxyz = np.array(self.wxyz)
        mujoco.mju_normalize4(normalized_wxyz)
        return SO3(wxyz=normalized_wxyz)

    def apply(self, target: np.ndarray) -> np.ndarray:
        assert target.shape == (SO3.space_dim,)
        rotated_target = np.empty(SO3.space_dim, dtype=np.float64)
        mujoco.mju_rotVecQuat(rotated_target, target, self.wxyz)
        return rotated_target

    def multiply(self, other: SO3) -> SO3:
        res = np.empty(self.parameters_dim, dtype=np.float64)
        mujoco.mju_mulQuat(res, self.wxyz, other.wxyz)
        return SO3(wxyz=res)

    def __matmul__(self, other: SO3 | np.ndarray) -> SO3 | np.ndarray:
        if isinstance(other, np.ndarray):
            return self.apply(other)
        return self.multiply(other)

    @classmethod
    def exp(cls, tangent: np.ndarray) -> SO3:
        axis = np.array(tangent, dtype=np.float64)
        theta = mujoco.mju_normalize3(axis)
        wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(wxyz, axis, theta)
        return SO3(wxyz=wxyz)

    def log(self) -> np.ndarray:
        q = np.array(self.wxyz)
        q *= np.sign(q[0])
        w, v = q[0], q[1:]
        norm = mujoco.mju_normalize3(v)
        if norm < get_epsilon(v.dtype):
            return np.zeros_like(v)
        return 2 * np.arctan2(norm, w) * v

    def adjoint(self) -> np.ndarray:
        return self.as_matrix()

    @classmethod
    def ljac(cls, other: np.ndarray) -> np.ndarray:
        theta = np.float64(mujoco.mju_norm3(other))
        t2 = theta * theta
        if theta < get_epsilon(theta.dtype):
            alpha = (1.0 / 2.0) * (
                1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))
            )
            beta = (1.0 / 6.0) * (
                1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))
            )
        else:
            t3 = t2 * theta
            alpha = (1 - np.cos(theta)) / t2
            beta = (theta - np.sin(theta)) / t3
        ljac = np.empty((3, 3))
        mujoco.mju_mulMatMat(ljac, other.reshape(3, 1), other.reshape(1, 3))
        inner_product = mujoco.mju_dot3(other, other)
        ljac[0, 0] -= inner_product
        ljac[1, 1] -= inner_product
        ljac[2, 2] -= inner_product
        ljac *= beta
        alpha_vec = alpha * other
        ljac[0, 1] += -alpha_vec[2]
        ljac[0, 2] += alpha_vec[1]
        ljac[1, 0] += alpha_vec[2]
        ljac[1, 2] += -alpha_vec[0]
        ljac[2, 0] += -alpha_vec[1]
        ljac[2, 1] += alpha_vec[0]
        ljac[0, 0] += 1.0
        ljac[1, 1] += 1.0
        ljac[2, 2] += 1.0
        return ljac

    @classmethod
    def ljacinv(cls, other: np.ndarray) -> np.ndarray:
        theta = np.float64(mujoco.mju_norm3(other))
        t2 = theta * theta
        if theta < get_epsilon(theta.dtype):
            beta = (1.0 / 12.0) * (
                1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0))
            )
        else:
            beta = (1.0 / t2) * (
                1.0 - (theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
            )
        ljacinv = np.empty((3, 3))
        mujoco.mju_mulMatMat(ljacinv, other.reshape(3, 1), other.reshape(1, 3))
        inner_product = mujoco.mju_dot3(other, other)
        ljacinv[0, 0] -= inner_product
        ljacinv[1, 1] -= inner_product
        ljacinv[2, 2] -= inner_product
        ljacinv *= beta
        alpha_vec = -0.5 * other
        ljacinv[0, 1] += -alpha_vec[2]
        ljacinv[0, 2] += alpha_vec[1]
        ljacinv[1, 0] += alpha_vec[2]
        ljacinv[1, 2] += -alpha_vec[0]
        ljacinv[2, 0] += -alpha_vec[1]
        ljacinv[2, 1] += alpha_vec[0]
        ljacinv[0, 0] += 1.0
        ljacinv[1, 1] += 1.0
        ljacinv[2, 2] += 1.0
        return ljacinv

    @classmethod
    def rjac(cls, other: np.ndarray) -> np.ndarray:
        return cls.ljac(-other)

    @classmethod
    def rjacinv(cls, other: np.ndarray) -> np.ndarray:
        return cls.ljacinv(-other)

    def jlog(self) -> np.ndarray:
        return self.rjacinv(self.log())
