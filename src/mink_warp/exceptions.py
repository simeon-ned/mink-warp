"""Exceptions specific to mink-warp."""

from typing import Sequence

import mujoco


class MinkWarpError(Exception):
    """Base class for mink-warp exceptions."""


class UnsupportedFrame(MinkWarpError):
    """Exception raised when a frame type is unsupported."""

    def __init__(self, frame_type: str, supported_types: Sequence[str]):
        message = (
            f"{frame_type} is not supported. "
            f"Supported frame types are: {supported_types}"
        )
        super().__init__(message)


class InvalidFrame(MinkWarpError):
    """Exception raised when a frame name is not found in the robot model."""

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        model: mujoco.MjModel,
    ):
        if frame_type == "body":
            available = [model.body(i).name for i in range(model.nbody)]
        elif frame_type == "site":
            available = [model.site(i).name for i in range(model.nsite)]
        else:
            assert frame_type == "geom"
            available = [model.geom(i).name for i in range(model.ngeom)]

        message = (
            f"{frame_type} '{frame_name}' does not exist in the model. "
            f"Available {frame_type} names: {available}"
        )
        super().__init__(message)


class InvalidKeyframe(MinkWarpError):
    """Exception raised when a keyframe name is not found in the robot model."""

    def __init__(self, keyframe_name: str, model: mujoco.MjModel):
        available = [model.key(i).name for i in range(model.nkey)]
        message = (
            f"Keyframe {keyframe_name} does not exist in the model. "
            f"Available keyframe names: {available}"
        )
        super().__init__(message)


class TaskDefinitionError(MinkWarpError):
    """Exception raised when a task definition is ill-formed."""


class TargetNotSet(MinkWarpError):
    """Exception raised when attempting to use a task with an unset target."""

    def __init__(self, cls_name: str):
        super().__init__(f"No target set for {cls_name}")


class InvalidTarget(MinkWarpError):
    """Exception raised when the target is invalid."""


class InvalidGain(MinkWarpError):
    """Exception raised when the gain is outside the valid range."""


class InvalidDamping(MinkWarpError):
    """Exception raised when the damping is outside the valid range."""
