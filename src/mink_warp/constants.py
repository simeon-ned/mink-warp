import mujoco

SUPPORTED_FRAMES = ("body", "geom", "site")

FRAME_TO_ENUM = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}

FRAME_TO_POS_ATTR = {
    "body": "xpos",
    "geom": "geom_xpos",
    "site": "site_xpos",
}

FRAME_TO_XMAT_ATTR = {
    "body": "xmat",
    "geom": "geom_xmat",
    "site": "site_xmat",
}


def constraint_width(constraint: int) -> int:
    """Dimensionality of an equality constraint in MuJoCo ``efc_*`` arrays."""
    return {
        mujoco.mjtEq.mjEQ_CONNECT.value: 3,
        mujoco.mjtEq.mjEQ_WELD.value: 6,
        mujoco.mjtEq.mjEQ_JOINT.value: 1,
        mujoco.mjtEq.mjEQ_TENDON.value: 1,
    }[constraint]
