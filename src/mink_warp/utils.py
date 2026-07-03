import mujoco


def get_freejoint_dims(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """Get all floating joint configuration and tangent indices."""
    q_ids: list[int] = []
    v_ids: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            q_ids.extend(range(qadr, qadr + 7))
            v_ids.extend(range(vadr, vadr + 6))
    return q_ids, v_ids
