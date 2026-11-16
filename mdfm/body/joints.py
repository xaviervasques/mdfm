"""The program's joint table and its forward kinematics, independent of the body model.

The model predicts 22 body joints: their names, their order and their tree are those of the program, the same on
the SMPL+H skeleton of v1 (mdfm/body/smplh.py) and on the open body Anny (mdfm/synth/openbody.py BODY_JOINTS and
PARENTS22, the joint map of docs/OPEN_BODY_PORT_DESIGN.md section 3.1; tests/test_body_joints.py checks the
equality). Only the rest skeleton under them changes with the body model, and it enters fk() as an argument.

Joint order: 0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee, 5 right_knee, 6 spine2, 7 left_ankle,
8 right_ankle, 9 spine3, 10 left_foot, 11 right_foot, 12 neck, 13 left_collar, 14 right_collar, 15 head,
16 left_shoulder, 17 right_shoulder, 18 left_elbow, 19 right_elbow, 20 left_wrist, 21 right_wrist.
"""
import torch

BODY_JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
               "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
               "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
N_BODY = 22
# parents of the 22 body joints, parent first
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def fk(R_local, transl, rest, parents=PARENTS):
    """Forward kinematics of the 22 body joints.

    R_local (..., 22, 3, 3) local joint rotations with row 0 the root orientation, transl
    (..., 3) m, rest (..., 22, 3) m rest joints (one subject's (22, 3) broadcasts over all
    timestamps). Leading dimensions broadcast. Returns joint positions (..., 22, 3) m and
    global rotations (..., 22, 3, 3), in the world the root orientation maps to."""
    batch = torch.broadcast_shapes(R_local.shape[:-3], transl.shape[:-1], rest.shape[:-2])
    bones = rest[..., 1:, :] - rest[..., list(parents[1:N_BODY]), :]
    g_rot = [R_local[..., 0, :, :].expand(batch + (3, 3))]
    g_pos = [(rest[..., 0, :] + transl).expand(batch + (3,))]
    for i in range(1, N_BODY):
        p = parents[i]
        g_rot.append(g_rot[p] @ R_local[..., i, :, :])
        g_pos.append(g_pos[p] + (g_rot[p] @ bones[..., i - 1, :, None])[..., 0])
    return torch.stack(g_pos, -2), torch.stack(g_rot, -3)
