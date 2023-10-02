from random import random, choices
import numpy as np
from scipy.spatial.transform import Rotation


def patient_position_to_quarter_rotation(patient_position):
    match patient_position.lower():
        case "hfs":
            return np.array([0, 0, 0], dtype=float)
        case "hfp":
            return np.array([2, 0, 0], dtype=float)
        case "ffs":
            return np.array([2, 2, 0], dtype=float)
        case "ffp":
            return np.array([0, 2, 0], dtype=float)
        case _:
            raise ("bad string")


def patient_to_iec_room_shift(quarter_rotations, patient_shift):
    rot = Rotation.from_euler("xyz", 90 * quarter_rotations, degrees=True)
    return np.append(patient_shift[:3], rot.apply(patient_shift[3:]))


def iec_room_to_mac_shift(mac_dirs, iec_room_shift):
    return mac_dirs * iec_room_shift


def generate_patient_chart(num_sessions=3):
    rng = np.random.default_rng()
    # rng.choice()

    patient_position = choices(["hfs", "hfp"], weights=[90, 10])[0]
    quarter_rotations = patient_position_to_quarter_rotation(patient_position)
    yield "quater_rotations", quarter_rotations

    mac_dirs = np.array([1, 1, 1, 1, 1, 1])
    yield "mac xlate dir", mac_dirs[:3]
    yield "mac rot dir", mac_dirs[3:]

    init_couch = np.array([0, 0, 0, 0, 0, 0])
    yield "init_couch xlate", init_couch[:3]
    yield "init_couch rot", init_couch[3:]

    for n in range(num_sessions):
        patient_shift = rng.standard_normal(6)

        rotation_matrix = Rotation.from_euler("xyz", patient_shift[3:], degrees=True)
        rotation_matrix = rotation_matrix.as_matrix()
        yield "patient_shift rotx", rotation_matrix[0]
        yield "patient_shift roty", rotation_matrix[1]
        yield "patient_shift rotz", rotation_matrix[2]
        yield "patient_shift xlate", patient_shift[:3]

        iec_room_shift = patient_to_iec_room_shift(quarter_rotations, patient_shift)
        mac_shift = iec_room_to_mac_shift(
            mac_dirs=mac_dirs, iec_room_shift=iec_room_shift
        )
        yield "mac_shift xlate", mac_shift[:3]
        yield "mac_shift rot", mac_shift[3:]


from rich import print

print(list(generate_patient_chart(num_sessions=3)))
