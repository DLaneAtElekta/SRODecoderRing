from typing import Dict
from pprint import pprint

import torch


class SRODecoderRing(torch.nn.Module):
    def __init__(self):
        super(SRODecoderRing, self).__init__()

    def forward(self, x: Dict[str, torch.Tensor]):
        outputs = {
            "patient_position": {"HFS": 1.0, "HFP": 0.1},
            "offset_model": {
                "beam": 0.3,
                "anatomy": 0.6,
                "iec_patient": 0.1,
                "iec_tabletop": 0.1,
                "iec_fixed_lagrangian": 0.1,
                "iec_fixed_eulerian": 0.1,
            },
        }
        return outputs


if __name__ == "__main__":
    ring = SRODecoderRing()

    inputs = {
        "reference_isocenter": torch.tensor([0.0, 0.0, 0.0]),
        "localizations": [
            {
                "localization_isocenter": torch.tensor([1.0, 2.0, 3.0]),
                "registration_matrix": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
                ),
                "offset": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            },
            {
                "localization_isocenter": torch.tensor([1.0, 2.0, 3.0]),
                "registration_matrix": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
                ),
                "offset": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            },
        ],
    }
    pprint(inputs)

    matrix_basis_x = [0, 0, 0]
    matrix_basis_y = [0, 0, 0]
    matrix_basis_z = [0, 0, 0]
    matrix_xlate = [0, 0, 0]

    offset_xlate = [0, 0, 0]
    offset_rotate = [0, 0, 0]

    inputs = torch.tensor(
        [
            matrix_basis_x,
            matrix_basis_y,
            matrix_basis_z,
            matrix_xlate,
            offset_xlate,
            offset_rotate,
        ]
    )

    outputs = ring(inputs)
    pprint(outputs)
