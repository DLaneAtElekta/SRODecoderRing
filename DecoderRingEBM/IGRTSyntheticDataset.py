import os
from rich import print
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from DecoderRingEBM.DeepEnergyModel import DeepEnergyModel


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
    quarter_rotations = quarter_rotations * 90  # convert to degrees
    rot = Rotation.from_euler("xyz", quarter_rotations, degrees=True)
    return np.append(patient_shift[:3], rot.apply(patient_shift[3:]))


def iec_room_to_native_shift(native_dirs, iec_room_shift):
    return native_dirs * iec_room_shift


def generate_patient_chart(rng=np.random.default_rng(), num_sessions=3):
    patient_chart = {}

    patient_position = rng.choice(
        ["hfs", "hfp", "ffs", "ffp"], p=[0.9, 0.05, 0.02, 0.03]
    )
    quarter_rotations = patient_position_to_quarter_rotation(patient_position)
    patient_chart["patient position quarter rotations"] = quarter_rotations

    native_dirs = np.array([1, 1, 1, 1, 1, 1])
    patient_chart["native translate directions"] = native_dirs[:3]
    patient_chart["native rotate directions"] = native_dirs[3:]

    init_couch = np.array([0, 0, 0, 0, 0, 0])
    patient_chart["init couch translate"] = init_couch[:3]
    patient_chart["init couch rotate"] = init_couch[3:]

    # TODO: generate protocol parameters

    # tolerance for shift
    apply_shift_tolerance = 0.1  # cm

    # phase 1:
    phase1_duration = (0.12, 0.06)  # mean/stddev of relative fraction
    phase1_p_imaging = 0.9
    phase1_p_apply_shift = 0.9
    phase1_p_apply_systematic_offset = 0.9

    # phase 2:
    phase2_duration = (0.90, 1.0)  # mean/stddev of relative fraction
    phase2_p_imaging = 0.9
    phase2_p_apply_shift = 0.9

    target_match_mean_stddev = rng.standard_normal(6), np.exp(rng.standard_normal(6))
    print(f"{target_match_mean_stddev}")

    for n in range(num_sessions):
        # yield f"session{n} init couch translate", init_couch[:3]
        # yield f"session{n} init couch rotate", init_couch[3:]
        # yield f"session{n} localization offset translate"

        target_match = rng.normal(target_match_mean_stddev)

        rotation_matrix = Rotation.from_euler("xyz", target_match[3:], degrees=True)
        rotation_matrix = rotation_matrix.as_matrix()
        patient_chart[f"session{n} patient match translate"] = target_match[:3]
        patient_chart[f"session{n} patient match x rotate"] = rotation_matrix[0]
        patient_chart[f"session{n} patient match y rotate"] = rotation_matrix[1]
        patient_chart[f"session{n} patient match z rotate"] = rotation_matrix[2]

        iec_room_shift = patient_to_iec_room_shift(quarter_rotations, target_match)
        native_shift = iec_room_to_native_shift(
            native_dirs=native_dirs, iec_room_shift=iec_room_shift
        )
        patient_chart[f"session{n} native shift translate"] = native_shift[:3]
        patient_chart[f"session{n} native shift rotate"] = native_shift[3:]

        dose_tracking = np.array([(n + 1) * 0.03] * 3)
        patient_chart[f"session{n} dose tracking"] = dose_tracking

    return patient_chart


class IGRTSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_patients=10, num_sessions=3):
        self.patient_charts = [
            generate_patient_chart(num_sessions=num_sessions)
            for _ in range(num_patients)
        ]

    def get_tensor_labels(self):
        # TODO: check these are consistent?
        return list(self.patient_charts[0].keys())

    def get_patient_chart(self, n):
        return self.patient_charts[n]

    def __len__(self):
        return len(self.patient_charts)

    def __getitem__(self, idx):
        return np.stack(list(self.patient_charts[idx].values()))


CHECKPOINT_PATH = "./saved_models/igrt_synthetic_dataset"

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


ds = IGRTSyntheticDataset(num_patients=1000)

print(ds.get_patient_chart(0))
print(ds.get_tensor_labels())
# print(ds[0])

train_dl = torch.utils.data.DataLoader(ds, batch_size=28, shuffle=True)
val_dl = torch.utils.data.DataLoader(ds, batch_size=56, shuffle=False)

# Create a PyTorch Lightning trainer with the generation callback
trainer = pl.Trainer(
    default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=60,
    gradient_clip_val=0.1,
    callbacks=[
        ModelCheckpoint(
            save_weights_only=True, mode="min", monitor="val_contrastive_divergence"
        ),
        # GenerateCallback(every_n_epochs=5),
        # SamplerCallback(every_n_epochs=5),
        # OutlierCallback(),
        LearningRateMonitor("epoch"),
    ],
)
# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
if os.path.isfile(pretrained_filename):
    print("Found pretrained model, loading...")
    model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
else:
    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(model, train_dl, val_dl)
    model = DeepEnergyModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

# now serialize the model
