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

    # generate protocol parameters

    # tolerance for shift
    apply_shift_tolerance = np.array([rng.choice([0.1, 0.2, 0.3])])  # cm
    patient_chart["apply_shift_tolerance"] = apply_shift_tolerance

    # period parameters
    patient_chart["period1_duration"] = np.array([rng.normal(0.12, 0.06)])
    patient_chart["period1_p_imaging"] = np.array(
        [np.clip(rng.normal(0.9, 0.1), 0.0, 1.0)]
    )
    patient_chart["period1_p_apply_shift"] = np.array(
        [np.clip(rng.normal(0.9, 0.1), 0.0, 1.0)]
    )
    patient_chart["period1_p_apply_systematic_offset"] = np.array(
        [np.clip(rng.normal(0.9, 0.1), 0.0, 1.0)]
    )

    patient_chart["period2_duration"] = np.array([1.0])  # extends past the end
    patient_chart["period2_p_imaging"] = np.array(
        [np.clip(rng.normal(0.9, 0.1), 0.0, 1.0)]
    )
    patient_chart["period2_p_apply_shift"] = np.array(
        [np.clip(rng.normal(0.9, 0.1), 0.0, 1.0)]
    )

    target_match_mean_stddev = rng.standard_normal(6), np.exp(rng.standard_normal(6))
    print(f"{target_match_mean_stddev}")

    total_sessions = 30
    at_period = 1
    for n in range(num_sessions):
        fraction_session = n / total_sessions
        while patient_chart.get(f"period{at_period}_duration") < fraction_session:
            at_period += 1

        if rng.uniform() < patient_chart.get(f"period{at_period}_p_imaging"):
            target_match = rng.normal(
                target_match_mean_stddev[0], target_match_mean_stddev[1]
            )
        else:
            target_match = np.array([0, 0, 0, 0, 0, 0])

        patient_chart[f"session{n} patient match translate"] = target_match[:3]

        rotation_matrix = Rotation.from_euler("xyz", target_match[3:], degrees=True)
        rotation_matrix = rotation_matrix.as_matrix()
        patient_chart[f"session{n} patient match x rotate"] = rotation_matrix[0]
        patient_chart[f"session{n} patient match y rotate"] = rotation_matrix[1]
        patient_chart[f"session{n} patient match z rotate"] = rotation_matrix[2]

        if target_match[
            :3
        ].max() > apply_shift_tolerance and rng.uniform() < patient_chart.get(
            f"period{at_period}_p_apply_shift"
        ):
            table_shift = target_match
        else:
            table_shift = np.array([0, 0, 0, 0, 0, 0])

        patient_chart[f"session{n} table shift translate"] = table_shift[:3]
        patient_chart[f"session{n} table shift rotate"] = table_shift[3:]

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
        item = self.patient_charts[idx].values()
        item = list(item)
        item = np.concatenate(item)
        item = np.expand_dims(item, -1)
        return item


CHECKPOINT_PATH = "./saved_models/igrt_synthetic_dataset"

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


ds = IGRTSyntheticDataset(num_patients=1000)

print(ds.get_patient_chart(0))
print(ds.get_tensor_labels())
print(ds[0])

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
    model = DeepEnergyModel(np.random.default_rng(), (71, 1), train_dl.batch_size)
    trainer.fit(model, train_dl, val_dl)
    model = DeepEnergyModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

# now serialize the model
