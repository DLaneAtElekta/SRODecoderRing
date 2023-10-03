import os
from rich import print
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/igrt_synthetic_dataset"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


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
    patient_chart["patient position quater rotations"] = quarter_rotations

    native_dirs = np.array([1, 1, 1, 1, 1, 1])
    patient_chart["native translate directions"] = native_dirs[:3]
    patient_chart["native rotate directions"] = native_dirs[3:]

    init_couch = np.array([0, 0, 0, 0, 0, 0])
    patient_chart["init couch translate"] = init_couch[:3]
    patient_chart["init couch rotate"] = init_couch[3:]

    systematic_offset = rng.standard_normal(6)
    std_dev = rng.standard_normal(6)
    std_dev = np.exp(std_dev)
    print(f"{systematic_offset}")
    print(f"{std_dev}")

    for n in range(num_sessions):
        # yield f"session{n} init couch translate", init_couch[:3]
        # yield f"session{n} init couch rotate", init_couch[3:]
        # yield f"session{n} localization offset translate"

        patient_match = rng.normal(systematic_offset, std_dev)

        rotation_matrix = Rotation.from_euler("xyz", patient_match[3:], degrees=True)
        rotation_matrix = rotation_matrix.as_matrix()
        patient_chart[f"session{n} patient match translate"] = patient_match[:3]
        patient_chart[f"session{n} patient match x rotate"] = rotation_matrix[0]
        patient_chart[f"session{n} patient match y rotate"] = rotation_matrix[1]
        patient_chart[f"session{n} patient match z rotate"] = rotation_matrix[2]

        iec_room_shift = patient_to_iec_room_shift(quarter_rotations, patient_match)
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


class Sampler:
    def __init__(self, model, rng, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.rng = rng
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [
            (torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)
        ]

    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(
            self.rng.choices(self.examples, k=self.sample_size - n_new), dim=0
        )
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = (
            list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.examples
        )
        self.examples = self.examples[: self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(
        model, inp_imgs, steps=60, step_size=10, return_img_per_step=False
    ):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(
                -0.03, 0.03
            )  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        # Series of convolutions and Swish activation functions
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 6),  # [3x26]
            Swish(),
            torch.nn.AvgPool1d(3, stride=2),
            torch.nn.Linear(6, 12),
            Swish(),
            torch.nn.AvgPool1d(3, stride=2),
            torch.nn.Linear(12, 24),
            Swish(),
            torch.nn.AvgPool1d(3, stride=2),
            torch.nn.Linear(24, 6),
            Swish(),
            torch.nn.AvgPool1d(3, stride=2),
            torch.nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.layers(x)


class DeepEnergyModel(pl.LightningModule):
    def __init__(
        self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **MLP_args
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mlp = MLPModel(**MLP_args)
        self.sampler = Sampler(self.mlp, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.mlp(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999)
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.97
        )  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.mlp(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log("loss", loss)
        self.log("loss_regularization", reg_loss)
        self.log("loss_contrastive_divergence", cdiv_loss)
        self.log("metrics_avg_real", real_out.mean())
        self.log("metrics_avg_fake", fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.mlp(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("val_contrastive_divergence", cdiv)
        self.log("val_fake_out", fake_out.mean())
        self.log("val_real_out", real_out.mean())


def train_model(**kwargs):
    ds = IGRTSyntheticDataset(num_patients=1000)

    print(ds.get_patient_chart(0))
    print(ds.get_tensor_labels())
    # print(ds[0])

    train_dl = torch.utils.data.DataLoader(ds, batch_size=28, shuffle=True)
    # for batch_data in train_dl:
    #     print(batch_data)

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
            # LearningRateMonitor("epoch"),
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
    # No testing as we are more interested in other properties
    return model
