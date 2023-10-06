# Path to the folder where the pretrained models are saved
import torch
import torch.optim as optim
import pytorch_lightning as pl

from DecoderRingEBM.MCMCSampler import MCMCSampler


class Swish(torch.nn.Module):
    """basic Swish activation model"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class MLPModel(torch.nn.Module):
    def __init__(self):
        """multi-layered perceptron for computing energy from an IGRT patient chart"""
        super(MLPModel, self).__init__()

        # Series of convolutions and Swish activation functions
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 6),  # [3x26]
            Swish(),
            # torch.nn.AvgPool1d((2,1)),
            torch.nn.Linear(6, 12),
            Swish(),
            # torch.nn.AvgPool1d((2,1)),
            torch.nn.Linear(12, 24),
            Swish(),
            # torch.nn.AvgPool1d((2,1)),
            torch.nn.Linear(24, 6),
            Swish(),
            # torch.nn.AvgPool1d((2,1)),
            torch.nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.map_layer = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        self.conv1 = torch.nn.Conv2d(
            1, 16, kernel_size=(3, 3), stride=(1, 1), padding="same"
        )
        self.max = torch.nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )
        self.swish = Swish()
        self.conv2 = torch.nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding="same"
        )

        self.flatten = torch.nn.Flatten()
        self.map_final = torch.nn.Linear(2 * 2 * 32, 1)

    def forward(self, x):
        x = self.map_layer(x)
        x = self.conv1(x)
        x = self.max(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = self.max(x)
        x = self.swish(x)
        x = self.flatten(x)
        x = self.map_final(x)
        return x


class DeepEnergyModel(pl.LightningModule):
    def __init__(self, rng, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0):
        super(DeepEnergyModel, self).__init__()
        self.save_hyperparameters()

        self.mlp = MLPModel()
        self.cnn = CNNModel()
        self.sampler = MCMCSampler(
            self.cnn, rng=rng, img_shape=img_shape, sample_size=batch_size
        )
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
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
        real_imgs = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("val_contrastive_divergence", cdiv)
        self.log("val_fake_out", fake_out.mean())
        self.log("val_real_out", real_out.mean())
