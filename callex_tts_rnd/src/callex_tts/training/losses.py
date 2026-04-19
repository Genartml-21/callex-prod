"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Loss Functions                                        ║
║                                                                      ║
║  All loss components isolated and unit-testable:                     ║
║    • Generator adversarial loss (fool discriminators)                ║
║    • Discriminator loss (distinguish real vs. generated)             ║
║    • Feature matching loss (match intermediate representations)      ║
║    • Mel reconstruction loss (L1 on mel spectrograms)                ║
║    • KL divergence loss (regularize posterior to prior)              ║
║    • Duration loss (from stochastic duration predictor)              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generator_adversarial_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Least-squares GAN generator loss.
    Generator wants discriminator to output 1.0 for fake samples.
    """
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((dg - 1.0) ** 2)
    return loss


def discriminator_adversarial_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_gen_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Least-squares GAN discriminator loss.
    Discriminator wants to output 1.0 for real, 0.0 for fake.
    """
    loss_real = 0
    loss_fake = 0
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        loss_real += torch.mean((dr - 1.0) ** 2)
        loss_fake += torch.mean(dg ** 2)
    return loss_real, loss_fake


def feature_matching_loss(
    fmap_real: list[list[torch.Tensor]],
    fmap_gen: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss — L1 distance between intermediate
    discriminator features for real vs. generated audio.
    
    Stabilizes GAN training by providing a smoother gradient
    signal than the adversarial loss alone.
    """
    loss = 0
    for fr_list, fg_list in zip(fmap_real, fmap_gen):
        for fr, fg in zip(fr_list, fg_list):
            loss += F.l1_loss(fr.detach(), fg)
    return loss * 2.0


def mel_reconstruction_loss(
    mel_real: torch.Tensor, mel_gen: torch.Tensor,
) -> torch.Tensor:
    """L1 loss on mel spectrograms."""
    return F.l1_loss(mel_real, mel_gen) * 45.0


def kl_divergence_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence between posterior q(z|x) and prior p(z|text).
    
    KL[q(z|x) || p(z|text)] = E_q[log q - log p]
    
    Both distributions are diagonal Gaussians, so KL has a
    closed-form expression.
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * (
        (z_p - m_p) ** 2 * torch.exp(-2.0 * logs_p)
        + torch.exp(2.0 * (logs_q - logs_p))
    )
    kl = torch.sum(kl * z_mask, dim=[1, 2])
    return torch.mean(kl)


class GeneratorLoss:
    """
    Combined generator loss for VITS2 training.
    
    Components:
      • Adversarial loss (×3 discriminators: MPD, MSD, MRD)
      • Feature matching loss
      • Mel reconstruction loss
      • KL divergence loss
      • Duration predictor NLL
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_fm: float = 2.0,
        lambda_mel: float = 45.0,
        lambda_kl: float = 1.0,
        lambda_dur: float = 1.0,
    ):
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_dur = lambda_dur

    def __call__(
        self,
        disc_outputs: dict,
        model_outputs: dict,
        mel_real: torch.Tensor,
        mel_gen: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute total generator loss.
        
        Returns:
            (total_loss, loss_dict for logging)
        """
        # Adversarial losses (all discriminators)
        loss_adv = 0
        for key in ["mpd", "msd", "mrd"]:
            if key in disc_outputs:
                _, gen_outputs, _, _ = disc_outputs[key]
                loss_adv += generator_adversarial_loss(gen_outputs)

        # Feature matching
        loss_fm = 0
        for key in ["mpd", "msd", "mrd"]:
            if key in disc_outputs:
                _, _, fmap_r, fmap_g = disc_outputs[key]
                loss_fm += feature_matching_loss(fmap_r, fmap_g)

        # Mel reconstruction
        loss_mel = mel_reconstruction_loss(mel_real, mel_gen)

        # KL divergence
        loss_kl = kl_divergence_loss(
            model_outputs["z_p"],
            model_outputs["logs_q"],
            model_outputs["m_p"],
            model_outputs["logs_p"],
            model_outputs["mel_mask"],
        )

        # Duration
        loss_dur = torch.mean(model_outputs["duration_nll"])

        total = (
            self.lambda_adv * loss_adv +
            self.lambda_fm * loss_fm +
            self.lambda_mel * loss_mel +
            self.lambda_kl * loss_kl +
            self.lambda_dur * loss_dur
        )

        log_dict = {
            "loss/gen_total": total.item(),
            "loss/gen_adv": loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv,
            "loss/gen_fm": loss_fm.item() if isinstance(loss_fm, torch.Tensor) else loss_fm,
            "loss/gen_mel": loss_mel.item(),
            "loss/gen_kl": loss_kl.item(),
            "loss/gen_dur": loss_dur.item(),
        }

        return total, log_dict


class DiscriminatorLoss:
    """Combined discriminator loss across MPD + MSD + MRD."""

    def __call__(
        self, disc_outputs: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = 0
        log_dict = {}

        for key in ["mpd", "msd", "mrd"]:
            if key in disc_outputs:
                real_outputs, gen_outputs, _, _ = disc_outputs[key]
                loss_r, loss_f = discriminator_adversarial_loss(real_outputs, gen_outputs)
                loss = loss_r + loss_f
                total += loss
                log_dict[f"loss/disc_{key}"] = loss.item()

        log_dict["loss/disc_total"] = total.item() if isinstance(total, torch.Tensor) else total
        return total, log_dict
