"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Checkpoint Manager & Model Registry                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("callex.tts.training.checkpoint")


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""
    epoch: int
    global_step: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    model_version: str
    timestamp: str
    config_hash: str = ""


class CheckpointManager:
    """
    Production checkpoint manager with model registry.
    
    Features:
      • Save/load generator + discriminator + optimizers + schedulers
      • Automatic best-model tracking (lowest validation loss)
      • Keep-N retention policy (disk management)
      • Metadata JSON alongside each checkpoint
      • Atomic writes (write to temp, then rename — prevents corruption)
    
    Usage:
        mgr = CheckpointManager("checkpoints/", keep_last_n=5, save_best=True)
        
        # Save
        mgr.save(epoch=100, step=50000, model=gen, disc=disc,
                 opt_g=opt_g, opt_d=opt_d, scheduler_g=sched_g,
                 train_loss=0.5, val_loss=0.3)
        
        # Load
        state = mgr.load_latest()
        model.load_state_dict(state["generator"])
    """

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 5,
        save_best: bool = True,
        model_version: str = "2.0",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.model_version = model_version
        self.best_val_loss = float("inf")

        # Load existing best val loss if available
        best_meta = self.save_dir / "best" / "metadata.json"
        if best_meta.exists():
            data = json.loads(best_meta.read_text())
            self.best_val_loss = data.get("val_loss", float("inf")) or float("inf")

    def save(
        self,
        epoch: int,
        global_step: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module = None,
        optimizer_g: torch.optim.Optimizer = None,
        optimizer_d: torch.optim.Optimizer = None,
        scheduler_g=None,
        scheduler_d=None,
        train_loss: float = 0.0,
        val_loss: float = None,
        lr: float = 0.0,
    ) -> Path:
        """Save checkpoint with metadata."""
        checkpoint_name = f"checkpoint_e{epoch:05d}_s{global_step:08d}"
        checkpoint_dir = self.save_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build state dict
        state = {
            "generator": generator.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if discriminator is not None:
            state["discriminator"] = discriminator.state_dict()
        if optimizer_g is not None:
            state["optimizer_g"] = optimizer_g.state_dict()
        if optimizer_d is not None:
            state["optimizer_d"] = optimizer_d.state_dict()
        if scheduler_g is not None:
            state["scheduler_g"] = scheduler_g.state_dict()
        if scheduler_d is not None:
            state["scheduler_d"] = scheduler_d.state_dict()

        # Atomic save: write to temp file, then rename
        ckpt_path = checkpoint_dir / "model.pt"
        tmp_path = checkpoint_dir / "model.pt.tmp"
        torch.save(state, str(tmp_path))
        tmp_path.rename(ckpt_path)

        # Save metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=lr,
            model_version=self.model_version,
            timestamp=datetime.utcnow().isoformat(),
        )
        meta_path = checkpoint_dir / "metadata.json"
        meta_path.write_text(json.dumps(asdict(metadata), indent=2))

        logger.info(
            "Checkpoint saved: %s (epoch=%d, step=%d, val_loss=%s)",
            checkpoint_name, epoch, global_step,
            f"{val_loss:.4f}" if val_loss is not None else "N/A"
        )

        # Update symlink to latest
        latest_link = self.save_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_dir.name)

        # Best model tracking
        if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_dir = self.save_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            logger.info("🏆 New best model! val_loss=%.4f", val_loss)

        # Retention policy — keep only last N checkpoints
        self._enforce_retention()

        return ckpt_path

    def load_latest(self) -> dict:
        """Load most recent checkpoint."""
        latest = self.save_dir / "latest"
        if latest.is_symlink():
            return self._load_dir(latest.resolve())
        
        # Fallback: find newest by name
        dirs = sorted(
            [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda d: d.name,
        )
        if dirs:
            return self._load_dir(dirs[-1])
        
        raise FileNotFoundError(f"No checkpoints found in {self.save_dir}")

    def load_best(self) -> dict:
        """Load best-validation-loss checkpoint."""
        best_dir = self.save_dir / "best"
        if best_dir.exists():
            return self._load_dir(best_dir)
        raise FileNotFoundError("No best checkpoint found")

    def load_path(self, path: str) -> dict:
        """Load specific checkpoint by path."""
        path = Path(path)
        if path.is_dir():
            return self._load_dir(path)
        return torch.load(str(path), map_location="cpu", weights_only=False)

    def _load_dir(self, checkpoint_dir: Path) -> dict:
        model_path = checkpoint_dir / "model.pt"
        state = torch.load(str(model_path), map_location="cpu", weights_only=False)
        
        meta_path = checkpoint_dir / "metadata.json"
        if meta_path.exists():
            state["metadata"] = json.loads(meta_path.read_text())
        
        logger.info("Checkpoint loaded: %s", checkpoint_dir.name)
        return state

    def _enforce_retention(self):
        """Delete old checkpoints beyond retention limit."""
        dirs = sorted(
            [d for d in self.save_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda d: d.name,
        )
        while len(dirs) > self.keep_last_n:
            old = dirs.pop(0)
            shutil.rmtree(old)
            logger.debug("Pruned old checkpoint: %s", old.name)
