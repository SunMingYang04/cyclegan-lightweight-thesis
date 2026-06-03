from pathlib import Path

import torch


def save_checkpoint(path, trainer, epoch, config):
    """Save full training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "config": config,
        "G_A": trainer.G_A.state_dict(),
        "G_B": trainer.G_B.state_dict(),
        "D_A": trainer.D_A.state_dict(),
        "D_B": trainer.D_B.state_dict(),
        "optimizer_G": trainer.optimizer_G.state_dict(),
        "optimizer_D": trainer.optimizer_D.state_dict(),
        "scheduler_G": trainer.scheduler_G.state_dict(),
        "scheduler_D": trainer.scheduler_D.state_dict(),
    }
    if trainer.scaler is not None:
        state["scaler"] = trainer.scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, trainer, device):
    """Load full training state and return stored epoch."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    trainer.G_A.load_state_dict(state["G_A"])
    trainer.G_B.load_state_dict(state["G_B"])
    trainer.D_A.load_state_dict(state["D_A"])
    trainer.D_B.load_state_dict(state["D_B"])
    trainer.optimizer_G.load_state_dict(state["optimizer_G"])
    trainer.optimizer_D.load_state_dict(state["optimizer_D"])
    trainer.scheduler_G.load_state_dict(state["scheduler_G"])
    trainer.scheduler_D.load_state_dict(state["scheduler_D"])
    if trainer.scaler is not None and "scaler" in state:
        trainer.scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", 0))


def save_generator_weights(checkpoint_dir, trainer):
    """Save latest generators for test.py."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.G_A.state_dict(), checkpoint_dir / "latest_G_A.pth")
    torch.save(trainer.G_B.state_dict(), checkpoint_dir / "latest_G_B.pth")

