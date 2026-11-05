# main.py

# Module import
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import Agent
from trainer import Trainer


def _parse_args() -> argparse.Namespace:
    """
    Parse arguments given by user

    Returns:
        Namespace object from argparse
    """

    # Initialize parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Add argument to be parsed
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training based on given checkpoint"
    )

    # Parse arguments and return the result
    return parser.parse_args()

def _parse_json() -> dict:
    """
    Get JSON configurations

    Returns:
        dict consists of the parsed JSON contents
    """

    # Read config relative to this file, regardless of the working directory
    return json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

def main() -> None:
    """
    Main function

    Raises:
        ValueError: When iteration cannot be extracted from checkpoint file name

    Returns:
        None
    """

    # Parse arguments and configurations
    args = _parse_args()
    configs = _parse_json()

    # Initialize agent
    agent = Agent(
        temperature=configs["temperature"],
        move_selection_algo=configs["move_selection_algo"],
        learning_rate=configs["learning_rate"],
        device=configs["device"],
        simulation_count=configs["simulation_count"],
        exploration_strength=configs["exploration_strength"]
    )

    # Resume agent training based on given checkpoint path
    completed = 0
    if args.resume:
        agent.load(args.resume)

        # Checkpoint names use the total completed iteration, e.g.
        # agent_iteration_000100.pt, continue from that point
        try:
            completed = int(args.resume.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Cannot determine completed iterations from checkpoint filename: {args.resume.name}"
            )

    # Initialize trainer
    trainer = Trainer(
        agent,
        Path(configs["stockfish_dir"]) / "stockfish-windows-x86-64-universal.exe",
        device=configs["device"],
        data_capacity=configs["data_capacity"],
        batch_size=configs["batch_size"],
        train_steps=configs["train_steps"],
        games_per_iteration=configs["games_per_iteration"],
        max_move=configs["max_move"],
        gradient_clip=configs["gradient_clip"],
        engine_time_limit=configs["engine_time_limit"]
    )

    # Initialize checkpoint folder
    output = Path(configs["checkpoint_dir"])
    output.mkdir(parents=True, exist_ok=True)

    # Get remaining iterations
    remaining = max(0, configs["train_iterations"] - completed)

    # Loop until every iteration is done
    while remaining:

        # Get each block
        block = min(configs["save_in_every_iter"], remaining)

        # Train model
        losses = trainer.train(iterations=block)

        # Recount remaining iteration
        remaining -= block

        # Get completed iteration
        completed += block

        # Initialize checkpoint path
        checkpoint = output / f"agent_iteration_{completed:06d}.pt"

        # Save agent to checkpoint path
        agent.save(checkpoint)

        # Get valid losses
        valid_losses = [loss for loss in losses if loss is not None]

        # Print valid losses
        if valid_losses:
            means = tuple(sum(x[i] for x in valid_losses) / len(valid_losses) for i in range(3))
            print(f"iteration {completed}: loss={means[0]:.4f}, policy={means[1]:.4f}, value={means[2]:.4f}; saved {checkpoint}")
        else:
            print(f"iteration {completed}: insufficient replay data; saved {checkpoint}")

# Run main method
main()
