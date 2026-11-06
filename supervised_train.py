"""Pretrain Gambit's policy network on Stockfish self-play games.

Run with ``python supervised_train.py``. The resulting checkpoint can be
loaded by the normal training command with ``python main.py --train --resume
checkpoints/supervised_pretrained.pt``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import chess.engine
import torch
from torch.nn import functional as F

from agent import Agent
from environment import Environment


ROOT = Path(__file__).resolve().parent


def move_to_id(move: chess.Move) -> int:
    promotion_id = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }[move.promotion]
    return (move.from_square * 64 + move.to_square) * 5 + promotion_id


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    games = int(config.get("supervised_games", 1000))
    epochs = int(config.get("supervised_epochs", 5))
    batch_size = int(config.get("supervised_batch_size", 128))
    opening_plies = int(config.get("supervised_random_opening_plies", 12))
    stockfish_elo = int(config.get("stockfish_elo", 1400))
    think_time = float(config.get("engine_time_limit", 0.1))
    device = config.get("device", "cpu")
    engine_path = ROOT / config.get("stockfish_dir", "stockfish") / "src" / "stockfish.exe"

    if games < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("supervised_games, supervised_epochs, and supervised_batch_size must be positive")

    agent = Agent(
        temperature=config.get("temperature", 1.0),
        move_selection_algo=config.get("move_selection_algo", "sample"),
        learning_rate=config.get("learning_rate", 1e-3),
        device=device,
        simulation_count=config.get("simulation_count", 64),
        exploration_strength=config.get("exploration_strength", 1.5),
    )

    # Each recorded sample is (encoded board, Stockfish move ID).
    samples: list[tuple[torch.Tensor, int]] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        for game_no in range(games):
            board = chess.Board()
            # Random legal openings diversify positions while retaining a
            # Stockfish-selected target for every recorded training position.
            for _ in range(random.randint(0, max(0, opening_plies))):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))

            while not board.is_game_over(claim_draw=True):
                env = Environment()
                env.board.set_fen(board.fen())
                result = engine.play(board, chess.engine.Limit(time=think_time))
                if result.move is None:
                    break
                samples.append((agent._encode(env).cpu(), move_to_id(result.move)))
                board.push(result.move)

            if (game_no + 1) % 25 == 0 or game_no + 1 == games:
                print(f"Generated {game_no + 1}/{games} games; {len(samples)} positions")
    finally:
        engine.quit()

    if not samples:
        raise RuntimeError("Stockfish generated no training positions")

    agent.chess_net.train()
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            states = torch.stack([sample[0] for sample in batch]).to(device)
            targets = torch.tensor([sample[1] for sample in batch], dtype=torch.long, device=device)
            logits, _ = agent.chess_net(states)
            loss = F.cross_entropy(logits, targets)
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.chess_net.parameters(), config.get("gradient_clip", 5.0))
            agent.optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        print(f"Epoch {epoch + 1}/{epochs}: policy loss {total_loss / batches:.4f}")

    output = ROOT / config.get("checkpoint_dir", "checkpoints") / "supervised_pretrained.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)
    print(f"Saved supervised checkpoint: {output}")


if __name__ == "__main__":
    main()
