"""Pretrain Gambit's policy network on Stockfish self-play games.

Run with ``python supervised_train.py``. The resulting checkpoint can be
loaded by the normal training command with ``python main.py --train --resume
checkpoints/supervised_pretrained.pt``.
"""

from __future__ import annotations

import json
import random
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Resume from a numbered agent checkpoint")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    games = int(config.get("supervised_games", 1000))
    iterations = int(config.get("supervised_iterations", config.get("supervised_epochs", 100)))
    save_every = int(config.get("save_in_every_iter", 10))
    batch_size = int(config.get("supervised_batch_size", 128))
    opening_plies = int(config.get("supervised_random_opening_plies", 12))
    stockfish_elo = int(config.get("stockfish_elo", 1400))
    think_time = float(config.get("engine_time_limit", 0.1))
    device = config.get("device", "cpu")
    engine_path = ROOT / config.get("stockfish_dir", "stockfish") / "src" / "stockfish.exe"

    if games < 1 or iterations < 1 or batch_size < 1 or save_every < 1:
        raise ValueError("supervised_games, supervised_iterations, supervised_batch_size, and save_in_every_iter must be positive")

    agent = Agent(
        temperature=config.get("temperature", 1.0),
        move_selection_algo=config.get("move_selection_algo", "sample"),
        learning_rate=config.get("learning_rate", 1e-3),
        device=device,
        simulation_count=config.get("simulation_count", 64),
        exploration_strength=config.get("exploration_strength", 1.5),
    )
    completed = 0
    if args.resume:
        agent.load(args.resume)
        try:
            completed = int(args.resume.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot determine completed iterations from checkpoint filename: {args.resume.name}") from exc
        print(f"Resuming from {completed} iteration")

    # Each sample stores board, chosen move, and final outcome from the
    # perspective of the side to move in that position.
    samples: list[tuple[torch.Tensor, int, float]] = []
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

            game_samples: list[tuple[torch.Tensor, int, chess.Color]] = []
            while not board.is_game_over(claim_draw=True):
                env = Environment()
                env.board.set_fen(board.fen())
                result = engine.play(board, chess.engine.Limit(time=think_time))
                if result.move is None:
                    break
                game_samples.append((agent._encode(env).cpu(), move_to_id(result.move), board.turn))
                board.push(result.move)

            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                result_text = f"Unfinished (move limit: {board.fullmove_number})"
                value_by_color = {chess.WHITE: 0.0, chess.BLACK: 0.0}
            elif outcome.winner is None:
                result_text = f"Draw ({outcome.termination.name.lower().replace('_', ' ')})"
                value_by_color = {chess.WHITE: 0.0, chess.BLACK: 0.0}
            else:
                winner = "White" if outcome.winner else "Black"
                result_text = f"{winner} win ({outcome.termination.name.lower().replace('_', ' ')})"
                value_by_color = {
                    chess.WHITE: 1.0 if outcome.winner == chess.WHITE else -1.0,
                    chess.BLACK: 1.0 if outcome.winner == chess.BLACK else -1.0,
                }
            samples.extend((state, move_id, value_by_color[color]) for state, move_id, color in game_samples)
            print(f"Game {game_no + 1}/{games}: {result_text}; {len(game_samples)} positions\n{board}\n")
    finally:
        engine.quit()

    if not samples:
        raise RuntimeError("Stockfish generated no training positions")

    agent.chess_net.train()
    output_dir = ROOT / config.get("checkpoint_dir", "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = max(0, iterations - completed)
    while remaining:
        block = min(save_every, remaining)
        for _ in range(block):
            random.shuffle(samples)
            total_loss = 0.0
            policy_total = 0.0
            value_total = 0.0
            batches = 0
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                states = torch.stack([sample[0] for sample in batch]).to(device)
                targets = torch.tensor([sample[1] for sample in batch], dtype=torch.long, device=device)
                value_targets = torch.tensor([sample[2] for sample in batch], dtype=torch.float32, device=device)
                logits, values = agent.chess_net(states)
                policy_loss = F.cross_entropy(logits, targets)
                value_loss = F.mse_loss(values.reshape(-1), value_targets)
                loss = policy_loss + value_loss
                agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.chess_net.parameters(), config.get("gradient_clip", 5.0))
                agent.optimizer.step()
                total_loss += float(loss.item())
                policy_total += float(policy_loss.item())
                value_total += float(value_loss.item())
                batches += 1
            completed += 1
            print(
                f"Iteration {completed}/{iterations}: total loss {total_loss / batches:.4f}, "
                f"policy loss {policy_total / batches:.4f}, value loss {value_total / batches:.4f}"
            )
        remaining -= block
        agent.save(output_dir / f"agent_iteration_{completed:06d}.pt")
        print(f"Saved checkpoint at iteration {completed}")


if __name__ == "__main__":
    main()
