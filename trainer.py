# trainer.py

# Module import
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch
from torch.nn import functional as torch_func

from agent import Agent
from environment import Environment


# Dataclass definition
@dataclass
class TrainingData:
    """
    Dataclass that stores data to train model later
    """

    # ----- Public attributes -----

    board_tensor: torch.Tensor
    action_indices: list[int]
    action_probs: list[float]
    value_target: float


class Trainer:
    """
    Agent trainer using stockfish
    """

    # ----- Initialization -----

    def __init__(
        self,
        agent: Agent,
        stockfish_path: str | Path,
        *,
        device: str = "cpu",
        data_capacity: int = 20000,
        batch_size: int = 32,
        train_steps: int = 50,
        games_per_iteration: int = 4,
        max_move: int = 250,
        gradient_clip: float = 5.0,
        engine_time_limit: float = 0.1
    ) -> None:
        """
        Initialize trainer instance

        Args:
            agent: Agent whose network and optimizer will be trained
            stockfish_path: Stockfish executable path or command name

        Keyword Args:
            device: Torch device used for inference and training
            data_capacity: Maximum positions retained for training
            batch_size: Number of data per optimizer step
            train_steps: Update model weights n times with same data
            games_per_iteration: N games to play before train
            max_move: Max count of move for a game, if exceeded will return draw
            gradient_clip: Maximum gradient norm
            engine_time_limit: Stockfish thinking time per move, in seconds

        Returns:
            None
        """

        # Initialize class attributes
        self._agent: Agent = agent
        self._device: torch.device = torch.device(device)
        self._data_buffer: deque[TrainingData] = deque(maxlen=data_capacity)
        self._batch_size: int = batch_size
        self._train_steps: int = train_steps
        self._games_per_iteration: int = games_per_iteration
        self._max_move: int = max_move
        self._gradient_clip: float = gradient_clip
        self._stockfish_path: Path = Path(stockfish_path)
        self._engine_time_limit: float = engine_time_limit
        self._iteration: int = 0

        # Move agent to specified device
        self._agent.to(device)


    # ----- Public attributes -----

    @property
    def data_buffer(
        self
    ) -> int:
        """
        Number of datas currently stored for training
        
        Returns:
            Current data count
        """

        # Return data buffer count
        return len(self._data_buffer)

    @property
    def iteration(
        self
    ) -> int:
        """
        Number of completed training iterations
        
        Returns:
            Current training iterations
        """

        # Return current iteration
        return self._iteration


    # ----- Public methods -----

    def train(
        self,
        *,
        iterations: int = 1
    ) -> list[tuple[float, float, float] | None]:
        """
        Train agent weights

        Keyword Args:
            iterations: Complete training cycle counts

        Returns:
            Returns loss of each iteration or none if there is no training
            has been done in that iteration
        """

        # Initialize losses container
        losses_by_iteration = []

        # Initialize chess engine
        with chess.engine.SimpleEngine.popen_uci(self._stockfish_path) as engine:

            # Loop through each iteration
            for _ in range(iterations):

                # Print separator
                print("\n===============================================\n")

                # Set model to evaluation mode
                self._agent.chess_net.eval()

                # Play games to get data and append to data buffer
                for game_number in range(self._games_per_iteration):
                    datas, result, board_state = self._play_game(engine)
                    self._data_buffer.extend(datas)

                    # Print terminal state
                    if result is None:
                        result_text = f"Unfinished (Move limit: {self._max_move})"
                    elif result == 0.0:
                        result_text = "Draw"
                    else:
                        result_text = f"{"Agent" if result == 1.0 else "Stockfish"} win"
                    print(
                        f"Iteration {self._iteration + 1}, "
                        f"Game {game_number + 1}/{self._games_per_iteration}: "
                        f"{result_text}\n{board_state}\n"
                    )

                # If there is no enough data to train skip current iteration
                if (
                    len(self._data_buffer) < self._batch_size
                    or self._train_steps == 0
                ):
                    losses_by_iteration.append(None)

                # Else if enough
                else:

                    # Print message
                    print("\nUpdating model")

                    # Set model to train mode
                    self._agent.chess_net.train()

                    # Update weights and get losses for current iter
                    step_losses = [self._train_step() for _ in range(self._train_steps)]

                    # Calculate the average of the losses
                    losses_avg = tuple(
                        sum(loss[index] for loss in step_losses) / len(step_losses)
                        for index in range(3)
                    )

                    # Append loss average for current iter
                    losses_by_iteration.append(losses_avg)

                    # Print current iter loss
                    print(f"Current iter loss: {losses_avg}")

                # Increment the iteration
                self._iteration += 1

        # Return iteration losses
        return losses_by_iteration


    # ----- Helpers -----

    def _play_game(
        self,
        engine: chess.engine.SimpleEngine
    ) -> tuple[list[TrainingData], int]:
        """
        Play game to get training data
        
        Args:
            engine: Python chess engine

        Returns:
            List of training data as first index and second 
        """

        # Initialize new environment
        environment = Environment()

        # Randomly assign color
        agent_color = random.choice((chess.WHITE, chess.BLACK))

        # Initialize trajectory container
        trajectory = []

        # Loop through each move
        for _ in range(self._max_move):

            # Break if environment reached terminal state
            if environment.state is not None:
                break

            # Infer with agent if it is agent turn
            if environment.turn == agent_color:
                board_tensor, move_ids, probabilities, _ = self._agent.run(environment)

                # Append to trajectory
                trajectory.append((
                    board_tensor,
                    move_ids,
                    probabilities
                ))

            # Else infer with trainer engine
            else:
                engine_move = engine.play(
                    environment.board,
                    chess.engine.Limit(time=self._engine_time_limit)
                ).move
                environment.move(self._move_to_id(engine_move))

        # Parse trajectory into training data
        datas = [
            TrainingData(
                state,
                indices,
                probabilities,
                environment.state
            )
            for state, indices, probabilities in trajectory
        ]

        # Return training data and terminal state
        return (datas, environment.state, environment.board)

    def _move_to_id(
        self,
        move: chess.Move
    ) -> int:
        """
        Encode a python-chess move to respective ID

        Args:
            move: Python chess move instance

        Returns:
            ID of the move
        """

        # Initialize promotion IDs
        promotion_ids = {
            None: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4
        }

        # Calculate and return the ID
        return (move.from_square * 64 + move.to_square) * 5 + promotion_ids[move.promotion]

    def _train_step(
        self
    ) -> tuple[float, float, float]:
        """
        Update model weights

        Returns:
            Tuple with value as total loss as first index, policy head loss as
            second index and value head loss as third index
        """

        # Get a batch of data
        batch = random.sample(
            list(self._data_buffer),
            k=self._batch_size
        )

        # Get states of each data
        states = torch.stack(
            [data.board_tensor for data in batch]
        ).to(self._device)

        # Get value targets of each data
        value_targets = torch.tensor(
            [data.value_target for data in batch],
            dtype=torch.float32,
            device=self._device
        )

        # Let model infer policy logits and values based on
        # states
        policy_logits, values = self._agent.chess_net(states)

        # Get the log softmax of the prediction
        log_probs = torch_func.log_softmax(policy_logits, dim=1)

        # Initialize policy losses container
        policy_losses = []

        # Enumerate through each data
        for index, data in enumerate(batch):

            # Get action indices
            action_indices = torch.tensor(
                data.action_indices,
                dtype=torch.long,
                device=self._device
            )

            # Get action probs
            action_probs = torch.tensor(
                data.action_probs,
                dtype=torch.float32,
                device=self._device
            )

            # Calculate cross entropy loss for current data
            policy_losses.append(
                -(action_probs * log_probs[index, action_indices]).sum()
            )

        # Calculate mean of losses of each data
        policy_loss = torch.stack(policy_losses).mean()

        # Calculate value loss with mean squared error
        value_loss = torch_func.mse_loss(values, value_targets)

        # Add both loss
        loss = policy_loss + value_loss

        # Set parameters of optimizer
        self._agent.optimizer.zero_grad(set_to_none=True)

        # Calculate gradients of each parameters
        loss.backward()

        # Clip the gradient so it does not get too large
        torch.nn.utils.clip_grad_norm_(
            self._agent.chess_net.parameters(),
            self._gradient_clip
        )

        # Update neural network parameters
        self._agent.optimizer.step()

        # Return loss of each respective domain
        return (
            float(loss.item()),
            float(policy_loss.item()),
            float(value_loss.item())
        )

