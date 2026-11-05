# agent.py

# Module import
import random
from pathlib import Path
from typing import Literal

import torch

from environment import Environment
from model import MCTS, ChessNet, Encoder, MCTSNode

# Type hints
MoveSelectionAlgo = Literal["max", "sample"]


class Agent:
    """
    Chess agent class
    """

    # ----- Initialization -----

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        move_selection_algo: MoveSelectionAlgo = "sample",
        learning_rate: float = 1e-3,
        device: str = "cpu",
        simulation_count: int = 64,
        exploration_strength: float = 1.5
    ) -> None:
        """
        Initialize agent

        Keyword Args:
            temperature: Encourages less confident move
            move_selection_algo: Select a move based on given algorithm
            learning_rate: Learning rate of the agent
            device: Device to run the agent
            simulation_count: MCTS simulation count
            exploration_strength: Encourages model to explore new moves if bigger
            
        Returns:
            None
        """

        # Initialize instance attributes
        self._temperature: float = temperature
        self._move_selection_algo: MoveSelectionAlgo = move_selection_algo
        self._learning_rate: float = learning_rate
        self._device: torch.device = torch.device(device)
        self._simulation_count: int = simulation_count
        self._exploration_strength: float = exploration_strength

        # Initialize model
        self._chess_net: ChessNet = ChessNet(
            Encoder.output_channels_size
        ).to(self._device)
        self._optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self._chess_net.parameters(),
            lr=self._learning_rate
        )
        self._mcts: MCTS = MCTS(
            self._evaluate_position,
            simulation_count=simulation_count,
            exploration_strength=exploration_strength
        )


    # ----- Public attributes -----

    @property
    def chess_net(
        self
    ) -> ChessNet:
        """
        Return the trainable neural network

        Returns:
            ChessNet class
        """

        # Return chessnet model instance
        return self._chess_net

    @property
    def optimizer(
        self
    ) -> torch.optim.AdamW:
        """
        Return the optimizer used to train the network
        
        Returns:
            Optimizer instance
        """

        # Return optimizer instance
        return self._optimizer


    # ----- Public methods -----

    def run(
        self,
        environment: Environment
    ) -> tuple[torch.Tensor, list[int], list[float], int]:
        """
        React to given environment

        Args:
            environment: Current environment

        Returns:
            Encoded position, legal move IDs, move probabilities, and move ID
        """

        # Agent coordinates encoding, inference, and the MCTS search.
        board_tensor = self._encode(environment)
        root_node = self._search(environment)

        # Calculate next move probabilities
        move_ids, move_probs = self._calculate_move_probs(root_node)

        # Choose a move based on move selection algorithm
        move_id = self._choose_move(move_ids, move_probs)

        # Execute the move
        environment.move(move_id)

        # Return move trajectory
        return (board_tensor, move_ids, move_probs, move_id)

    def to(
        self,
        device: str
    ) -> None:
        """
        Move the agent to a device
        
        Args:
            device: Device name

        Returns:
            None
        """

        # Convert into torch device
        self._device = torch.device(device)

        # Move chess net to given device
        self._chess_net.to(self._device)

        # Move optimizer to given device
        for state in self._optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self._device)

    def load(
        self,
        file_path: str | Path
    ) -> None:
        """
        Load agent parameters from given path

        Args:
            file_path: Path to the model weights

        Returns:
            None
        """

        # Create pathlib instance
        file_path = Path(file_path)

        # Load checkpoint
        checkpoint = torch.load(file_path, map_location="cpu")
        
        # Load chess net weights
        self._chess_net.load_state_dict(checkpoint["chess_net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._temperature = checkpoint["temperature"]
        self._move_selection_algo = checkpoint["move_selection_algo"]
        self._learning_rate = checkpoint["learning_rate"]

    def save(
        self,
        file_path: str | Path
    ) -> None:
        """
        Save agent parameters to given path

        Args:
            file_path: Path to the model weights

        Returns:
            None
        """

        # Create pathlib instance
        file_path = Path(file_path)

        # Save agent weights to given file path
        torch.save(
            {
                "chess_net": self._chess_net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "temperature": self._temperature,
                "move_selection_algo": self._move_selection_algo,
                "learning_rate": self._learning_rate
            },
            file_path
        )


    # ----- Helpers -----

    def _encode(
        self,
        environment: Environment
    ) -> torch.Tensor:
        """
        Encode an environment for network inference

        Args:
            environment: Environment instance

        Returns:
            Encoded chess board
        """

        # Return encoded chess board
        return Encoder.encode_board(environment.board)

    @torch.no_grad()
    def _search(
        self,
        environment: Environment
    ) -> MCTSNode:
        """
        Search a position and evaluating nodes through this agent's network

        Args:
            environment: Environment instance

        Returns:
            Root MCTS node
        """

        # Temporarily store training state and set to eval mode
        was_training = self._chess_net.training
        self._chess_net.eval()

        # Run MCTS and return the root node
        try:
            return self._mcts.run(environment)

        # Change back model mode to training back
        finally:
            if was_training:
                self._chess_net.train()

    @torch.no_grad()
    def _evaluate_position(
        self,
        environment: Environment
    ) -> tuple[torch.Tensor, float]:
        """
        Encode and evaluate position for MCTS

        Args:
            environment: Environment instance

        Returns:
            Tuple with first index as policy logits score and second index
            as value score
        """

        # Encode board into tensor
        board_tensor = self._encode(environment).unsqueeze(0)

        # Move board tensor to intended device
        board_tensor = board_tensor.to(self._device)

        # Input into chess net and get scores
        policy_logits, value = self._chess_net(board_tensor)

        # Return value
        return (policy_logits[0], float(value.item()))

    def _calculate_move_probs(
        self,
        root_node: MCTSNode
    ) -> tuple[list[int], list[float]]:
        """
        Calculate next move probabilities

        Args:
            root_node: Root node of the MCTS output

        Returns:
            Tuple with first index as move ID, and second index
            as probability of each move
        """

        # Get children moves
        moves = list(root_node.children.keys())

        # Get the visit count of each children
        counts = [root_node.children[m].visit_count for m in moves]

        # If there are no possible moves return empty
        if not moves:
            return ([], [])

        # If temperature is near 0 set best move probability as biggest
        if self._temperature <= 1e-6:
            best_move_index = max(range(len(moves)), key=lambda i: counts[i])
            probs = [0.0] * len(moves)
            probs[best_move_index] = 1.0

        # Else calculate move probability based on temperature
        else:
            adjusted = [count ** (1.0 / self._temperature) for count in counts]
            total = sum(adjusted)

            # Equally distribute probabilities if no node has been
            # visited by MCTS
            if total <= 0:
                probs = [1.0 / len(moves)] * len(moves)
            else:
                probs = [v / total for v in adjusted]

        # Return move IDs and their probabilities
        return [move for move in moves], probs

    def _choose_move(
        self,
        move_ids: list[int],
        move_probs: list[float]
    ) -> int:
        """
        Choose next move

        Args:
            move_ids: List of move ID
            move_probs: List of probability of move

        Raises:
            ValueError: When given move id is empty or 
            length of move ids and probs is not equal

        Returns:
            Move ID to run
        """

        # Validate move IDs length
        if not move_ids:
            raise ValueError("No legal moves available to choose")

        # Validate both move IDs and probabilities
        if len(move_ids) != len(move_probs):
            raise ValueError(
                "Moves and probabilities must have the same length"
            )

        # Select a move based on setted algorithm
        if self._move_selection_algo == "sample":
            return random.choices(
                move_ids,
                weights=move_probs,
                k=1
            )[0]
        elif self._move_selection_algo == "max":
            best_move_index = max(
                range(len(move_probs)),
                key=lambda i: move_probs[i],
            )
            return move_ids[best_move_index]

