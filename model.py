# model.py

# Module import
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import torch
from torch import nn
from torch.nn import functional as torch_func

from .environment import Environment


# Dataclass definition
@dataclass
class MCTSNode:
    """
    Traverse node for MCTS model
    """

    # ----- Public attributes -----

    environment: Environment
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def node_value(
        self
    ) -> float:
        """
        Calculate node value

        Returns:
            Node value
        """

        # Return node value
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class Encoder:
    """
    Encoder class
    """

    # ----- Public attributes -----

    output_channels_size: int = 18


    # ----- Public methods -----

    @classmethod
    def encode_board(
        cls,
        board: chess.Board
    ) -> torch.Tensor:
        """
        Encode given board

        Args:
            board: Board instance

        Returns:
            Encoded board data
        """

        # Initialize torch container
        x = torch.zeros(18, 8, 8, dtype=torch.float32)

        # Initialize piece to channel map
        piece_to_channel = {
            (chess.WHITE, chess.PAWN): 0,
            (chess.WHITE, chess.KNIGHT): 1,
            (chess.WHITE, chess.BISHOP): 2,
            (chess.WHITE, chess.ROOK): 3,
            (chess.WHITE, chess.QUEEN): 4,
            (chess.WHITE, chess.KING): 5,
            (chess.BLACK, chess.PAWN): 6,
            (chess.BLACK, chess.KNIGHT): 7,
            (chess.BLACK, chess.BISHOP): 8,
            (chess.BLACK, chess.ROOK): 9,
            (chess.BLACK, chess.QUEEN): 10,
            (chess.BLACK, chess.KING): 11
        }

        # Traverse through each pieces and mark its location based on channel
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            x[
                piece_to_channel[(piece.color, piece.piece_type)], 
                row,
                col
            ] = 1.0

        # Set other channels as other informations
        x[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
        x[13].fill_(float(board.has_kingside_castling_rights(chess.WHITE)))
        x[14].fill_(float(board.has_queenside_castling_rights(chess.WHITE)))
        x[15].fill_(float(board.has_kingside_castling_rights(chess.BLACK)))
        x[16].fill_(float(board.has_queenside_castling_rights(chess.BLACK)))

        # 17th channel as en passant move
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            x[17, row, col] = 1.0

        # Return encoded board
        return x


class ResidualBlock(nn.Module):
    """
    Residual block class
    """

    # ----- Initialization -----

    def __init__(
        self,
        channels: int
    ) -> None:
        """
        Initialize block

        Args:
            channels: Feature map sizes

        Returns:
            None

        Notes:
            - Input channels and output channels for current block
              are the same size
        """

        # Call base class init
        super().__init__()

        # Initialize layers
        self._layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )


    # ----- Public methods -----

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward data

        Args:
            x: Data to input
        
        Returns:
            Output from residual block
        """

        # Feedforward data to network, add it with original input as residual,
        # then return the activation output
        return torch_func.relu(x + self._layers(x))


class ChessNet(nn.Module):
    """
    ChessNet class
    """

    # ----- Initialization -----

    def __init__(
        self,
        input_channels: int,
        *,
        trunk_channels: int = 64,
        policy_hidden_channels: int = 32,
        value_hidden_channels: int = 128,
        residual_blocks_count: int = 3
    ) -> None:
        """
        Initialize class instance

        Args:
            input_channels: Input channel size

        Keyword Args:
            trunk_channels: Number of feature maps used throughout
            the stem and residual trunk
            policy_hidden_channels: Number of intermediate feature maps
            inside the policy head
            value_hidden_channels: Number of intermediate feature maps
            inside the value head
            residual_blocks_count: Residual block count inside hidden block

        Returns:
            None
        """

        # Call base class init
        super().__init__()

        # Initialize stem block
        self._stem_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU()
        )

        # Initialize trunk block
        self._trunk_block: nn.Sequential = nn.Sequential(
            *[
                ResidualBlock(trunk_channels)
                for _ in range(residual_blocks_count)
            ]
        )

        # Initialize policy head
        # 64 means destination square, while 5 means promotion type including
        # no promotion
        self._policy_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(trunk_channels, policy_hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(policy_hidden_channels, 64 * 5, 1)
        )

        # Initialize value head
        self._value_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(trunk_channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, value_hidden_channels),
            nn.ReLU(),
            nn.Linear(value_hidden_channels, 1),
            nn.Tanh()
        )


    # ----- Public methods -----

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Feedforward data

        Args:
            x: Data to input
        
        Returns:
            Tuple with first index as policy head output and second index
            as value head output
        """

        # Get the output of stem and trunk
        stem_output = self._stem_block(x)
        trunk_output = self._trunk_block(stem_output)

        # Get the output of policy head
        policy_output = self._policy_head(trunk_output)

        # Reorder the shape of policy output
        # Expected [batch, dest/promotion choice, src row, src col] ->
        # [batch, src row, src col, dest/promotion choice]
        policy_output = policy_output.permute(0, 2, 3, 1).contiguous()

        # Reshape the policy output to one tensor shape / batch
        # Expected [batch, src row, src col, dest/promotion choice] ->
        # [batch, all combinations of move / promotion choice]
        policy_output = policy_output.view(x.shape[0], 64 * 64 *5)

        # Get the output of value head and remove inner shape as it is redundant
        # Expected [batch, 1] -> [batch]
        value_output = self._value_head(trunk_output).squeeze(-1)

        # Return policy and value output
        return policy_output, value_output


class MCTS:
    """
    Monte carlo tree search class
    """

    # ----- Initialization -----

    def __init__(
        self,
        evaluation_method: Callable,
        *,
        simulation_count: int = 64,
        exploration_strength: float = 1.5,
        noise_distribution: float = 0.3,
        noise_fraction: float = 0.25
    ) -> None:
        """
        Initialize search parameters

        Args:
            evaluation_method: Caller-provided network evaluator returning legal-action
            scores and a value from the current player's perspective

        Keyword Args:
            simulation_count: Perform n simulations
            exploration_strength: Controls how strongly MCTS
            favors exploring less-visited moves, higher means more exploration
            noise_distribution: Controls how evenly random noise
            is distributed across legal moves, higher means more evenly
            noise_fraction: Controls how much random noise is mixed
            into the original policy probabilities, higher means more random noise

        Returns:
            None
        """

        # Initialize class attributes
        self._evaluation_method: Callable = evaluation_method
        self._simulation_count: int = simulation_count
        self._exploration_strength: float = exploration_strength
        self._noise_distribution: float = noise_distribution
        self._noise_fraction: float = noise_fraction


    # ----- Public methods -----

    def run(
        self,
        environment: Environment,
        *,
        add_noise: bool = False
    ) -> MCTSNode:
        """
        Run MCTS process

        Args:
            environment: Chess environment

        Keyword Args:
            add_noise: Add dirichlet noise to root move priors

        Returns:
            A root MCTS node
        """

        # Set current environment as root node
        root = MCTSNode(environment.copy())

        # Evaluate and expand root node
        self._evaluate_and_expand(root)

        # Add noise if add noise flag is True
        if add_noise:
            self._add_root_noise(root)

        # Loop through each iteration
        for _ in range(self._simulation_count):

            # Initialize variables
            node = root
            path = [node]

            # Traverse through each child node
            while node.expanded and node.children:
                _, node = self._select_child(node)

                # Append child node to path
                path.append(node)

                # Break if node is terminal node
                if node.environment.state is not None:
                    break

            # If current node is terminal set terminal value
            if node.environment.state is not None:
                value = node.environment.state

            # Else expand current node again
            else:
                value = self._evaluate_and_expand(node)

            # Backpropagate the values
            for visited_node in reversed(path):
                visited_node.visit_count += 1
                visited_node.value_sum += value
                value = -value

        # Return root node
        return root


    # ----- Helpers -----

    def _evaluate_and_expand(
        self,
        node: MCTSNode
    ) -> float:
        """
        Evaluate current state and expand possible moves from current state

        Args:
            node: State node

        Returns:
            Estimated value of current state
        """

        # Check whether the game is over
        game_state = node.environment.state
        if game_state is not None:

            # Set expanded to True and return terminal value
            node.expanded = True
            return game_state

        # Call caller owns encoding and neural-network inference
        policy_logits, value = self._evaluation_method(node.environment)

        # Parse legal moves ID into tensor type
        indices = torch.tensor(
            node.environment.legal_moves,
            dtype=torch.long,
            device=policy_logits.device,
        )

        # Get each legal move logits
        legal_logits = policy_logits[indices]

        # Apply softmax to the logits
        priors = torch.softmax(legal_logits, dim=0).cpu().tolist()

        # Clear current node children
        node.children = {}

        # Traverse through each priors
        for move_id, prior in zip(node.environment.legal_moves, priors):

            # Simulate move and draw a connection to current node
            next_state = node.environment.copy()
            next_state.move(move_id)
            node.children[move_id] = MCTSNode(next_state, prior=float(prior))

        # Set current node expanded state to True
        node.expanded = True

        # Return current state value
        return value

    def _add_root_noise(
        self,
        root: MCTSNode
    ) -> None:
        """
        Add little noise to move probabilities
        
        Args:
            root: Root node

        Returns:
            None
        """

        # If given move has no children directly skip
        if not root.children:
            return

        # Get all legal moves ID
        moves_id = list(root.children.keys())

        # Generate random noises
        noises = torch.distributions.Dirichlet(
            torch.full(
                (len(moves_id),), 
                self._noise_distribution
            )
        ).sample().tolist()

        # Traverse through each legal move and noise
        for move_id, noise in zip(moves_id, noises):

            # Get child node
            child = root.children[move_id]

            # Add noise to prior based on fraction
            frac = self._noise_fraction
            child.prior = (1.0 - frac) * child.prior + frac * noise

    def _select_child(
        self,
        parent: MCTSNode
    ) -> tuple[int, MCTSNode]:
        """
        Select which child to explore next based on PUCT algorithm

        Args:
            parent: Parent node

        Returns:
            Best move ID and child node
        """

        # Calculate square root of the parent visit count
        sqrt_parent_visits = math.sqrt(
            max(1, parent.visit_count)
        )

        # Initialize variables
        best_move_id = 0
        best_child = None
        best_score = -float("inf")

        # Traverse through each child node
        for move_id, child_node in parent.children.items():

            # Get child node value from parent perspective
            # As chess game player turn turns each action, moves
            # that is good for parent means it is bad for child
            # and vice versa, hence use negative sign to flip the score
            q_from_parent = -child_node.node_value

            # Calculate exploration bonus
            exploration_bonus = (
                self._exploration_strength
                * child_node.prior
                * sqrt_parent_visits
                / (1 + child_node.visit_count)
            )

            # Calculate final score based on past exploration score
            score = q_from_parent + exploration_bonus

            # Set current state as best score if bigger
            if score > best_score:
                best_move_id = move_id
                best_child = child_node
                best_score = score

        # Return best move ID and child node
        return best_move_id, best_child

