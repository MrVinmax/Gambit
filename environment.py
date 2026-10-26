# environment.py

# Module import
from __future__ import annotations

from copy import deepcopy

import chess


class Environment:
    """
    Environment class
    """

    # ----- Initialization -----

    def __init__(
        self,
        *,
        claim_draw: bool = True
    ) -> None:
        """
        Initialize class instance

        Keyword Args:
            claim_draw: Whether game will end in threefold repetition or
            fifty-move rule

        Returns:
            None
        """

        # Initialize instance attributes
        self._board: chess.Board = chess.Board()
        self._claim_draw: bool = claim_draw


    # ----- Public attributes -----

    @property
    def board(
        self
    ) -> chess.Board:
        """
        Return board instance

        Returns:
            Board instance
        """

        # Return board instance
        return self._board

    @property
    def turn(
        self
    ) -> chess.Color:
        """
        Return current turn

        Returns:
            Current turn color
        """

        # Return current turn color
        return self._board.turn

    @property
    def legal_moves(
        self
    ) -> list[int]:
        """
        Return possible legal move IDs

        Returns:
            List of legal move IDs
        """

        # Initialize promotion piece ID map
        promotion_id_map = {
            None: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4
        }

        # Return all possible move ID
        return [
            (
                (move.from_square * 64 + move.to_square)
                * 5
                + promotion_id_map.get(move.promotion, 0)
            )
            for move in self._board.legal_moves
        ]

    @property
    def state(
        self
    ) -> float | None:
        """
        Return current game state

        Returns:
            float if current state reached terminal state, else None
        """

        # Get current state outcome
        outcome = self._board.outcome(claim_draw=self._claim_draw)

        # If outcome is None it means the game isnt over
        if outcome is None:
            return None

        # If there is no winner it means draw
        if outcome.winner is None:
            return 0.0

        # Return value based on who is the winner
        return 1.0 if outcome.winner == self._board.turn else -1.0


    # ----- Public methods -----

    def move(
        self,
        move_id: int
    ) -> None:
        """
        Run a move

        Args:
            move_id: ID of the move

        Raises:
            ValueError: When given move is illegal

        Returns:
            None
        """

        # Decode ID to move instances
        move = self._decode_id_to_move(move_id)

        # Validate move legality
        if move not in self.board.legal_moves:
            raise ValueError("Given move is illegal")

        # Execute the move
        self._board.push(move)

    def copy(
        self,
    ) -> Environment:
        """
        Return a copied version of current instance

        Returns:
            Copy of current instance
        """

        # Return copied version of current instance
        return deepcopy(self)


    # ----- Helpers -----

    def _decode_id_to_move(
        self,
        move_id: str
    ) -> chess.Move:
        """
        Decode ID to move instance

        Args:
            move_id: Move ID

        Returns:
            Move instance
        """

        # Initialize ID promotion piece map
        id_promotion_map = {
            0: None,
            1: chess.KNIGHT,
            2: chess.BISHOP,
            3: chess.ROOK,
            4: chess.QUEEN
        }

        # Extract the promotion ID
        promotion_id = move_id % 5

        # Remove the promotion ID and extract the squares
        square_id = move_id // 5
        from_square = square_id // 64
        to_square = square_id % 64

        # Convert the promotion ID to a piece type
        promotion_piece = id_promotion_map.get(promotion_id)

        # Return move instance
        return chess.Move(
            from_square,
            to_square,
            promotion=promotion_piece,
        )

    def __str__(
        self
    ) -> str:
        """
        Return string-type board

        Returns:
            String-type board
        """

        # Return board in string type
        return str(self._board)

