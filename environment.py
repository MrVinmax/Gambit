# environment.py

# Module import
import chess


class Environment:
    """
    Environment class
    """

    # ----- Initialization -----

    def __init__(
        self
    ) -> None:
        """
        Initialize class instance

        Returns:
            None
        """

        # Initialize instance attributes
        self._board: chess.Board = chess.Board()


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

        # Return all possible move ID
        return [
            move.from_square * 64 + move.to_square
            for move in self._board.legal_moves
        ]


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


    # ----- Helpers -----

    def _decode_id_to_move(
        self,
        move_id: str
    ) -> chess.Move:
        """
        Decode ID to move

        Args:
            move_id: Move ID

        Returns:
            Move instance
        """

        # Get corresponding square based on given ID
        from_square = move_id // 64
        to_square = move_id % 64

        # Create move instance and return it
        return chess.Move(from_square, to_square)

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

