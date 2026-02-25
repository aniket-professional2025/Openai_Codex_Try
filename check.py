"""Utilities for factor calculations."""


def find_factors_and_count(number):
    """Return factors of ``number`` and how many factors it has.

    Args:
        number (int): Integer to evaluate.

    Returns:
        tuple[list[int], int]: A list of factors in ascending order and
        the number of factors.

    Raises:
        ValueError: If number is 0.
        TypeError: If number is not an int.
    """
    if not isinstance(number, int):
        raise TypeError("number must be an integer")
    if number == 0:
        raise ValueError("0 has infinitely many factors")

    target = abs(number)
    factors = [candidate for candidate in range(1, target + 1) if target % candidate == 0]
    return factors, len(factors)
