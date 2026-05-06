"""LCP exception hierarchy."""


class LCPError(Exception):
    """Base exception for all LCP-related errors."""
    pass


class LCPInputError(LCPError):
    """Invalid input parameters (bad frequency, negative length, etc.)."""
    pass


class LCPGenerationError(LCPError):
    """Z/Y matrix computation failed."""
    pass


class LCPFittingError(LCPError):
    """Vector fitting failed."""
    pass


class FitULMExportError(LCPError):
    """Writing or verifying a fitULM file failed."""
    pass
