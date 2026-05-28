class ModelFileNotFoundError(FileNotFoundError):
    """Raised when a 3D model file cannot be found."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"Model file not found: {file_path}")
