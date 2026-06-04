class BackgroundDirectoryNotFoundError(FileNotFoundError):
    """Raised when a background image directory cannot be found."""

    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
        super().__init__(f"Background directory not found: {dir_path}")


class ModelFileNotFoundError(FileNotFoundError):
    """Raised when a 3D model file cannot be found."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"Model file not found: {file_path}")
