class Utils:
    @staticmethod
    def to_hms(seconds: float, include_frac_sec: bool = False) -> str:
        """
        Convert seconds to `hh:mm:ss` format
        """
        if seconds < 0:
            raise ValueError("seconds must be non-negative")

        hrs, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)

        if include_frac_sec:
            return f"{hrs:02.0f}:{mins:02.0f}:{secs:05.2f}"
        return f"{hrs:02.0f}:{mins:02.0f}:{secs:02.0f}"
