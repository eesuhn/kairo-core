from datetime import timedelta


class Utils:
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to `HH:MM:SS.mm` format."""
        if seconds < 0:
            return "00:00:00.00"
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
