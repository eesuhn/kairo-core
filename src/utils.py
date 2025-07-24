class Utils:
    COMPLETE_SYMBOLS = (".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'")

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

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not text:
            return text

        lines = text.split("\n")
        processed_lines: list = []
        current_sentence: list = []

        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                if current_sentence:
                    processed_lines.append(" ".join(current_sentence))
                    current_sentence = []
                if i > 0 and i < len(lines) - 1:
                    processed_lines.append("")
                continue

            if Utils.is_complete_sentence(line):
                if current_sentence:
                    processed_lines.append(" ".join(current_sentence))
                    current_sentence = []
                processed_lines.append(line)
            else:
                current_sentence.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if (
                        next_line
                        and next_line[0].isupper()
                        and line.endswith((",", ":"))
                    ):
                        processed_lines.append(" ".join(current_sentence))
                        current_sentence = []

        if current_sentence:
            processed_lines.append(" ".join(current_sentence))

        result = []
        for i, line in enumerate(processed_lines):
            if line:
                result.append(line)
            elif i > 0 and i < len(processed_lines) - 1:
                result.append("\n")

        return "\n".join(result)

    @staticmethod
    def is_complete_sentence(text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        if text.endswith(Utils.COMPLETE_SYMBOLS):
            return True

        if len(text.split()) <= 5 and text[0].isupper():
            continuation_words = [
                "and",
                "or",
                "but",
                "the",
                "a",
                "an",
                "with",
                "of",
                "in",
                "is",
                "are",
                "was",
                "were",
                "to",
                "for",
                "by",
                "from",
                "at",
            ]
            words = text.split()
            if len(words) > 1:
                if words[-1].lower() in continuation_words:
                    return False
            return True

        if text.endswith((",", ":", ";", "-")):
            return False

        return False

    @staticmethod
    def ensure_complete_sentence(text: str) -> str:
        text = text.strip()
        if text and not text.endswith(Utils.COMPLETE_SYMBOLS):
            text += "."
        return text

    @staticmethod
    def ensure_capitalized(text: str) -> str:
        if not text:
            return text

        if text[0].islower():
            text = text[0].upper() + text[1:]

        for symbol in Utils.COMPLETE_SYMBOLS:
            parts = text.split(symbol)
            if len(parts) > 1:
                result = parts[0]
                for i in range(1, len(parts)):
                    part = parts[i].lstrip()
                    if part and part[0].islower():
                        part = part[0].upper() + part[1:]
                    whitespace_prefix = parts[i][
                        : len(parts[i]) - len(parts[i].lstrip())
                    ]
                    result += symbol + whitespace_prefix + part
                text = result

        return text
