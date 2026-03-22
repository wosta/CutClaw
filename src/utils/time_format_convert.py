def timecode_to_seconds(timecode: str) -> float:
    """Convert timecode HH:MM:SS.mmm to seconds."""
    hours, minutes, seconds_milliseconds = timecode.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds


def hhmmss_to_seconds(time_str: str, fps: float = 24.0) -> float:
    """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 4:
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        fps = fps or 24.0
        return h * 3600 + m * 60 + s + (f / fps)
    if len(parts) == 3:
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    return float(parts[0])


def seconds_to_hhmmss(sec: float) -> str:
    """Convert seconds to HH:MM:SS.s format."""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"

def format_srt_timestamp(milliseconds: int) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    ms = int(milliseconds)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    return f"{h:02d}:{mi:02d}:{s:02d},{tail:03d}"
