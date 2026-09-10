"""Full annotated JPEG evidence, including Windows Unicode paths."""

from pathlib import Path

import cv2


class EvidenceWriter:
    def __init__(self, directory):
        self.directory = Path(directory)

    def save(self, frame, event):
        if not event.event_id.isalnum():
            raise ValueError("Snapshot event ID must be alphanumeric")
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise OSError("Could not encode snapshot")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{event.event_id}.jpg"
        # Exclusive creation prevents accidental evidence replacement.
        created = False
        try:
            with path.open("xb") as stream:
                created = True
                stream.write(encoded.tobytes())
        except OSError:
            if created:
                path.unlink(missing_ok=True)
            raise
        return str(path.resolve())
