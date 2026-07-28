"""
crop_saver.py  --  per-track crop helper (on-disk saving is DISABLED).

Once every person has a stable track_id (from Stage 4), we can CROP the pixels
inside their bounding box. This helper throttles that per track (at most one crop
every `interval` frames) and returns the crops IN MEMORY.

IMPORTANT: this does NOT write anything to disk. Saving per-person crop images to
`crops/<camera>/id_<track>/` was removed -- the ReID path never needs on-disk
crops (the embedder makes its own in-memory crop via `crop_person()` in
`reid/service.py`), and the pipeline discards this helper's return value. It is
only constructed when `crops.save: true` in config.yaml (off by default); leave
it off unless you re-add a real consumer for the returned crops.
"""

from detector import crop_person   # the one shared "box -> safe crop" primitive


class CropSaver:
    """Return throttled per-track bbox crops in memory. Writes nothing to disk."""

    def __init__(self, output_dir="crops", interval=10, padding=0):
        # Kept for API compatibility (main.py passes a per-camera dir); unused
        # while disk saving is disabled.
        self.output_dir = output_dir

        # Emit a crop for a given track only once every `interval` frames.
        self.interval = max(1, interval)

        # Optionally grow each box by `padding` pixels on every side.
        self.padding = padding

        # Last frame index at which we emitted each track, to enforce interval.
        self._last_saved = {}

    def save(self, frame, detections, frame_index):
        """
        For each tracked detection, crop it out of the CLEAN `frame` (before boxes
        are drawn) if enough frames have passed since we last emitted that person.
        Returns the list of {track_id, crop} extracted this call (in memory; not
        written to disk).
        """
        extracted_crops = []

        for det in detections:
            # No ID yet -> can't file it under a person. Skip.
            if det.track_id is None:
                continue

            # Throttle: only emit this person again after `interval` frames.
            last = self._last_saved.get(det.track_id)
            if last is not None and (frame_index - last) < self.interval:
                continue

            # Clamp + slice via the shared primitive; None = degenerate box.
            crop = crop_person(frame, det, padding=self.padding)
            if crop is None:
                continue

            extracted_crops.append({"track_id": det.track_id, "crop": crop})
            self._last_saved[det.track_id] = frame_index

        return extracted_crops
