"""Conservative moment groups using capture time and visual agreement."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


def _time_key(photo: dict[str, Any]) -> tuple[float, bool]:
    try:
        moment = datetime.fromisoformat(photo["captureTime"].replace("Z", "+00:00"))
        has_offset = moment.tzinfo is not None
        if not has_offset:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp(), has_offset
    except (KeyError, ValueError, TypeError):
        return 0.0, False


def _hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _visually_related(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if not first.get("phash") or not second.get("phash"):
        return False
    if _hash_distance(first["phash"], second["phash"]) > 12:
        return False
    first_signature = first.get("visualSignature")
    second_signature = second.get("visualSignature")
    if not first_signature or not second_signature:
        return False
    # A flat dark/blue/white frame has no reliable visual structure. Its DCT
    # hash alone must not manufacture a moment from unrelated blank photos.
    for signature in (first_signature, second_signature):
        spatial_ranges = (
            max(signature[channel::3]) - min(signature[channel::3])
            for channel in range(3)
        )
        if max(spatial_ranges) < 0.035:
            return False
    if all(first.get(key) and second.get(key) for key in ("width", "height")):
        first_ratio = first["width"] / first["height"]
        second_ratio = second["width"] / second["height"]
        if max(first_ratio, second_ratio) / min(first_ratio, second_ratio) > 1.2:
            return False
    distance = sum(abs(a - b) for a, b in zip(first_signature, second_signature))
    return distance / len(first_signature) <= 0.12


def _fits_group(group: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    if len(group) >= 48:
        return False
    anchor_time, anchor_has_offset = _time_key(group[0])
    candidate_time, candidate_has_offset = _time_key(candidate)
    if anchor_has_offset != candidate_has_offset:
        return False
    if not 0 <= candidate_time - anchor_time <= 15:
        return False
    if candidate_time - _time_key(group[-1])[0] > 8:
        return False
    # Require agreement with every member. Adjacent-frame similarity alone
    # creates long chains across an entire walk or camera pan.
    return all(_visually_related(member, candidate) for member in group)


def group_photos(photos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(photos, key=lambda photo: (_time_key(photo)[0], photo["path"]))
    moments: list[list[dict[str, Any]]] = []
    for photo in ordered:
        if moments and _fits_group(moments[-1], photo):
            moments[-1].append(photo)
        else:
            moments.append([photo])
    groups = []
    for index, moment in enumerate(moments):
        ids = [photo["id"] for photo in moment]
        identifier = hashlib.sha256("\0".join(ids).encode("utf-8")).hexdigest()[:24]
        usable = sorted(
            (photo for photo in moment if not photo.get("analysisError")),
            key=lambda photo: (-photo["qualityScore"], photo["id"]),
        )
        recommended = [usable[0]["id"]] if usable else []
        if (
            len(usable) > 2
            and usable[0]["qualityScore"] - usable[1]["qualityScore"] <= 4
        ):
            recommended.append(usable[1]["id"])
        count = len(moment)
        label = f"Moment {index + 1} · {count} {'photo' if count == 1 else 'photos'}"
        groups.append(
            {
                "id": identifier,
                "label": label,
                "photoIds": ids,
                "recommendedPhotoIds": recommended,
            }
        )
    return groups
