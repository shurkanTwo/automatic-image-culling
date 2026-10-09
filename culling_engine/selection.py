"""Reversible first-pass proposals based only on preview technical evidence.

Confidence expresses strength of the technical evidence, not aesthetic merit
or a calibrated probability. Unselected photos remain undecided.
"""

from __future__ import annotations

import math
from typing import Any


def valid_metrics(metrics: Any) -> bool:
    """Reject missing, non-finite, or implausible cached analysis inputs."""
    limits = {
        "detail": 4.0,
        "gradient": 1.0,
        "contrast": 1.0,
        "highlightClipping": 1.0,
        "shadowClipping": 1.0,
        "luminanceStd": 0.5,
    }
    return isinstance(metrics, dict) and all(
        type(metrics.get(key)) in (int, float)
        and math.isfinite(metrics[key])
        and 0 <= metrics[key] <= limit
        for key, limit in limits.items()
    )


def _featureless_extreme(metrics: dict[str, float]) -> str | None:
    # Clipping alone is insufficient: high/low key subjects often include large
    # white/black backgrounds. Require almost all pixels plus negligible detail.
    if (
        metrics["contrast"] > 0.008
        or metrics["luminanceStd"] > 0.004
        or metrics["detail"] > 0.003
        or metrics["gradient"] > 0.0015
    ):
        return None
    if metrics["shadowClipping"] >= 0.998:
        return "Almost entirely black with negligible visible structure in the preview"
    if metrics["highlightClipping"] >= 0.998:
        return "Almost entirely white with negligible visible structure in the preview"
    return None


def _adequate(photo: dict[str, Any]) -> bool:
    metrics = photo["technicalMetrics"]
    return (
        photo["qualityScore"] >= 58
        and metrics["contrast"] >= 0.10
        and (metrics["detail"] >= 0.014 or metrics["gradient"] >= 0.012)
    )


def _clearly_weaker_peer(
    photo: dict[str, Any], peer: dict[str, Any], mode: str
) -> bool:
    weak = photo["technicalMetrics"]
    strong = peer["technicalMetrics"]
    cautious = mode == "cautious"
    if not (
        strong["detail"] >= (0.04 if cautious else 0.03)
        and strong["gradient"] >= (0.025 if cautious else 0.018)
        and weak["detail"] < (0.018 if cautious else 0.028)
        and weak["detail"] < strong["detail"] * (0.25 if cautious else 0.40)
        and weak["gradient"] < strong["gradient"] * (0.60 if cautious else 0.80)
        and weak["contrast"] >= max(0.15, strong["contrast"] * 0.70)
    ):
        return False
    first = photo.get("visualSignature")
    second = peer.get("visualSignature")
    if not first or not second or len(first) != 48 or len(second) != 48:
        return False
    # A coarse composition and hash match protects against marking an unrelated
    # smooth scene as blurry simply because a textured scene scored higher.
    if any(
        max(max(signature[c::3]) - min(signature[c::3]) for c in range(3)) < 0.10
        for signature in (first, second)
    ):
        return False
    distance = sum(abs(a - b) for a, b in zip(first, second)) / 48
    if distance > (0.04 if cautious else 0.065):
        return False
    if not photo.get("phash") or not peer.get("phash"):
        return False
    hash_distance = (int(photo["phash"], 16) ^ int(peer["phash"], 16)).bit_count()
    return hash_distance <= (10 if cautious else 12)


def suggest_selection(
    photos: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    mode: str = "cautious",
) -> list[dict[str, Any]]:
    """One deterministic proposal per photo; never treat decoding as rejection."""
    if mode not in ("cautious", "stronger"):
        raise ValueError("Selection mode must be cautious or stronger")
    ordered = sorted(photos, key=lambda photo: photo["id"])
    by_id = {photo["id"]: photo for photo in ordered}
    proposals = {}
    usable = set()

    def propose(identifier: str, decision: str, reason: str, confidence: float) -> None:
        proposals[identifier] = {
            "photoId": identifier,
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
        }

    for photo in ordered:
        identifier = photo["id"]
        if photo.get("analysisError") or not valid_metrics(
            photo.get("technicalMetrics")
        ):
            propose(
                identifier,
                "undecided",
                "Preview analysis unavailable; review the original",
                0.0,
            )
            continue
        reason = _featureless_extreme(photo["technicalMetrics"])
        if reason:
            propose(identifier, "pass", reason, 0.99)
            continue
        usable.add(identifier)
        propose(
            identifier,
            "undecided",
            "No clear technical selection; review this photo",
            0.2,
        )

    grouped = set()
    for group in groups:
        members = [
            by_id[identifier]
            for identifier in group["photoIds"]
            if identifier in usable
        ]
        # A singleton has no relative evidence, even when other group members
        # failed decoding or are blank. Handle it through the independent quota.
        if len(members) < 2:
            continue
        grouped.update(photo["id"] for photo in members)
        ranked = sorted(
            members, key=lambda photo: (-photo["qualityScore"], photo["id"])
        )
        adequate = [photo for photo in ranked if _adequate(photo)]
        if not adequate:
            continue
        best = adequate[0]
        propose(
            best["id"],
            "favorite",
            "Strongest technical preview in this similar moment; compare at 100%",
            0.80,
        )
        for photo in ranked:
            if photo["id"] == best["id"]:
                continue
            if _clearly_weaker_peer(photo, best, mode):
                propose(
                    photo["id"],
                    "pass",
                    "Much less visible detail than a sharper preview with matching composition in this moment",
                    0.94 if mode == "cautious" else 0.90,
                )
        if mode == "cautious" and len(adequate) > 1:
            runner_up = adequate[1]
            if best["qualityScore"] - runner_up["qualityScore"] <= 2.5:
                propose(
                    runner_up["id"],
                    "favorite",
                    "Similar technical strength to the best preview; kept both for comparison",
                    0.66,
                )

    independent = [photo for photo in ordered if photo["id"] in usable - grouped]
    candidates = sorted(
        (photo for photo in independent if _adequate(photo)),
        key=lambda photo: (-photo["qualityScore"], photo["id"]),
    )
    # A batch quota prevents every unrelated frame becoming a favorite. It is
    # only a selection quota: no rejected photos are manufactured to meet it.
    budget = math.ceil(len(independent) * (0.30 if mode == "cautious" else 0.22))
    for photo in candidates[:budget]:
        propose(
            photo["id"],
            "favorite",
            "Among the strongest technically adequate independent previews in this batch",
            0.68,
        )
    return [proposals[photo["id"]] for photo in ordered]
