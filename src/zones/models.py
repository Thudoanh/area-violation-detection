"""Zone evaluation result; reuses the existing core Zone model."""

from dataclasses import dataclass
from typing import Optional

from src.core.models import Zone

ZONE_TYPES = ("SIDEWALK", "MONITORED", "ALLOWED", "IGNORE")


@dataclass
class ZoneMembership:
    matched_zones: list[Zone]
    effective_zone: Optional[Zone]

    @property
    def by_type(self) -> dict[str, bool]:
        return {kind: any(z.zone_type == kind for z in self.matched_zones)
                for kind in ZONE_TYPES}

    @property
    def target_zone(self) -> Optional[Zone]:
        if self.effective_zone and self.effective_zone.zone_type in ("SIDEWALK", "MONITORED"):
            return self.effective_zone
        return None
