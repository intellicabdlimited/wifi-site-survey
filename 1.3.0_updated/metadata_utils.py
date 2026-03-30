from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

PARAM_ALIASES = {
    "signal_strength": ["signal strength", "signal_strength", "signal_strength_main", "rssi"],
    "secondary_signal_strength": ["secondary signal strength", "secondary_signal_strength"],
    "tertiary_signal_strength": ["tertiary signal strength", "tertiary_signal_strength"],
    "snr": ["snr", "signal to noise ratio", "signal_to_noise_ratio"],
    "noise": ["noise"],
    "data_rate": ["data rate", "datarate", "data_rate"],
    "throughput": ["throughput"],
    "channel_utilization": ["channel utilization", "channel_utilization"],
    "channel_interference": ["channel interference", "channel interferecne", "channel_interference"],
    "channel_width": ["channel width", "channel_width"],
    "spectrum_channel_power": ["spectrum channel power", "spectrum_channel_power"],
    "network_health": ["network health", "network_health"],
    "network_issues": ["network issues", "network_issues"],
    "number_of_access_points": [
        "number of aps",
        "number of access points",
        "number_of_aps",
        "number_of_access_points",
    ],
    "bluetooth_coverage": ["bluetooth coverage", "bluetooth_coverage"],
    "associated_access_points": [
        "associated access point",
        "associated access points",
        "associated_access_point",
        "associated_access_points",
    ],
    "interferers": ["interferers", "interferer"],
    "survey_routes_and_access_points": [
        "survey routes and access points",
        "survey_routes_and_access_points",
    ],
}

PARAM_PRETTY = {
    "signal_strength": "Signal Strength",
    "secondary_signal_strength": "Secondary Signal Strength",
    "tertiary_signal_strength": "Tertiary Signal Strength",
    "snr": "SNR",
    "noise": "Noise",
    "data_rate": "Data Rate",
    "throughput": "Throughput",
    "channel_utilization": "Channel Utilization",
    "channel_interference": "Channel Interference",
    "channel_width": "Channel Width",
    "spectrum_channel_power": "Spectrum Channel Power",
    "network_health": "Network Health",
    "network_issues": "Network Issues",
    "number_of_access_points": "Number of Access Points",
    "bluetooth_coverage": "Bluetooth Coverage",
    "associated_access_points": "Associated Access Points",
    "interferers": "Interferers",
    "survey_routes_and_access_points": "Survey Routes and Access Points",
}

PROJECT_TO_CORE_KEY = {
    "signal_strength": "signal_strength_main",
    "number_of_access_points": "number_of_aps",
}

FLOOR_PATTERNS = [
    ("Ground Floor", [r"\bground\s*floor\b", r"\bgroundfloor\b", r"\bkitchen\b", r"\bmain floor\b"]),
    ("Lower Floor", [r"\blower\s*floor\b", r"\bliving\s*room\b", r"\bbasement\b"]),
    ("Upper Floor", [r"\bupper\s*floor\b", r"\bupstairs\b", r"\bbedroom\b"]),
]

BAND_RE = re.compile(r"\b(?:on\s+(?:the\s+)?)?(2\.4|5|6)\s*ghz\b", re.IGNORECASE)
FLOOR_CAPTURE_RE = re.compile(
    r"\bfor\s+(?P<floor>.+?)\s+on\s+(?:the\s+)?(?:2\.4|5|6)\s*ghz\b",
    re.IGNORECASE,
)

ROUTER_TRAILING_PATTERNS = [
    re.compile(r"\s+survey\b.*$", re.IGNORECASE),
    re.compile(r"\s+with\s+mesh\b.*$", re.IGNORECASE),
    re.compile(r"\s+without\s+mesh\b.*$", re.IGNORECASE),
    re.compile(r"\s+mesh\s+extender\b.*$", re.IGNORECASE),
    re.compile(r"\s+mesh\b.*$", re.IGNORECASE),
]


@dataclass(frozen=True)
class AssetMetadata:
    router_key: str
    parameter_key: str
    parameter_display: str
    floor_name: str
    band: str
    role: str
    caption_text: str = ""
    source_docx: str = ""
    group_id: str = ""
    path: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_spaces(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    value = str(text).replace("\u00a0", " ")
    value = value.strip()
    if value.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", value).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", normalize_spaces(text)).strip("_")


def clean_router_name(name: str) -> str:
    value = normalize_spaces(Path(str(name)).stem)
    if not value:
        return value

    month_match = MONTH_RE.search(value)
    if month_match:
        value = value[: month_match.start()].strip(" -_")

    for pattern in ROUTER_TRAILING_PATTERNS:
        value = pattern.sub("", value).strip(" -_")

    if "_" in value:
        first = value.split("_", 1)[0].strip()
        if 2 <= len(first) <= len(value):
            value = first

    if not value:
        value = normalize_spaces(Path(str(name)).stem)
    return value


def normalize_band_value(text: str) -> str:
    s = normalize_spaces(text)
    m = BAND_RE.search(s)
    if not m:
        return s
    return f"{m.group(1)}GHz"


def canonical_band(text: str) -> Optional[str]:
    band = normalize_band_value(text)
    return band or None


def normalize_floor_name(text: str) -> str:
    s = normalize_spaces(text).strip(" .-_")
    if not s:
        return ""
    low = s.lower()
    for floor_name, patterns in FLOOR_PATTERNS:
        if any(re.search(pat, low, re.IGNORECASE) for pat in patterns):
            return floor_name
    return s.title() if s.islower() else s


def canonical_floor_name(text: str) -> Optional[str]:
    value = normalize_floor_name(text)
    return value or None


def _normalize_keyish(text: str) -> str:
    value = normalize_spaces(text).lower().replace("-", " ").replace("/", " ")
    value = re.sub(r"[_]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def canonical_metric_key(text: str) -> Optional[str]:
    s = _normalize_keyish(text)
    if not s:
        return None

    alias_map = {
        key: {_normalize_keyish(alias) for alias in aliases if _normalize_keyish(alias)}
        for key, aliases in PARAM_ALIASES.items()
    }

    # Prefer exact matches first.
    for key, alias_norms in alias_map.items():
        if s == key or s in alias_norms:
            return key

    # Then prefer the most specific embedded alias. This prevents
    # "secondary signal strength" and "tertiary signal strength" from being
    # collapsed into the broader "signal strength" bucket.
    candidates = []
    for key, alias_norms in alias_map.items():
        for alias in alias_norms:
            if re.search(r"\b" + re.escape(alias) + r"\b", s):
                candidates.append((len(alias), key, alias))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    return None


def to_core_metric_key(text: str) -> Optional[str]:
    project_key = canonical_metric_key(text)
    if not project_key:
        return None
    return PROJECT_TO_CORE_KEY.get(project_key, project_key)


def canonical_param_key(text: str) -> Optional[str]:
    return canonical_metric_key(text)


def canonical_param_display(text: str) -> Optional[str]:
    key = canonical_metric_key(text)
    if not key:
        return None
    return PARAM_PRETTY.get(key, key.replace("_", " ").title())


def strip_figure_prefix(text: str) -> str:
    return re.sub(r"^(figure|fig)\s*\d+\s*[:\-]\s*", "", normalize_spaces(text), flags=re.IGNORECASE)


def extract_floor_name(text: str) -> Optional[str]:
    caption = strip_figure_prefix(text)
    m = FLOOR_CAPTURE_RE.search(caption)
    if m:
        return canonical_floor_name(m.group("floor"))
    if re.search(r"\bfor\b", caption, re.IGNORECASE):
        tail = re.split(r"\bfor\b", caption, maxsplit=1, flags=re.IGNORECASE)[1]
        tail = re.split(r"\bon\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
        return canonical_floor_name(tail)
    return None


def parse_caption_metadata(caption_text: str, default_router: str = "") -> Optional[AssetMetadata]:
    caption = strip_figure_prefix(caption_text)
    parameter_key = canonical_param_key(caption)
    floor_name = extract_floor_name(caption)
    band = canonical_band(caption)
    if not parameter_key or not floor_name or not band:
        return None
    router_key = clean_router_name(default_router)
    return AssetMetadata(
        router_key=router_key,
        parameter_key=parameter_key,
        parameter_display=PARAM_PRETTY.get(parameter_key, parameter_key.replace("_", " ").title()),
        floor_name=floor_name,
        band=band,
        role="heatmap",
        caption_text=caption,
        source_docx="",
        group_id="",
        path="",
    )


def parse_filename_metadata(filename: str) -> Optional[AssetMetadata]:
    p = Path(filename)
    stem = p.stem
    role = "scale" if stem.lower().endswith("_scale") else "heatmap"
    if role == "scale":
        stem = stem[:-6]
    if "_" not in stem:
        return None
    router_prefix, caption = stem.split("_", 1)
    base = parse_caption_metadata(caption, default_router=router_prefix)
    if not base:
        return None
    return AssetMetadata(
        router_key=clean_router_name(router_prefix),
        parameter_key=base.parameter_key,
        parameter_display=base.parameter_display,
        floor_name=base.floor_name,
        band=base.band,
        role=role,
        caption_text=base.caption_text,
        source_docx="",
        group_id="",
        path=str(filename),
    )
