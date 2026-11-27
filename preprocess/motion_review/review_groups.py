"""Conservative behavior grouping for the motion-inclusion review."""

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from motion_review_data import MotionDescription


GROUPING_VERSION = "behavior-context-v1"
LOCOMOTION = r"walk(?:ing)?|jog(?:ging)?|run(?:ning)?|crawl(?:ing)?"
ANGLE = r"(?:000|045|090|135|180|225|270|315|360)"
PHASE = r"(?:start|stop|loop)"
PACE = r"(?:very_fast|very_slow|normal_pace|fast|slow)"


def _filename_family(name: str) -> str:
    """Collapse locomotion directions/phases, retaining action and style words.

    Only known variant patterns are rewritten. A preceding instruction such as
    'kick_stop_walk' stays intact. Unknown patterns keep their exact type name.
    """
    shelf = r"(?:behind|front|right_side|left_side)_(?:high|medium|low)"
    transfer = re.fullmatch(
        rf"(?P<base>(?:(?:small|medium|big)_)*(?:heavy|light)_(?:one_hand|two_hands))_"
        rf"{shelf}_to_{shelf}(?:_[rl])?", name)
    if transfer:
        return transfer["base"] + "_shelf_transfer"
    match = re.fullmatch(
        rf"(?P<base>.*?(?:^|_)(?:{LOCOMOTION}))(?:_ff)?"
        rf"(?:_(?:{PHASE}))?_(?:{ANGLE})(?:_(?:{PHASE}))?"
        rf"(?:_[rl])?(?P<pace>_(?:{PACE}))?", name)
    if match:
        return match["base"] + (match["pace"] or "")
    match = re.fullmatch(
        rf"(?P<base>.*?(?:^|_)(?:{LOCOMOTION}))"
        r"(?:_(?:forward|forwards|backward|backwards|sideways|sideway|left|right))?"
        rf"(?:_(?:{PHASE}))?(?:_[rl])?", name)
    return match["base"] if match else name


def behavior_family(name: str) -> str:
    family = _filename_family(name)
    return {"neutral_walk": "walk", "walking": "walk", "sideway_walk": "walk",
            "neutral_jog": "jog", "jogging": "jog", "sideway_jog": "jog",
            "neutral_run": "run", "running": "run", "sideway_run": "run",
            "neutral_crawl": "crawl", "crawling": "crawl"}.get(family, family)


def context_signature(description: 'MotionDescription') -> tuple[str, str, str]:
    """Keep prop, body-support and load distinctions; unknown context stays apart."""
    normalize = lambda value: " ".join(value.lower().split())
    props = ";".join(sorted(normalize(p) for p in description.props.split(";")))
    body = ",".join(sorted(normalize(p) for p in description.body_position.split(",")))
    # A filename family retains context words; this additionally checks text
    # for physical conditions which can differ despite a common family name.
    evidence = normalize(description.text + " " + description.props)
    terms = re.findall(
        r"\b(?:heavy|light|weighted|loaded|unloaded|chair|sofa|couch|bench|"
        r"wall|ladder|stairs?|railing|handrail|crutch(?:es)?|wheelchair|"
        r"table|shelf|shelves|platform|floor|ground|horse|bicycle|motorcycle)\b", evidence)
    return props, body, ",".join(sorted(set(terms)))


def build_review_groups(
    motion_types: dict[str, str],
    descriptions: 'Mapping[str, MotionDescription]',
) -> dict[str, tuple[str, str]]:
    """Map original paths to (review group, reason); never infer missing context.

    Different context within one filename type also creates separate groups.
    This mapping never removes BVH examples from training.
    """
    candidates = defaultdict(list)
    groups = {}
    contexts_by_type = defaultdict(set)
    for path, name in motion_types.items():
        record = descriptions.get(path)
        if record is None or not record.text or not record.body_position:
            contexts_by_type[name].add(None)
            continue
        contexts_by_type[name].add(context_signature(record))
    for path, name in motion_types.items():
        record = descriptions.get(path)
        if record is None or not record.text or not record.body_position:
            key = name + "__missing_context" if len(contexts_by_type[name]) > 1 else name
            groups[path] = (key, "Exact type retained: incomplete official context")
            continue
        candidates[behavior_family(name)].append((path, name, context_signature(record)))
    reserved_groups = {group for group, reason in groups.values()}
    for family, members in candidates.items():
        by_context = defaultdict(list)
        for path, name, signature in members:
            by_context[signature].append((path, name))
        for signature, entries in by_context.items():
            names = {name for path, name in entries}
            if len(names) == 1 and len(contexts_by_type[next(iter(names))]) == 1:
                for path, name in entries:
                    groups[path] = (name, "Exact type retained: no equivalent behavior variant")
                continue
            suffix = hashlib.sha256(repr(signature).encode()).hexdigest()[:10]
            missing = any(None in contexts_by_type[name] for name in names)
            group = family if len(by_context) == 1 and not missing and family not in reserved_groups else f"{family}__context_{suffix}"
            reason = ("Same behavior and metadata context; direction/phase or shelf-location variants grouped"
                      if len(names) > 1 else "Filename type split by official prop/body/load context")
            for path, name in entries:
                groups[path] = (group, reason)
    return groups
