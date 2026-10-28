from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    if not tag:
        return tag
    if tag[0] == "{":
        return tag.split("}", 1)[1]
    return tag


def load_dumpinfo_map(code_dir: Path) -> Dict[str, str]:
    """
    Build in-memory map from ConfigDumpInfo.xml:
      key = Metadata@name (XCF canonical name)
      value = Metadata@id (GUID)
    - One streaming pass via iterparse.
    - If file missing or unreadable, return empty dict.
    """
    try:
        code_dir = code_dir.resolve()
    except Exception:
        pass

    xml_path = code_dir / "ConfigDumpInfo.xml"
    if not xml_path.exists():
        logger.info("ConfigDumpInfo.xml not found under %s (GUIDs come from uuid attributes of object XML)", code_dir)
        return {}

    mapping: Dict[str, str] = {}
    logger.info("Loading GUID mapping from %s ...", xml_path)

    try:
        # Stream parse; release elements after use to keep memory low
        context = ET.iterparse(str(xml_path), events=("end",))
        for event, elem in context:
            try:
                if not isinstance(elem.tag, str):
                    elem.clear()
                    continue
                if _local_name(elem.tag) != "Metadata":
                    elem.clear()
                    continue
                name = elem.get("name")
                mid = elem.get("id")
                if name and mid:
                    # First wins; duplicates are rare but keep earliest
                    if name not in mapping:
                        mapping[name] = mid
            except Exception:
                # Be resilient to any malformed nodes
                pass
            finally:
                # Free children and element to limit memory
                elem.clear()
        logger.info("Loaded %d GUID entries from ConfigDumpInfo.xml", len(mapping))
        return mapping
    except Exception as e:
        logger.warning("Failed to parse %s: %s (GUID enrichment skipped)", xml_path, e)
        return {}