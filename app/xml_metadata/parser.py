"""
XML metadata parser. Parses 1C configuration descriptors directly from
code/<...>.xml into Configuration / MetadataCategory / MetadataObject etc.
compatible with the existing TXT pipeline.

Public entry point: ``XmlMetadataParser`` (see end of module).

Ported from the experimental lab parser. Removed: extension-base overlay,
property overlays, diagnostics helpers, CLI entry points, and dependency on
the lab's xml_scan walker (the parser now consumes a ready file list from
``CodeFileIndexer``).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from . import property_extractor as xml_props
from .folder_map import FOLDER_TO_RU_CATEGORY
from .rules import (
    CFG_TYPE_PREFIX_TO_RU,
    CFG_TYPE_VALUE_TO_RU,
    EN_SINGULAR_TO_RU_SINGULAR,
    FOLDER_TO_OWNER_TYPE,
    FOLDER_TO_REF_TYPE,
    METADATA_SEGMENT_TO_RU,
    STANDARD_ATTR_BOOLEAN_TYPES,
    STANDARD_ATTR_DATE_TYPES,
    STANDARD_ATTR_DEFAULT_XML_VALUES,
    STANDARD_ATTR_NAME,
    STANDARD_ATTR_NAME_BY_FOLDER,
    STANDARD_ATTR_OWNER_PROPERTY_MAP,
    STANDARD_ATTR_STRING_TYPES,
    STD_DEFAULTS,
    STD_KEEP_EMPTY_NAMES,
    STD_NIL,
    STD_NOISY_PROPS,
    STD_SUPPRESSED_BY_NAME,
    STD_SUPPRESSED_BY_TYPE_AND_NAME,
    STD_SUPPRESSED_NAMES,
    VALUE_TRANSLATIONS,
    XML_PROP_TO_RU,
)
from parsers.metadata_parser import (
    Attribute,
    Command,
    Configuration,
    EnumValue,
    Form,
    Layout,
    MetadataCategory,
    MetadataObject,
    TabularPart,
)

logger = logging.getLogger(__name__)

EXTERNAL_DS_CONTAINER_TAG_BY_FOLDER = {
    "Tables": "Table",
    "Cubes": "Cube",
    "DimensionTables": "DimensionTable",
}

















BINARY_STORAGE_PROPS = {
    "ИспользованиеХраненияВХранилищеДвоичныхДанных",
    "ПолеИспользованияХраненияВХранилищеДвоичныхДанных",
}

LOCAL_PROP_TYPE_BY_CHILD = {
    "Attribute": "attribute",
    "AddressingAttribute": "addressing_attribute",
    "TabularSection": "tabular_section",
    "Resource": "resource",
    "Dimension": "dimension",
    "AccountingFlag": "accounting_flag",
    "ExtDimensionAccountingFlag": "ext_dimension_accounting_flag",
    "EnumValue": "enum_value",
    "Command": "command",
    "Column": "document_journal_column",
    "URLTemplate": "http_service_url_template",
    "Method": "http_service_method",
    "Table": "external_data_source_table",
    "Cube": "external_data_source_cube",
    "Function": "external_data_source_function",
    "Field": "external_data_source_field",
    "DimensionTable": "external_data_source_dimension_table",
}

def _strip_txt_quotes(value: str) -> str:
    return value.strip('"')


def _legacy_type_value(value: Any) -> Any:
    if not isinstance(value, list):
        text = _strip_txt_quotes(str(value)).strip()
        if not text:
            return None
        if "," in text:
            return [part.strip() for part in " ".join(text.split()).split(",") if part.strip()]
        return [text]
    values = [_strip_txt_quotes(str(item)).strip() for item in value if str(item).strip()]
    values = [
        "КонстантыНабор" if item == "НаборКонстант" else "Диаграмма" if item == "d7p1:Chart" else item
        for item in values
    ]
    if not values:
        return None
    if len(values) == 1:
        text = values[0]
        if "," in text:
            return [part.strip() for part in " ".join(text.split()).split(",") if part.strip()]
        return values
    return values


def _legacy_comma_multiline_value(value: Any) -> Any:
    if not isinstance(value, list):
        return _compat_choice_link_item(_strip_txt_quotes(str(value)))
    raw_values = [_compat_choice_link_item(_strip_txt_quotes(str(item))) for item in value]
    keep_leading_blank = any(str(item).strip() in {"", ","} for item in raw_values)
    values = [item for item in raw_values if item and item != ","]
    if not values:
        return ""
    if values[0].startswith("Отбор.НачалоПериода") or values[0].startswith("Отбор.ОкончаниеПериода"):
        values = [item for item in values if not (item.startswith("Отбор.НачалоПериода") or item.startswith("Отбор.ОкончаниеПериода"))]
    if keep_leading_blank and values:
        values.insert(0, ",")
    if len(values) == 1:
        return values[0]
    return [f"{item}," if idx < len(values) - 1 and not item.endswith(",") else item for idx, item in enumerate(values)]


def _compat_choice_link_item(item: str) -> str:
    if not item.endswith(")") or "(" not in item:
        return item
    open_index = item.find("(")
    left, right = item[:open_index], item[open_index + 1:-1]
    if left.endswith(".StandardAttribute.Owner") and right.startswith("Отбор."):
        return f"{right}(Владелец)"
    if right.startswith(("Строка:", "Булево:", "Число:", "Дата:")):
        return item
    compact_right = _compact_data_path(right)
    return f"{left}({compact_right})" if compact_right != right else item


def _legacy_single_or_list_value(value: Any) -> Any:
    if not isinstance(value, list):
        return _strip_txt_quotes(str(value))
    values = [_strip_txt_quotes(str(item)) for item in value if str(item) != ""]
    if not values:
        return ""
    return values[0] if len(values) == 1 else values


def _legacy_property_value(name: str, value: Any) -> Any:
    if name == "Тип":
        return _legacy_type_value(value)
    if name in {"ПараметрыВыбора", "СвязиПараметровВыбора"}:
        return _legacy_comma_multiline_value(value)
    if name == "Ссылки":
        return _legacy_single_or_list_value(value)
    if name == "ФормаВыбора" and isinstance(value, str):
        return _format_metadata_path(value)
    if isinstance(value, str):
        return _strip_txt_quotes(value).replace(" \n", "\n")
    if isinstance(value, list):
        return [_strip_txt_quotes(item) if isinstance(item, str) else item for item in value]
    return value


def _legacy_props_compat(props: Dict[str, Any]) -> Dict[str, Any]:
    out = {name: _legacy_property_value(name, value) for name, value in props.items()}
    out.pop("ScheduleLink", None)
    return out


def _normalized_properties(elem: ET.Element, type_key: str | None = None, folder: str | None = None) -> Dict[str, Any]:
    return xml_props.extract_properties(elem, type_key=type_key, folder=folder)


def _merge_structural_props(target: Dict[str, Any], source: Dict[str, Any], names: Iterable[str]) -> None:
    for name in names:
        value = source.get(name)
        if value not in (None, "", []):
            if name == "Движения" and isinstance(value, list) and len(value) == 1:
                value = value[0]
            target[name] = value


def _local_name(tag: str) -> str:
    if not tag:
        return tag
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _first_child(elem: ET.Element | None, name: str) -> ET.Element | None:
    if elem is None:
        return None
    for child in list(elem):
        if isinstance(child.tag, str) and _local_name(child.tag) == name:
            return child
    return None


def _children(elem: ET.Element | None, name: str | None = None) -> List[ET.Element]:
    if elem is None:
        return []
    out = []
    for child in list(elem):
        if not isinstance(child.tag, str):
            continue
        if name is None or _local_name(child.tag) == name:
            out.append(child)
    return out


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _local_string(elem: ET.Element) -> str:
    first_content = None
    seen_content = False
    for item in elem.iter():
        if item is elem or not isinstance(item.tag, str):
            continue
        if _local_name(item.tag) != "item":
            continue
        lang_val = ""
        content_val = ""
        for sub in list(item):
            if not isinstance(sub.tag, str):
                continue
            ln = _local_name(sub.tag)
            if ln == "lang":
                lang_val = (sub.text or "").strip()
            elif ln == "content":
                seen_content = True
                content_val = (sub.text or "").strip("\n\t")
        if not content_val:
            continue
        if lang_val == "ru":
            return content_val
        if first_content is None:
            first_content = content_val
    if first_content is None:
        for child in elem.iter():
            if isinstance(child.tag, str) and _local_name(child.tag) == "content":
                seen_content = True
                value = (child.text or "").strip("\n\t")
                if value:
                    first_content = value
                    break
    if first_content is not None:
        return first_content
    if seen_content:
        return ""
    return _text(elem)


def _map_cfg_type(value: str) -> str:
    value = (value or "").strip()
    if value in CFG_TYPE_VALUE_TO_RU:
        return CFG_TYPE_VALUE_TO_RU[value]
    if value.startswith("cfg:"):
        value = value[4:]
    if value in CFG_TYPE_VALUE_TO_RU:
        return CFG_TYPE_VALUE_TO_RU[value]
    if "." not in value:
        return value
    head, tail = value.split(".", 1)
    mapped = CFG_TYPE_PREFIX_TO_RU.get(head)
    if mapped:
        return f"{mapped}.{tail}"
    return value


def _map_register_record(value: str) -> str:
    value = (value or "").strip()
    if "." not in value:
        return value
    head, tail = value.split(".", 1)
    mapped = EN_SINGULAR_TO_RU_SINGULAR.get(head)
    if mapped:
        return f"{mapped}.{tail}"
    return value


def _format_metadata_path(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return ".".join(
        EN_SINGULAR_TO_RU_SINGULAR.get(part, METADATA_SEGMENT_TO_RU.get(part, STANDARD_ATTR_NAME.get(part, _translate_value(part))))
        for part in value.split(".")
    )


def _compact_data_path(value: str) -> str:
    value = (value or "").strip().split()[0] if (value or "").strip() else ""
    parts = [part for part in (value or "").split(".") if part]
    if "StandardAttribute" in parts:
        return _last_metadata_part(value)
    if "TabularSection" in parts and "Attribute" in parts:
        section_index = parts.index("TabularSection")
        attribute_index = parts.index("Attribute")
        if section_index + 1 < len(parts) and attribute_index + 1 < len(parts):
            return f"{parts[section_index + 1]}.{parts[attribute_index + 1]}"
    if "Attribute" in parts:
        attribute_index = parts.index("Attribute")
        if attribute_index + 1 < len(parts):
            return parts[attribute_index + 1]
    for marker in ("Dimension", "Resource"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index + 1 < len(parts):
                return parts[marker_index + 1]
    return _format_metadata_path(value)


def _last_metadata_part(value: str | None) -> str:
    if not value:
        return ""
    parts = value.split(".")
    if len(parts) >= 2 and parts[-2] == "StandardAttribute":
        return STANDARD_ATTR_NAME.get(parts[-1], _translate_value(parts[-1]))
    return _translate_value(parts[-1])


def _blank_if_minus_one(value: str | None) -> str:
    if value in (None, "", "-1", "0"):
        return ""
    return _last_metadata_part(value)


def _blank_value(value: str | None) -> str:
    return "" if value in (None, "", "-1", "0") else value


def _child_text(elem: ET.Element | None, child_name: str) -> str:
    child = _first_child(elem, child_name)
    return _text(child)


def _raw_text_preserve_spaces(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    for child in elem.iter():
        if isinstance(child.tag, str) and _local_name(child.tag) == "content":
            value = child.text
            if value is not None:
                return value.strip("\n\t")
    if elem.text is None:
        return None
    return elem.text.strip("\n\t")


def _format_type_value(value: str, qualifiers: Dict[str, ET.Element]) -> str:
    base = _map_cfg_type(value)
    if value == "xs:string":
        qualifier = qualifiers.get("StringQualifiers")
        length = _child_text(qualifier, "Length")
        allowed = _child_text(qualifier, "AllowedLength")
        parts = [part for part in (length, _translate_value(allowed) if allowed else None) if part]
        return f"{base}({', '.join(parts)})" if parts else base
    if value == "xs:decimal":
        qualifier = qualifiers.get("NumberQualifiers")
        precision = _child_text(qualifier, "Digits") or _child_text(qualifier, "Precision")
        scale = _child_text(qualifier, "FractionDigits") or _child_text(qualifier, "Scale")
        sign = _child_text(qualifier, "NonNegative") or _child_text(qualifier, "AllowedSign")
        parts = [part for part in (precision, scale, _translate_value(sign) if sign and sign != "Any" else None) if part]
        return f"{base}({', '.join(parts)})" if parts else base
    if value == "xs:dateTime":
        qualifier = qualifiers.get("DateQualifiers")
        fraction = _child_text(qualifier, "DateFractions")
        return f"{base}({_translate_value(fraction)})" if fraction else base
    return base


def _raw_type_values(type_elem: ET.Element | None) -> List[str]:
    if type_elem is None:
        return []
    qualifiers: Dict[str, ET.Element] = {}
    for child in type_elem.iter():
        name = _local_name(child.tag) if isinstance(child.tag, str) else ""
        if name in {"StringQualifiers", "NumberQualifiers", "DateQualifiers"} and name not in qualifiers:
            qualifiers[name] = child
    values = []
    for child in type_elem.iter():
        if child is type_elem:
            continue
        if not isinstance(child.tag, str) or _local_name(child.tag) not in {"Type", "TypeSet"}:
            continue
        value = _text(child)
        if value:
            values.append(_format_type_value(value, qualifiers))
    direct = _text(type_elem)
    if not values and direct:
        values.append(_format_type_value(direct, qualifiers))
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


GUIDISH_RE = re.compile(r"\b[0-9a-fA-F]:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def _choice_parameter_type_label(value_type: str) -> str:
    if not value_type:
        return ""
    return _map_cfg_type(value_type)


def _choice_parameter_value(node: ET.Element | None) -> Tuple[str, str]:
    if node is None:
        return "", ""
    value_type = _attr_by_local_name(node, "type")
    value = _text(node)
    if value_type.endswith("DesignTimeRef"):
        return "", _format_design_time_ref(value)
    if value_type.endswith("FixedArray"):
        return "ФиксированныйМассив", "ФиксированныйМассив"
    return _choice_parameter_type_label(value_type), _translate_value(value)


def _format_design_time_ref(value: str) -> str:
    parts = [part for part in (value or "").split(".") if part]
    if GUIDISH_RE.search(value or "") or (parts and all(xml_props.PLAIN_UUID_RE.fullmatch(part) for part in parts)):
        return ""
    if len(parts) >= 4 and parts[2] == "EnumValue":
        return f"ПеречислениеСсылка.{parts[1]}:{parts[3]}"
    if len(parts) >= 3 and parts[2] == "EmptyRef":
        prefix = {
            "Enum": "ПеречислениеСсылка",
            "Catalog": "СправочникСсылка",
            "Document": "ДокументСсылка",
            "ChartOfCharacteristicTypes": "ПланВидовХарактеристикСсылка",
            "ChartOfCalculationTypes": "ПланВидовРасчетаСсылка",
        }.get(parts[0])
        if prefix:
            return f"{prefix}.{parts[1]}:"
    if len(parts) >= 3 and parts[0] == "Catalog":
        return f"СправочникСсылка.{parts[1]}:{parts[-1]}"
    if len(parts) >= 3 and parts[0] == "Document":
        return f"ДокументСсылка.{parts[1]}:{parts[-1]}"
    return _format_metadata_path(value)


def _choice_parameters_raw(node: ET.Element | None) -> List[str]:
    result = []
    for item in _children(node):
        name = item.attrib.get("name") or item.attrib.get("Name") or _attr_by_local_name(item, "name")
        value_node = _first_child(item, "value")
        type_label, value = _choice_parameter_value(value_node)
        value_type = _attr_by_local_name(value_node, "type") if value_node is not None else ""
        if name and (value or type_label or value_type.endswith("DesignTimeRef")):
            result.append(f"{name}({type_label}:{value})" if type_label else f"{name}({value})")
    return result


def _choice_parameter_links_raw(node: ET.Element | None) -> List[str]:
    result = []
    leading_blank_seen = False
    leading_skipped_guid_seen = False
    for link in _children(node):
        name = _child_text(link, "Name")
        data_path = _child_text(link, "DataPath")
        if not name or not data_path:
            if not result:
                leading_blank_seen = True
            continue
        if GUIDISH_RE.search(name) or GUIDISH_RE.search(data_path):
            if not result:
                leading_skipped_guid_seen = True
            continue
        if name.startswith("Отбор.НачалоПериода") or name.startswith("Отбор.ОкончаниеПериода"):
            continue
        if "." in name and data_path.strip().startswith("Отбор."):
            result.append(f"{data_path}({_compact_data_path(name)})")
        else:
            result.append(f"{name}({_compact_data_path(data_path)})")
    if (leading_blank_seen or leading_skipped_guid_seen) and result:
        result.insert(0, "")
    return result


def _data_path_field(types: ET.Element) -> str:
    value = _child_text(types, "DataPathField")
    if value.endswith(".StandardAttribute.Ref"):
        filter_path = _child_text(types, "TypesFilterField")
        filter_field = _last_metadata_part(filter_path)
        if ".TabularSection." in filter_path and ".Attribute." in filter_path:
            filter_field = filter_path.rsplit(".Attribute.", 1)[1]
        ref = STANDARD_ATTR_NAME.get("Ref", "Ссылка")
        return f"{filter_field}.{ref}" if filter_field else ref
    if value in {"", "-1", "0"}:
        return _last_metadata_part(_child_text(types, "TypesFilterField"))
    filter_field = _last_metadata_part(_child_text(types, "TypesFilterField"))
    if filter_field and ".Attribute." in value:
        data_field = _last_metadata_part(value)
        return f"{filter_field}.{data_field}" if data_field else filter_field
    return _last_metadata_part(value)


def _prop_value(prop: ET.Element) -> Any:
    name = _local_name(prop.tag)
    if name in {"Synonym", "Comment", "ToolTip", "Format", "EditFormat"}:
        return _local_string(prop)
    if name == "Type":
        types = [_map_cfg_type(_text(t)) for t in prop.iter() if isinstance(t.tag, str) and _local_name(t.tag) == "Type"]
        if types:
            return types
        value = _text(prop)
        return [_map_cfg_type(value)] if value else []
    if name == "RegisterRecords":
        values = [_map_register_record(_text(item)) for item in _children(prop, "Item")]
        return [v for v in values if v]
    if name == "Owners":
        values = [_map_register_record(_text(item)) for item in _children(prop, "Item")]
        return [v for v in values if v]
    if name == "ChoiceParameterLinks":
        values = []
        for link in list(prop):
            link_name = _text(_first_child(link, "Name"))
            data_path = _text(_first_child(link, "DataPath"))
            if link_name and data_path:
                values.append(f"{data_path}({link_name})")
        return values
    localized = _local_string(prop) if list(prop) else ""
    if localized and "\n" not in localized:
        return localized
    if list(prop):
        values = []
        for child in list(prop):
            if isinstance(child.tag, str) and _local_name(child.tag) in {"Item", "Type"}:
                value = _text(child)
                if value:
                    values.append(value)
        if values:
            return values
    return _text(prop)


def _attr_by_local_name(elem: ET.Element, name: str) -> str:
    for key, value in elem.attrib.items():
        if key == name or key.endswith("}" + name):
            return value
    return ""


def _fill_value(prop: ET.Element) -> str:
    if _attr_by_local_name(prop, "nil") == "true":
        return ""
    if _text(prop).endswith(".00000000-0000-0000-0000-000000000000"):
        return ""
    value_type = _attr_by_local_name(prop, "type")
    if value_type == "xs:string":
        return f"{VALUE_TRANSLATIONS.get(value_type, value_type)}:{prop.text or ''}"
    if value_type == "xs:boolean":
        return f"Булево:{_translate_value(_text(prop))}"
    if value_type == "xs:dateTime":
        value = _text(prop)
        if "T" in value:
            date_part, time_part = value.split("T", 1)
            y, m, d = date_part.split("-", 2)
            hh, mm, ss = (time_part.split(":", 2) + ["00", "00"])[:3]
            ss = ss.split(".", 1)[0]
            return f"Дата:{int(d):02d}.{int(m):02d}.{y} {int(hh)}:{int(mm):02d}:{int(ss):02d}"
        return f"Дата:{value}"
    return _prop_value(prop)


def _translate_value(value: str) -> str:
    value = value or ""
    return VALUE_TRANSLATIONS.get(value, value)


CONFIG_METADATA_PATH_PROPS = {
    "ОсновнаяФорма",
    "ОсновнаяФормаОтчета",
    "ОсновнаяФормаВариантаОтчета",
    "ОсновнаяФормаНастроекОтчета",
    "ОсновнаяФормаПоиска",
    "ОсновнаяФормаКонстант",
    "ОсновнаяФормаВыбораПользователейСистемыВзаимодействия",
    "ОсновнаяФормаДанныхВерсииИсторииДанных",
    "ОсновнаяФормаРазличийВерсийИсторииДанных",
    "ОсновнаяФормаИсторииИзмененийИсторииДанных",
    "ОсновнаяФормаНастроекДинамическогоСписка",
    "ОсновнойМакетОформленияОтчета",
    "ОсновнойСтиль",
    "ОсновнойЯзык",
    "ОсновныеРоли",
}

BASE_CONFIGURATION_DEFAULTS = {
    "ПринадлежностьОбъекта": "Собственный",
    "ПоддерживатьСоответствиеОбъектамРасширяемойКонфигурацииПоВнутреннимИдентификаторам": "Истина",
    "РежимИспользованияБлочногоХраненияДвоичныхДанных": "НеИспользовать",
    "РежимХранилищаДвоичныхДанных": "НеИспользовать",
}

INHERITED_EXTENSION_CONFIGURATION_PROPS = (
    "РежимИспользованияМодальности",
    "РежимИспользованияСинхронныхВызововРасширенийПлатформыИВнешнихКомпонент",
    "РежимИспользованияТабличныхПространствБазыДанных",
)


_VERSION_8_3_RE = re.compile(r"^Version8_3_(\d+)$")


def _normalize_configuration_value(name: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_configuration_value(name, item) for item in value]
    if not isinstance(value, str):
        return value
    if name in CONFIG_METADATA_PATH_PROPS:
        return _format_metadata_path(value)
    translated = _translate_value(value)
    if translated == value:
        m = _VERSION_8_3_RE.match(value)
        if m:
            return f"Версия8_3_{m.group(1)}"
    return translated


def _normalize_configuration_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {name: _normalize_configuration_value(name, value) for name, value in props.items()}


def _standard_prop_value(prop: ET.Element) -> Any:
    value = xml_props.prop_value(prop)
    return "" if value is None else value


def _properties_ru(props_elem: ET.Element | None) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for prop in _children(props_elem):
        if _attr_by_local_name(prop, "nil") == "true":
            continue
        xml_name = _local_name(prop.tag)
        ru_name = XML_PROP_TO_RU.get(xml_name)
        if not ru_name:
            continue
        value = _prop_value(prop)
        if ru_name in {"МинимальноеЗначение", "МаксимальноеЗначение"} and value == "":
            continue
        props[ru_name] = value
    return props


def _is_std_default(name: str, value: Any) -> bool:
    if isinstance(value, list):
        return False
    value_s = str(value)
    default = STD_DEFAULTS.get(name)
    if default is not None and value_s == default:
        return True
    if name == "ЗначениеЗаполнения" and value_s in {"Ложь", "Булево:Ложь", "0001-01-01T00:00:00", "Дата:01.01.0001 0:00:00"}:
        return True
    if name == "ЗначениеЗаполнения" and value_s.endswith(".00000000-0000-0000-0000-000000000000"):
        return True
    return value_s.endswith(".EmptyRef")


def _standard_attr_raw_default_value(prop: ET.Element) -> Any:
    name = _local_name(prop.tag)
    if _attr_by_local_name(prop, "nil") == "true":
        return STD_NIL
    if name in {"ChoiceParameterLinks", "ChoiceParameters"}:
        if not _children(prop):
            return []
        return tuple(
            ET.tostring(child, encoding="unicode")
            for child in _children(prop)
        )
    if name in {"Synonym", "Comment", "ToolTip", "Format", "EditFormat"}:
        return xml_props.local_string(prop)
    if name in {"LinkByType", "ChoiceForm"}:
        return _text(prop)
    return _text(prop)


def _is_standard_attr_xml_default(prop: ET.Element) -> bool:
    name = _local_name(prop.tag)
    if name not in STANDARD_ATTR_DEFAULT_XML_VALUES:
        return False
    return _standard_attr_raw_default_value(prop) == STANDARD_ATTR_DEFAULT_XML_VALUES[name]


def _should_preserve_listed_noisy_default(
    *,
    folder: str,
    attr_name: str,
    out_name: str,
    value: Any,
    props: Dict[str, Any],
    listed_props: Dict[str, Any] | None = None,
    owner_props: Dict[str, Any] | None = None,
) -> bool:
    listed_props = listed_props or {}
    owner_props = owner_props or {}
    if out_name == "ЗаполнятьИзДанныхЗаполнения" and str(value) == "Ложь":
        if attr_name == "Владелец" and (
            props.get("ИспользованиеПодчинения") not in (None, "", [])
            or props.get("Синоним") not in (None, "", [])
            or listed_props.get("ПроверкаЗаполнения") == "ВыдаватьОшибку"
        ):
            return True
        if attr_name == "Родитель" and any(props.get(key) not in (None, "", []) for key in ("Синоним", "Подсказка")):
            return True
    if out_name == "ПроверкаЗаполнения" and str(value) == "НеПроверять":
        if attr_name == "Владелец" and (
            props.get("ЗаполнятьИзДанныхЗаполнения") == "Истина"
            or listed_props.get("ЗаполнятьИзДанныхЗаполнения") == "Истина"
        ):
            return True
        if attr_name == "Владелец" and any(props.get(key) not in (None, "", []) for key in ("Тип", "Синоним")):
            return True
        if attr_name == "Наименование" and folder == "ChartsOfCharacteristicTypes" and props.get("Подсказка") not in (None, "", []):
            return True
        if attr_name == "Период" and folder == "InformationRegisters" and props.get("Синоним") not in (None, "", []):
            return True
    return False


def _is_empty_design_ref_fill_value(child: ET.Element) -> bool:
    if _local_name(child.tag) != "FillValue":
        return False
    value_type = _attr_by_local_name(child, "type")
    if not value_type.endswith("DesignTimeRef"):
        return False
    raw_value = _text(child)
    if raw_value == "" or raw_value.endswith(".EmptyRef") or raw_value.endswith(".00000000-0000-0000-0000-000000000000"):
        return True
    return bool(raw_value) and all(xml_props.PLAIN_UUID_RE.fullmatch(part) for part in raw_value.split(".") if part)


def _should_preserve_listed_empty_fill_value(
    *,
    child: ET.Element,
    attr_name: str,
    out_name: str,
    value: Any,
    listed_props: Dict[str, Any] | None = None,
) -> bool:
    if out_name != "ЗначениеЗаполнения" or value != "":
        return False
    if attr_name not in {"Владелец", "Родитель"}:
        return False
    if not _is_empty_design_ref_fill_value(child):
        return False
    listed_props = listed_props or {}
    return (
        listed_props.get("ЗаполнятьИзДанныхЗаполнения") == "Истина"
        or listed_props.get("ПроверкаЗаполнения") == "ВыдаватьОшибку"
        or any(listed_props.get(key) not in (None, "", []) for key in ("Синоним", "Подсказка"))
    )


def _should_skip_standard_attr_prop(
    *,
    child: ET.Element,
    owner_type: str | None,
    folder: str,
    attr_name: str,
    out_name: str,
    value: Any,
    owner_props: Dict[str, Any] | None = None,
    listed_props: Dict[str, Any] | None = None,
    preserve_listed_standard_attrs: bool = False,
) -> bool:
    xml_name = _local_name(child.tag)
    if preserve_listed_standard_attrs:
        # Keep properties that anchor an explicitly-listed standard attr so the
        # attr itself is not dropped after the usual default/empty filtering.
        if (
            attr_name == "Номер"
            and out_name == "ЗначениеЗаполнения"
            and value == "Строка:"
            and _attr_by_local_name(child, "type") == "xs:string"
        ):
            return False
        if attr_name in STD_KEEP_EMPTY_NAMES and out_name == "ЗначениеЗаполнения" and value == "":
            return False
        if attr_name == "Наименование" and out_name == "ЗаполнятьИзДанныхЗаполнения" and str(value) == "Истина":
            return False
    if (
        attr_name == "Номер"
        and out_name == "ЗначениеЗаполнения"
        and isinstance(value, str)
        and value.startswith("Строка:")
        and value[len("Строка:"):].strip() == ""
    ):
        number_length = str((owner_props or {}).get("ДлинаНомера", "")).strip()
        allowed_length = str((owner_props or {}).get("ДопустимаяДлинаНомера", "")).strip()
        fill_length = len(value[len("Строка:"):])
        if allowed_length and allowed_length != "Фиксированная":
            return False
        if not number_length.isdigit() or fill_length != int(number_length):
            return False
    if _is_standard_attr_xml_default(child) and xml_name not in {"FillChecking", "FillFromFillingValue"}:
        return True
    if out_name in {"МинимальноеЗначение", "МаксимальноеЗначение"} and value == "":
        return True
    if out_name == "ЗначениеЗаполнения" and value == "":
        return True
    if out_name == "ЗаполнятьИзДанныхЗаполнения" and str(value) == "Истина" and attr_name in {"Дата", "Код", "Период"}:
        return False
    if (
        out_name == "ПроверкаЗаполнения"
        and str(value) == "НеПроверять"
        and attr_name in {"Код", "Наименование"}
        and folder in {"Catalogs", "ExchangePlans"}
        and any((owner_props or {}).get(object_prop) not in (None, "", []) for object_prop, _ in STANDARD_ATTR_OWNER_PROPERTY_MAP.get(folder, {}).get(attr_name, ()))
    ):
        return False
    if _is_std_default(out_name, value):
        return True
    if attr_name in STD_SUPPRESSED_NAMES and out_name in STD_NOISY_PROPS:
        return True
    if out_name in STD_SUPPRESSED_BY_NAME.get(attr_name, set()):
        return True
    if owner_type and out_name in STD_SUPPRESSED_BY_TYPE_AND_NAME.get(owner_type, {}).get(attr_name, set()):
        return True
    return False


def _parse_standard_attributes(
    props_elem: ET.Element | None,
    owner_name: str,
    folder: str,
    *,
    include_empty: bool = False,
    owner_props: Dict[str, Any] | None = None,
    preserve_listed_standard_attrs: bool = False,
) -> List[Dict[str, Any]]:
    standard_attrs = None
    for prop in _children(props_elem):
        if _local_name(prop.tag) == "StandardAttributes":
            standard_attrs = prop
            break
    if standard_attrs is None:
        return []

    owner_type = FOLDER_TO_OWNER_TYPE.get(folder)
    out = []

    for std in _children(standard_attrs, "StandardAttribute"):
        raw_name = std.attrib.get("name") or std.attrib.get("Name") or ""
        folder_names = STANDARD_ATTR_NAME_BY_FOLDER.get(folder, {})
        if raw_name not in folder_names and raw_name not in STANDARD_ATTR_NAME:
            continue
        name = folder_names.get(raw_name) or STANDARD_ATTR_NAME[raw_name]
        if not name:
            continue
        props: Dict[str, Any] = {}
        listed_props: Dict[str, Any] = {}
        for child in _children(std):
            out_name = XML_PROP_TO_RU.get(_local_name(child.tag))
            if out_name:
                listed_props[out_name] = _standard_prop_value(child)
        listed_default_props: Dict[str, Any] = {}
        listed_empty_fill_value: Any = None
        for child in _children(std):
            xml_name = _local_name(child.tag)
            out_name = XML_PROP_TO_RU.get(xml_name)
            if not out_name:
                continue
            value = _standard_prop_value(child)
            if (
                preserve_listed_standard_attrs
                and _should_preserve_listed_empty_fill_value(
                    child=child,
                    attr_name=name,
                    out_name=out_name,
                    value=value,
                    listed_props=listed_props,
                )
            ):
                listed_empty_fill_value = value
            if (
                preserve_listed_standard_attrs
                and out_name in {"ПроверкаЗаполнения", "ЗаполнятьИзДанныхЗаполнения"}
                and _is_std_default(out_name, value)
            ):
                listed_default_props[out_name] = value
            if (
                preserve_listed_standard_attrs
                and name == "Номер"
                and out_name == "ЗначениеЗаполнения"
                and value == ""
                and _attr_by_local_name(child, "type") == "xs:string"
                and str((owner_props or {}).get("ДопустимаяДлинаНомера", "")).strip() == "Фиксированная"
            ):
                value = "Строка:"
            if not include_empty and _should_skip_standard_attr_prop(
                child=child,
                owner_type=owner_type,
                folder=folder,
                attr_name=name,
                out_name=out_name,
                value=value,
                owner_props=owner_props,
                listed_props=listed_props,
                preserve_listed_standard_attrs=preserve_listed_standard_attrs,
            ):
                continue
            if isinstance(value, list):
                if value or include_empty:
                    props[out_name] = value
                continue
            props[out_name] = value
        props = _legacy_props_compat(props)
        if "Подсказка" in props:
            props["Подсказка"] = xml_props.normalize_legacy_tooltip(props["Подсказка"], "attribute")
        if preserve_listed_standard_attrs:
            if listed_empty_fill_value is not None and (props or name == "Владелец"):
                props.setdefault("ЗначениеЗаполнения", listed_empty_fill_value)
            for key, val in listed_default_props.items():
                if _should_preserve_listed_noisy_default(
                    folder=folder,
                    attr_name=name,
                    out_name=key,
                    value=val,
                    props=props,
                    listed_props=listed_props,
                    owner_props=owner_props,
                ):
                    props.setdefault(key, val)
        if not props and not include_empty:
            continue
        props["Стандартный"] = True
        out.append({"name": name, "properties": props})

    # The old TXT parser materializes Owner as a standard attribute when it is
    # listed, even if only noisy/default XML properties remain after filtering.
    has_owner = any((std.attrib.get("name") or std.attrib.get("Name")) == "Owner" for std in _children(standard_attrs, "StandardAttribute"))
    if has_owner and not include_empty and not any(attr.get("name") == "Владелец" for attr in out):
        out.append({"name": "Владелец", "properties": {"Стандартный": True}})
    return out


def _standard_attr_self_ref_type(folder: str, owner_name: str) -> str | None:
    prefix = FOLDER_TO_REF_TYPE.get(folder)
    if not prefix:
        return None
    return f"{prefix}.{owner_name}"


def _copy_standard_attr_owner_properties(
    props: Dict[str, Any],
    obj_props: Dict[str, Any],
    folder: str,
    attr_name: str,
) -> None:
    for object_prop, attr_prop in STANDARD_ATTR_OWNER_PROPERTY_MAP.get(folder, {}).get(attr_name, ()):
        value = obj_props.get(object_prop)
        if value not in (None, "", []):
            props.setdefault(attr_prop, value)


def _enrich_standard_attr_type(
    props: Dict[str, Any],
    *,
    folder: str,
    owner_name: str,
    attr_name: str,
    obj_props: Dict[str, Any],
    basic_props: Dict[str, Any] | None = None,
) -> None:
    if attr_name == "Ссылка":
        ref_type = _standard_attr_self_ref_type(folder, owner_name)
        if ref_type:
            props.setdefault("Тип", ref_type)
    elif attr_name == "Родитель":
        ref_type = _standard_attr_self_ref_type(folder, owner_name)
        if ref_type:
            props.setdefault("Тип", ref_type)
    elif attr_name in STANDARD_ATTR_BOOLEAN_TYPES:
        props.setdefault("Тип", "Булево")
    elif attr_name in STANDARD_ATTR_DATE_TYPES:
        props.setdefault("Тип", "Дата")
    elif attr_name in STANDARD_ATTR_STRING_TYPES:
        props.setdefault("Тип", "Строка")
    elif attr_name == "Владелец":
        owners = obj_props.get("Владельцы")
        if owners:
            props.setdefault("Тип", xml_props.legacy_property_value("Владельцы", owners))
        usage = obj_props.get("ИспользованиеПодчинения") or (basic_props or {}).get("ИспользованиеПодчинения")
        if usage:
            usage_text = {"Items": "Элементам", "FoldersAndItems": "ГруппамИЭлементам", "Folders": "Группам"}.get(str(usage), _translate_value(str(usage)))
            usage_text = {"ToItems": "Элементам", "ToFoldersAndItems": "ГруппамИЭлементам", "ToFolders": "Группам"}.get(usage_text, usage_text)
            props.setdefault("ИспользованиеПодчинения", usage_text)

    _copy_standard_attr_owner_properties(props, obj_props, folder, attr_name)


def _enrich_standard_attrs(
    attrs: List[Dict[str, Any]],
    obj_props: Dict[str, Any],
    *,
    folder: str,
    owner_name: str,
    basic_props: Dict[str, Any] | None = None,
) -> None:
    for attr in attrs:
        attr_name = attr.get("name")
        if not isinstance(attr_name, str) or not attr_name:
            continue
        props = attr.setdefault("properties", {})
        props["Стандартный"] = True
        _enrich_standard_attr_type(
            props,
            folder=folder,
            owner_name=owner_name,
            attr_name=attr_name,
            obj_props=obj_props,
            basic_props=basic_props,
        )
        props.update(_legacy_props_compat(props))


def _enrich_owner_standard_attr(attrs: List[Dict[str, Any]], obj_props: Dict[str, Any], basic_props: Dict[str, Any] | None = None) -> None:
    owners = obj_props.get("Владельцы")
    has_owners_property = "Владельцы" in obj_props
    owner_attr = None
    for attr in attrs:
        if attr.get("name") == "Владелец":
            owner_attr = attr
            break
    if owner_attr is None:
        if not has_owners_property:
            return
        owner_attr = {"name": "Владелец", "properties": {}}
        attrs.append(owner_attr)
    props = owner_attr.setdefault("properties", {})
    props["Стандартный"] = True
    if owners:
        props.setdefault("Тип", xml_props.legacy_property_value("Владельцы", owners))
    usage = obj_props.get("ИспользованиеПодчинения") or (basic_props or {}).get("ИспользованиеПодчинения")
    if usage:
        usage_text = {"Items": "Элементам", "FoldersAndItems": "ГруппамИЭлементам", "Folders": "Группам"}.get(str(usage), _translate_value(str(usage)))
        usage_text = {"ToItems": "Элементам", "ToFoldersAndItems": "ГруппамИЭлементам", "ToFolders": "Группам"}.get(usage_text, usage_text)
        props.setdefault("ИспользованиеПодчинения", usage_text)
    props.update(_legacy_props_compat(props))
    if owners:
        owner_type = xml_props.legacy_property_value("Владельцы", owners)
        props["Тип"] = [owner_type] if isinstance(owner_type, str) else owner_type


def _normalize_tabular_choice_links(attr: Dict[str, Any], tabular_name: str) -> None:
    props = attr.get("properties") or {}
    value = props.get("СвязиПараметровВыбора")
    if not isinstance(value, list):
        return
    keep_leading_blank = any(str(item).strip() in {"", ","} for item in value)
    items = [str(item) for item in value if str(item).strip() not in {"", ","}]
    filtered = []
    for item in items:
        clean = item.rstrip(",")
        if clean.startswith(("Отбор.НачалоПериодаПримененияОтбора(", "Отбор.ОкончаниеПериодаПримененияОтбора(")):
            match = re.search(r"\(([^).]+)\.", clean)
            if match and match.group(1) != tabular_name:
                continue
        filtered.append(clean)
    if not filtered:
        props.pop("СвязиПараметровВыбора", None)
    elif len(filtered) == 1:
        props["СвязиПараметровВыбора"] = [",", filtered[0]] if keep_leading_blank else filtered[0]
    else:
        result = [
            f"{item}," if idx < len(filtered) - 1 and not item.endswith(",") else item
            for idx, item in enumerate(filtered)
        ]
        if keep_leading_blank:
            result.insert(0, ",")
        props["СвязиПараметровВыбора"] = result


def _name_from_props(props: Dict[str, Any], fallback: str) -> str:
    name = props.get("Имя")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _camel_synonym(name: str) -> str:
    if not name:
        return ""
    parts = re.sub(r"(?<=[а-яёa-z0-9])(?=[А-ЯЁA-Z])", " ", name).split()
    return " ".join([parts[0], *[part[:1].lower() + part[1:] for part in parts[1:]]]) if parts else ""


def _is_probably_xml(path: Path) -> bool:
    try:
        data = path.read_bytes()[:8]
    except Exception:
        return False
    if data.startswith(b"\xef\xbb\xbf<") or data.startswith(b"<"):
        return True
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return True
    return False


def _parse_xml_root(path: Path) -> ET.Element:
    return ET.parse(str(path)).getroot()


def _metadata_elem(root: ET.Element) -> ET.Element:
    if _local_name(root.tag) == "MetaDataObject":
        for child in list(root):
            if isinstance(child.tag, str):
                return child
    return root


def _metadata_version_lt(version: str | None, major: int, minor: int) -> bool:
    if not version:
        return False
    parts = version.split(".", 1)
    try:
        current = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return False
    return current < (major, minor)


def _has_fixed_string_type(elem: ET.Element) -> bool:
    type_elem = _first_child(_first_child(elem, "Properties"), "Type")
    for value in xml_props.type_values(type_elem):
        if isinstance(value, str) and value.startswith("Строка(") and "Фиксированная" in value:
            return True
    return False


def _has_string_like_type(elem: ET.Element) -> bool:
    type_elem = _first_child(_first_child(elem, "Properties"), "Type")
    for value in xml_props.type_values(type_elem):
        if isinstance(value, str) and (
            value.startswith("ОпределяемыйТип.")
            or (value.startswith("Строка(") and "Фиксированная" in value)
        ):
            return True
    return False


def _has_self_closed_string_fill_value(elem: ET.Element) -> bool:
    fill_value = _first_child(_first_child(elem, "Properties"), "FillValue")
    if fill_value is None or fill_value.text is not None:
        return False
    value_type = _attr_by_local_name(fill_value, "type")
    return value_type == "xs:string"


def _typed_empty_design_ref_fill_value(elem: ET.Element) -> str:
    props_elem = _first_child(elem, "Properties")
    fill_value = _first_child(props_elem, "FillValue")
    value_type = _attr_by_local_name(fill_value, "type")
    if not value_type.endswith("DesignTimeRef"):
        return ""
    raw_value = _text(fill_value)
    if not raw_value or not all(xml_props.PLAIN_UUID_RE.fullmatch(part) for part in raw_value.split(".") if part):
        return ""
    type_values = xml_props.type_values(_first_child(props_elem, "Type"))
    if len(type_values) != 1:
        return ""
    type_value = str(type_values[0])
    if type_value.startswith(("ПланСчетовСсылка.", "ПеречислениеСсылка.")):
        return f"{type_value}:"
    return ""


def _link_by_type_source_tabular(elem: ET.Element) -> str:
    props_elem = _first_child(elem, "Properties")
    data_path = _child_text(_first_child(props_elem, "LinkByType"), "DataPath")
    match = re.search(r"\.TabularSection\.([^.]+)\.Attribute\.", data_path)
    return match.group(1) if match else ""


def _parse_attribute(
    elem: ET.Element,
    fallback: str,
    type_key: str = "attribute",
    owner_folder: str | None = None,
    xml_version: str | None = None,
) -> Dict[str, Any]:
    props = _normalized_properties(elem, type_key=type_key, folder=owner_folder)
    fill_value = props.get("ЗначениеЗаполнения")
    if (
        type_key in {"attribute", "dimension", "resource", "tabular_section_attribute"}
        and fill_value == ""
        and _has_self_closed_string_fill_value(elem)
        and _has_string_like_type(elem)
    ):
        props["ЗначениеЗаполнения"] = "Строка:"
        fill_value = props.get("ЗначениеЗаполнения")
    if type_key == "attribute" and owner_folder == "Documents" and fill_value == "":
        typed_empty_fill = _typed_empty_design_ref_fill_value(elem)
        if typed_empty_fill:
            props["ЗначениеЗаполнения"] = typed_empty_fill
            fill_value = props.get("ЗначениеЗаполнения")
    if (
        type_key in {"attribute", "dimension", "resource", "tabular_section_attribute"}
        and isinstance(fill_value, str)
        and fill_value.startswith("Строка:")
        and fill_value[len("Строка:"):] != ""
        and fill_value[len("Строка:"):].strip() == ""
        and _has_fixed_string_type(elem)
    ):
        props["ЗначениеЗаполнения"] = ""
    link_by_type = props.get("СвязьПоТипу")
    if isinstance(link_by_type, str) and GUIDISH_RE.search(link_by_type) and type_key in {"attribute", "dimension", "resource"}:
        props["СвязьПоТипу"] = ""
    if _metadata_version_lt(xml_version, 2, 20):
        for prop_name in BINARY_STORAGE_PROPS:
            props.pop(prop_name, None)
    if owner_folder in {"Reports", "DataProcessors"}:
        props.pop("ИспользованиеХраненияВХранилищеДвоичныхДанных", None)
        props.pop("ПолеИспользованияХраненияВХранилищеДвоичныхДанных", None)
    if owner_folder == "Tasks" and type_key == "addressing_attribute":
        props.pop("ИспользованиеХраненияВХранилищеДвоичныхДанных", None)
        props.pop("ПолеИспользованияХраненияВХранилищеДвоичныхДанных", None)
    return {"name": _name_from_props(props, fallback), "properties": props}


def _parse_tabular(elem: ET.Element, fallback: str, owner_folder: str | None = None, owner_name: str | None = None, xml_version: str | None = None) -> Dict[str, Any]:
    props = _normalized_properties(elem, type_key="tabular_section")
    tabular_name = _name_from_props(props, fallback)
    attrs = []
    child_objects = _first_child(elem, "ChildObjects")
    for child in _children(child_objects, "Attribute"):
        attr = _parse_attribute(child, "", "tabular_section_attribute", owner_folder=owner_folder, xml_version=xml_version)
        link_by_type = attr.get("properties", {}).get("СвязьПоТипу")
        source_tabular = _link_by_type_source_tabular(child)
        if isinstance(link_by_type, str) and GUIDISH_RE.search(link_by_type):
            attr["properties"]["СвязьПоТипу"] = tabular_name
        elif source_tabular and source_tabular != tabular_name:
            attr["properties"]["СвязьПоТипу"] = tabular_name
        _normalize_tabular_choice_links(attr, tabular_name)
        attrs.append(attr)
    return {"name": tabular_name, "properties": props, "attributes": attrs}


def _parse_url_template(elem: ET.Element) -> Dict[str, Any] | None:
    props = _normalized_properties(elem, type_key="http_service_url_template")
    name = _name_from_props(props, "")
    if not name:
        return None
    methods = []
    child_objects = _first_child(elem, "ChildObjects")
    for method in _children(child_objects, "Method"):
        m_props = _normalized_properties(method, type_key="http_service_method")
        m_name = _name_from_props(m_props, "")
        if m_name:
            methods.append({"Имя": m_name, "Свойства": m_props})
    return {"Имя": name, "Свойства": props, "Методы": methods}


def _parse_external_ds_field(elem: ET.Element, fallback: str = "") -> Dict[str, Any]:
    props = _normalized_properties(elem, type_key="external_data_source_field")
    props["ВидОбъектаВнешнегоИсточникаДанных"] = "Field"
    return {"name": _name_from_props(props, fallback), "properties": props}


def _parse_external_ds_function(elem: ET.Element, fallback: str = "") -> Dict[str, Any]:
    props = _normalized_properties(elem, type_key="external_data_source_function")
    props["ВидОбъектаВнешнегоИсточникаДанных"] = "Function"
    return {"name": _name_from_props(props, fallback), "properties": props}


def _parse_external_ds_container(elem: ET.Element, fallback: str, md_kind: str) -> Dict[str, Any]:
    type_key = LOCAL_PROP_TYPE_BY_CHILD.get(md_kind)
    props = _normalized_properties(elem, type_key=type_key)
    props["ВидОбъектаВнешнегоИсточникаДанных"] = md_kind
    name = _name_from_props(props, fallback)
    attrs = []
    child_objects = _first_child(elem, "ChildObjects")
    for child in _children(child_objects):
        child_name = _local_name(child.tag)
        if child_name == "Field":
            attrs.append(_parse_external_ds_field(child))
        elif child_name == "Dimension":
            dim = _parse_attribute(child, "", "dimension", owner_folder="ExternalDataSources")
            dim.setdefault("properties", {})["ВидОбъектаВнешнегоИсточникаДанных"] = "Dimension"
            attrs.append(dim)
        elif child_name == "Resource":
            res = _parse_attribute(child, "", "resource", owner_folder="ExternalDataSources")
            res.setdefault("properties", {})["ВидОбъектаВнешнегоИсточникаДанных"] = "Resource"
            attrs.append(res)
        elif child_name == "Form":
            form_name = _text(child)
            if form_name:
                props.setdefault("Формы", []).append(form_name)
        elif child_name == "Template":
            template_name = _text(child)
            if template_name:
                props.setdefault("Макеты", []).append(template_name)
        elif child_name == "Command":
            cmd_props = _normalized_properties(child, type_key="command")
            cmd_name = _name_from_props(cmd_props, "")
            if cmd_name:
                props.setdefault("Команды", []).append({"Имя": cmd_name, "Свойства": cmd_props})
        elif child_name == "DimensionTable":
            table_name = _text(child)
            if table_name:
                props.setdefault("ТаблицыИзмерений", []).append(table_name)
    return {"name": name, "properties": props, "attributes": attrs}


def _external_ds_descriptor_context(rel_parts: Tuple[str, ...], md_kind: str) -> Dict[str, str] | None:
    if len(rel_parts) < 4 or rel_parts[0] != "ExternalDataSources":
        return None
    owner_name = rel_parts[1]
    parent_folder = rel_parts[-2]
    expected_tag = EXTERNAL_DS_CONTAINER_TAG_BY_FOLDER.get(parent_folder)
    if expected_tag and md_kind == expected_tag:
        cube_name = ""
        if parent_folder == "DimensionTables" and "Cubes" in rel_parts:
            cube_idx = rel_parts.index("Cubes")
            if cube_idx + 1 < len(rel_parts):
                cube_name = rel_parts[cube_idx + 1]
        return {
            "kind": "external_ds_container",
            "owner_name": owner_name,
            "container_tag": md_kind,
            "cube_name": cube_name,
        }
    return None


def _external_ds_container_form_context(rel_parts: Tuple[str, ...]) -> Dict[str, str] | None:
    if len(rel_parts) < 6 or rel_parts[0] != "ExternalDataSources" or rel_parts[-2] != "Forms":
        return None
    selected: Tuple[int, str, str] | None = None
    for folder_name, container_tag in EXTERNAL_DS_CONTAINER_TAG_BY_FOLDER.items():
        if folder_name not in rel_parts:
            continue
        idx = rel_parts.index(folder_name)
        if idx + 1 < len(rel_parts) and (selected is None or idx > selected[0]):
            selected = (idx, folder_name, container_tag)
    if selected:
        idx, folder_name, container_tag = selected
        cube_name = ""
        if folder_name == "DimensionTables" and "Cubes" in rel_parts:
            cube_idx = rel_parts.index("Cubes")
            if cube_idx + 1 < len(rel_parts):
                cube_name = rel_parts[cube_idx + 1]
        return {
            "kind": "external_ds_container_form",
            "owner_name": rel_parts[1],
            "container_name": rel_parts[idx + 1],
            "container_tag": container_tag,
            "cube_name": cube_name,
        }
    return None


def _external_ds_container_context_by_name(rel_parts: Tuple[str, ...], container_name: str) -> Dict[str, str] | None:
    if rel_parts[0] != "ExternalDataSources":
        return None
    for folder_name, container_tag in EXTERNAL_DS_CONTAINER_TAG_BY_FOLDER.items():
        if folder_name not in rel_parts:
            continue
        idx = rel_parts.index(folder_name)
        if idx + 1 < len(rel_parts) and rel_parts[idx + 1] == container_name:
            return {
                "owner_name": rel_parts[1],
                "container_name": container_name,
                "container_tag": container_tag,
            }
    return None


def _parse_characteristics(props_elem: ET.Element | None) -> List[Dict[str, Any]]:
    ch = None
    for prop in _children(props_elem):
        if _local_name(prop.tag) == "Characteristics":
            ch = prop
            break
    out = []
    for idx, item in enumerate(_children(ch, "Characteristic")):
        props: Dict[str, Any] = {"Индекс": str(idx)}
        types = _first_child(item, "CharacteristicTypes")
        values = _first_child(item, "CharacteristicValues")
        if types is not None:
            props["ВидыХарактеристик"] = _format_metadata_path(_attr_by_local_name(types, "from"))
            props["ПолеКлюча"] = _blank_if_minus_one(_last_metadata_part(_child_text(types, "KeyField")))
            props["ПолеОтбораВидов"] = _blank_if_minus_one(_last_metadata_part(_child_text(types, "TypesFilterField")))
            filter_value = _child_text(types, "TypesFilterValue")
            props["ЗначениеОтбораВидов"] = "" if "." in filter_value else _translate_value(filter_value)
            props["ПолеПутиКДанным"] = _blank_value(_data_path_field(types))
        if values is not None:
            props["ЗначенияХарактеристик"] = _format_metadata_path(_attr_by_local_name(values, "from"))
            props["ПолеОбъекта"] = _blank_if_minus_one(_last_metadata_part(_child_text(values, "ObjectField")))
            props["ПолеВида"] = _blank_if_minus_one(_last_metadata_part(_child_text(values, "TypeField")))
            props["ПолеЗначения"] = _blank_if_minus_one(_last_metadata_part(_child_text(values, "ValueField")))
            props["ПолеИспользованияМножественныхЗначений"] = _blank_if_minus_one(_child_text(values, "MultipleValuesUseField"))
            props["ПолеКлючаМножественныхЗначений"] = _blank_if_minus_one(_child_text(values, "MultipleValuesKeyField"))
            props["ПолеПорядкаМножественныхЗначений"] = _blank_if_minus_one(_child_text(values, "MultipleValuesOrderField"))
        if len(props) > 1:
            out.append(props)
    return out


def _metadata_reference_values(node: ET.Element | None) -> List[str]:
    result = []
    for child in node.iter() if node is not None else []:
        if child is node:
            continue
        if _local_name(child.tag) not in {"Type", "TypeSet", "Item", "Value", "Metadata", "Field"}:
            continue
        value = _text(child) or _attr_by_local_name(child, "ref") or _attr_by_local_name(child, "value")
        if value:
            formatted = _format_metadata_path(value)
            if formatted not in result:
                result.append(formatted)
    return result


def _subsystem_chain(rel_parts: Tuple[str, ...], object_name: str) -> List[str]:
    """Путь подсистемы по пути дескриптора: имя после каждого `Subsystems` плюс имя объекта.

    `Subsystems/A.xml` → [A]; `Subsystems/A/Subsystems/B.xml` → [A, B];
    `Subsystems/A/Subsystems/A.xml` → [A, A]. Раньше последний сегмент
    отбрасывался, если совпадал с именем родителя, — вложенная подсистема получала путь
    родителя, а в графе оба дескриптора склеивались в один узел. Так же путь строят
    `incremental.xml_walker._resolve_owner_object_qn` и загрузчик расширений.
    """
    chain = []
    dirs = rel_parts[:-1]
    for idx in range(1, len(dirs)):
        if dirs[idx - 1] == "Subsystems" and dirs[idx] != "Subsystems":
            chain.append(dirs[idx])
    chain.append(object_name)
    return chain


def parse_descriptor(payload: Tuple[str, ...]) -> Dict[str, Any]:
    if len(payload) == 2:
        root_s, path_s = payload
        materialize_standard_attrs = False
        preserve_listed_standard_attrs = False
    elif len(payload) == 3:
        root_s, path_s, materialize_standard_attrs = payload
        preserve_listed_standard_attrs = False
    else:
        root_s, path_s, materialize_standard_attrs, preserve_listed_standard_attrs = payload
    root = Path(root_s)
    path = Path(path_s)
    rel = path.relative_to(root)
    rel_parts = rel.parts
    # Filtering of payload XML (Ext/Form.xml, Predefined.xml, Rights.xml, ConfigDumpInfo.xml)
    # happens upstream in CodeFileIndexer.classify; the worker trusts its input.

    is_child_form = len(rel_parts) >= 4 and rel_parts[-2] == "Forms"
    is_child_template = len(rel_parts) >= 4 and rel_parts[-2] == "Templates"
    is_child_command = len(rel_parts) >= 4 and rel_parts[-2] == "Commands"

    if not _is_probably_xml(path):
        if is_child_form:
            form_name = path.stem
            return {
                "kind": "child_form",
                "owner_folder": rel_parts[0],
                "owner_name": rel_parts[1],
                "name": form_name,
                "properties": {
                    "Имя": form_name,
                    "Синоним": _camel_synonym(form_name),
                    "Комментарий": "",
                    "ТипФормы": "Управляемая",
                    "ПринадлежностьОбъекта": "Собственный",
                    "ОбъектРасширяемойКонфигурации": "",
                    "ВключатьСправкуВСодержание": "Ложь",
                    "НазначенияИспользования": "ПриложениеПлатформы, ПриложениеМобильнойПлатформы",
                },
                "warning": "not_xml",
            }
        return {"kind": "error", "rel_path": str(rel), "error": "not_xml"}

    try:
        parsed_root = _parse_xml_root(path)
        xml_version = parsed_root.attrib.get("version", "")
        md = _metadata_elem(parsed_root)
        md_kind = _local_name(md.tag)
        props_elem = _first_child(md, "Properties")
        basic_props = _properties_ru(props_elem)
        name = _name_from_props(basic_props, path.stem)

        external_form_context = _external_ds_container_form_context(rel_parts)
        if external_form_context:
            props = _normalized_properties(md, type_key="form")
            return {
                "kind": "external_ds_container_form",
                "owner_name": external_form_context["owner_name"],
                "container_name": external_form_context["container_name"],
                "container_tag": external_form_context["container_tag"],
                "cube_name": external_form_context.get("cube_name") or "",
                "name": name,
                "properties": props,
            }

        if is_child_form:
            props = _normalized_properties(md, type_key="form")
            return {
                "kind": "child_form",
                "owner_folder": rel_parts[0],
                "owner_name": rel_parts[1],
                "name": name,
                "properties": props,
            }
        if is_child_template:
            props = _normalized_properties(md, type_key="template")
            return {
                "kind": "child_template",
                "owner_folder": rel_parts[0],
                "owner_name": rel_parts[1],
                "name": name,
                "properties": props,
            }
        if is_child_command:
            props = _normalized_properties(md, type_key="command")
            return {
                "kind": "child_command",
                "owner_folder": rel_parts[0],
                "owner_name": rel_parts[1],
                "name": name,
                "properties": props,
            }

        if len(rel_parts) == 1 and rel_parts[0] == "Configuration.xml":
            props = _normalize_configuration_props(_normalized_properties(md))
            return {"kind": "configuration", "name": name, "properties": props}

        external_ds_context = _external_ds_descriptor_context(rel_parts, md_kind)
        if external_ds_context:
            if external_ds_context["kind"] == "external_ds_container":
                container = _parse_external_ds_container(md, name, md_kind)
                if external_ds_context.get("cube_name"):
                    container["properties"]["РодительскийКуб"] = external_ds_context["cube_name"]
                return {
                    "kind": "external_ds_container",
                    "owner_name": external_ds_context["owner_name"],
                    "container_tag": external_ds_context["container_tag"],
                    "name": container["name"],
                    "properties": container["properties"],
                    "attributes": container["attributes"],
                }

        folder = rel_parts[0]
        category = FOLDER_TO_RU_CATEGORY.get(folder)
        if not category:
            return {"kind": "skip", "reason": f"unknown_folder_{folder}"}
        props = _normalized_properties(md, folder=folder)
        if folder == "ChartsOfAccounts":
            props.setdefault("АвтоПорядокПоКоду", "Истина")
            if "ДлинаПорядка" not in props and props.get("ДлинаКода") not in (None, "", []):
                props["ДлинаПорядка"] = props["ДлинаКода"]
        _merge_structural_props(props, basic_props, ["Движения"])
        characteristic_schemes = props.pop("Характеристики", None)
        if not isinstance(characteristic_schemes, list):
            characteristic_schemes = _parse_characteristics(props_elem)
        for scheme in characteristic_schemes:
            if isinstance(scheme, dict) and str(scheme.get("ЗначениеОтбораВидов", "")).startswith("Справочник.ВидыКонтактнойИнформации."):
                scheme["ЗначениеОтбораВидов"] = ""

        obj: Dict[str, Any] = {
            "kind": "object",
            "folder": folder,
            "category": category,
            "name": name,
            "properties": props,
            "attributes": [],
            "tabulars": [],
            "forms": [],
            "commands": [],
            "layouts": [],
            "resources": [],
            "dimensions": [],
            "account_flags": [],
            "subconto_flags": [],
            "enum_values": [],
            "url_templates": [],
            "journal_graphs": [],
            "characteristic_schemes": characteristic_schemes,
        }

        standard_attributes = _parse_standard_attributes(
            props_elem,
            name,
            folder,
            include_empty=materialize_standard_attrs,
            owner_props=props,
            preserve_listed_standard_attrs=preserve_listed_standard_attrs,
        )
        obj["attributes"].extend(standard_attributes)
        # _enrich_standard_attrs only enriches already-kept standard attrs (type,
        # length, owner-derived props); it never creates absent ones. Creation of
        # absent attrs stays gated by include_empty=materialize_standard_attrs above.
        if materialize_standard_attrs or preserve_listed_standard_attrs:
            _enrich_standard_attrs(obj["attributes"], obj["properties"], folder=folder, owner_name=name, basic_props=basic_props)
        _enrich_owner_standard_attr(obj["attributes"], obj["properties"], basic_props)

        if category == "Подсистемы":
            obj["properties"]["ПутьПодсистемы"] = _subsystem_chain(rel_parts, name)
            if len(obj["properties"]["ПутьПодсистемы"]) > 1:
                obj["properties"]["РодительскаяПодсистема"] = obj["properties"]["ПутьПодсистемы"][-2]

        child_objects = _first_child(md, "ChildObjects")
        for child in _children(child_objects):
            child_name = _local_name(child.tag)
            if child_name == "Attribute":
                parsed_attr = _parse_attribute(child, "", "attribute", owner_folder=folder, xml_version=xml_version)
                if folder == "InformationRegisters" and parsed_attr.get("name") == "Владелец":
                    parsed_attr.setdefault("properties", {})["Стандартный"] = True
                obj["attributes"].append(parsed_attr)
            elif child_name == "AddressingAttribute":
                parsed_attr = _parse_attribute(child, "", "addressing_attribute", owner_folder=folder, xml_version=xml_version)
                parsed_attr.setdefault("properties", {})["ЭтоРеквизитАдресации"] = True
                obj["attributes"].append(parsed_attr)
            elif child_name == "TabularSection":
                obj["tabulars"].append(_parse_tabular(child, "", owner_folder=folder, owner_name=name, xml_version=xml_version))
            elif child_name == "Resource":
                obj["resources"].append(_parse_attribute(child, "", "resource", owner_folder=folder, xml_version=xml_version))
            elif child_name == "Dimension":
                obj["dimensions"].append(_parse_attribute(child, "", "dimension", owner_folder=folder, xml_version=xml_version))
            elif child_name == "AccountingFlag":
                obj["account_flags"].append(_parse_attribute(child, "", "accounting_flag", owner_folder=folder, xml_version=xml_version))
            elif child_name == "ExtDimensionAccountingFlag":
                obj["subconto_flags"].append(_parse_attribute(child, "", "ext_dimension_accounting_flag", owner_folder=folder, xml_version=xml_version))
            elif child_name == "EnumValue":
                ev_props = _normalized_properties(child, type_key="enum_value")
                obj["enum_values"].append({"name": _name_from_props(ev_props, ""), "properties": ev_props})
            elif child_name == "Command":
                cmd_props = _normalized_properties(child, type_key="command")
                obj["commands"].append({"name": _name_from_props(cmd_props, ""), "properties": cmd_props})
            elif child_name == "Form":
                form_name = _text(child)
                if form_name:
                    obj["forms"].append({"name": form_name, "properties": {"Имя": form_name}})
            elif child_name == "Template":
                lay_name = _text(child)
                if lay_name:
                    obj["layouts"].append({"name": lay_name, "properties": {"Имя": lay_name}})
            elif child_name == "URLTemplate":
                url_template = _parse_url_template(child)
                if url_template:
                    obj["url_templates"].append(url_template)
            elif folder == "ExternalDataSources" and child_name in {"Table", "Cube"}:
                child_ref = _text(child)
                if child_ref:
                    obj["tabulars"].append({
                        "name": child_ref,
                        "properties": {
                            "Имя": child_ref,
                            "ВидОбъектаВнешнегоИсточникаДанных": child_name,
                        },
                        "attributes": [],
                    })
            elif folder == "ExternalDataSources" and child_name == "Function":
                function = _parse_external_ds_function(child)
                if function.get("name"):
                    obj["attributes"].append(function)
            elif child_name == "Column":
                graph_props = _normalized_properties(child, type_key="document_journal_column")
                references = _metadata_reference_values(_first_child(_first_child(child, "Properties"), "References"))
                if references:
                    graph_props["Ссылки"] = _legacy_property_value("Ссылки", references)
                graph_name = _name_from_props(graph_props, "")
                if graph_name:
                    obj["journal_graphs"].append({"Имя": graph_name, **graph_props})

        if obj["url_templates"]:
            obj["properties"]["ШаблоныURL"] = obj["url_templates"]
        if obj["journal_graphs"]:
            obj["properties"]["ГрафыЖурнала"] = obj["journal_graphs"]
        return obj
    except Exception as exc:
        return {"kind": "error", "rel_path": str(rel), "error": str(exc)}


def _merge_unique_named(existing: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    name = item.get("name")
    if not name:
        return
    for current in existing:
        if current.get("name") == name:
            current_props = current.setdefault("properties", {})
            current_props.update(item.get("properties") or {})
            for child_list_name in ("attributes",):
                for child in item.get(child_list_name) or []:
                    _merge_unique_named(current.setdefault(child_list_name, []), child)
            return
    existing.append(item)


def _is_adopted_props(props: Dict[str, Any] | None) -> bool:
    value = (props or {}).get("ПринадлежностьОбъекта")
    return value in {"Заимствованный", "Adopted"}


OVERLAY_SKIP_PROPS = frozenset({"meta_uuid"})


def _merge_missing_props(
    target: Dict[str, Any],
    source: Dict[str, Any] | None,
    *,
    skip_keys: frozenset = OVERLAY_SKIP_PROPS,
) -> int:
    if not source:
        return 0
    applied = 0
    for key, value in source.items():
        if key in skip_keys:
            continue
        if key not in target:
            target[key] = value
            applied += 1
    return applied


def _index_named(items: List[Any], attr: str = "name") -> Dict[str, Any]:
    return {getattr(item, attr): item for item in items if getattr(item, attr, None)}


def _ensure_owner_attribute_from_object_props(obj: MetadataObject) -> bool:
    owners_val = obj.properties.get("Владельцы")
    if owners_val is None or any(attr.name == "Владелец" for attr in obj.attributes):
        return False
    # This runs only from apply_extension_base_overlay (after ownership stamping in
    # finish()), exclusively for adopted objects. Inherit the parent's ownership so the
    # new attr is not later mistaken for own — a generic re-stamp to 'Собственный' would
    # exclude it from ADOPTED_FROM (builder keeps NULL or != 'Собственный').
    ownership = obj.properties.get("ПринадлежностьОбъекта") or "Заимствованный"
    owner_attr = Attribute(name="Владелец", properties={"Стандартный": True, "ПринадлежностьОбъекта": ownership})
    owner_attr.properties["Тип"] = owners_val if isinstance(owners_val, list) else [owners_val]
    usage_val = obj.properties.get("ИспользованиеПодчинения")
    if usage_val is not None:
        owner_attr.properties["ИспользованиеПодчинения"] = usage_val
    obj.attributes.append(owner_attr)
    return True


def _merge_existing_children_from_base(ext_items: List[Any], base_items: List[Any]) -> int:
    applied = 0
    base_by_name = _index_named(base_items)
    for ext_item in ext_items:
        if not _is_adopted_props(getattr(ext_item, "properties", None)):
            continue
        base_item = base_by_name.get(getattr(ext_item, "name", ""))
        if base_item is None:
            continue
        applied += _merge_missing_props(ext_item.properties, getattr(base_item, "properties", None))
    return applied


def _merge_existing_tabulars_from_base(ext_obj: MetadataObject, base_obj: MetadataObject) -> int:
    applied = 0
    base_tabs = _index_named(base_obj.tabular_parts)
    for ext_tab in ext_obj.tabular_parts:
        if not _is_adopted_props(ext_tab.properties):
            continue
        base_tab = base_tabs.get(ext_tab.name)
        if base_tab is None:
            continue
        applied += _merge_missing_props(ext_tab.properties, base_tab.properties)
        applied += _merge_existing_children_from_base(ext_tab.attributes, base_tab.attributes)
    return applied


def _merge_named_dict_list(target_props: Dict[str, Any], source_props: Dict[str, Any], prop_name: str) -> int:
    target_items = target_props.get(prop_name)
    source_items = source_props.get(prop_name)
    if not isinstance(target_items, list) or not isinstance(source_items, list):
        return 0
    source_by_name = {
        item.get("Имя") or item.get("name"): item
        for item in source_items
        if isinstance(item, dict) and (item.get("Имя") or item.get("name"))
    }
    applied = 0
    for target_item in target_items:
        if not isinstance(target_item, dict):
            continue
        item_name = target_item.get("Имя") or target_item.get("name")
        source_item = source_by_name.get(item_name)
        if not isinstance(source_item, dict):
            continue
        applied += _merge_missing_props(target_item, source_item)
        if prop_name == "ШаблоныURL":
            target_item.setdefault("Свойства", {})
            source_item.setdefault("Свойства", {})
            if isinstance(target_item["Свойства"], dict) and isinstance(source_item["Свойства"], dict):
                applied += _merge_missing_props(target_item["Свойства"], source_item["Свойства"])
            applied += _merge_url_methods(target_item, source_item)
    return applied


def _merge_url_methods(target_template: Dict[str, Any], source_template: Dict[str, Any]) -> int:
    target_methods = target_template.get("Методы")
    source_methods = source_template.get("Методы")
    if not isinstance(target_methods, list) or not isinstance(source_methods, list):
        return 0
    source_by_name = {
        item.get("Имя"): item
        for item in source_methods
        if isinstance(item, dict) and item.get("Имя")
    }
    applied = 0
    for target_method in target_methods:
        if not isinstance(target_method, dict):
            continue
        source_method = source_by_name.get(target_method.get("Имя"))
        if not isinstance(source_method, dict):
            continue
        target_method.setdefault("Свойства", {})
        source_method.setdefault("Свойства", {})
        if isinstance(target_method["Свойства"], dict) and isinstance(source_method["Свойства"], dict):
            applied += _merge_missing_props(target_method["Свойства"], source_method["Свойства"])
    return applied


def _apply_txt_like_empty_defaults(obj: MetadataObject) -> int:
    applied = 0
    for graph in obj.properties.get("ГрафыЖурнала") or []:
        if isinstance(graph, dict):
            if "Синоним" not in graph:
                graph["Синоним"] = ""
                applied += 1
            if "Ссылки" not in graph:
                graph["Ссылки"] = ""
                applied += 1
    for template in obj.properties.get("ШаблоныURL") or []:
        if not isinstance(template, dict):
            continue
        props = template.setdefault("Свойства", {})
        if isinstance(props, dict) and "Синоним" not in props:
            props["Синоним"] = ""
            applied += 1
        for method in template.get("Методы") or []:
            if not isinstance(method, dict):
                continue
            m_props = method.setdefault("Свойства", {})
            if isinstance(m_props, dict) and "Синоним" not in m_props:
                m_props["Синоним"] = ""
                applied += 1
    return applied


def apply_extension_base_overlay(ext_config: Configuration, base_config: Configuration) -> Dict[str, int]:
    """XML-only overlay: fill missing properties on adopted extension objects from
    the base configuration. Mutates ext_config in place and returns stats.

    Applies only to objects whose ПринадлежностьОбъекта is Заимствованный/Adopted,
    matched to base by (category.name, object.name). Existing extension properties
    are never overwritten — only absent ones are merged. Own objects and the base
    configuration are left untouched.
    """
    base_objects = {
        (category.name, obj.name): obj
        for category in base_config.categories
        for obj in category.metadata_objects
    }
    stats = Counter()
    for key in INHERITED_EXTENSION_CONFIGURATION_PROPS:
        if key not in ext_config.properties and base_config.properties.get(key) not in (None, "", []):
            ext_config.properties[key] = base_config.properties[key]
            stats["configuration_inherited_props"] += 1
    for ext_category in ext_config.categories:
        for ext_obj in ext_category.metadata_objects:
            if not _is_adopted_props(ext_obj.properties):
                continue
            base_obj = base_objects.get((ext_category.name, ext_obj.name))
            if base_obj is None:
                stats["missing_base_object"] += 1
                continue
            stats["objects"] += 1
            stats["object_props"] += _merge_missing_props(ext_obj.properties, base_obj.properties)
            stats["object_attrs_props"] += _merge_existing_children_from_base(ext_obj.attributes, base_obj.attributes)
            stats["resources_props"] += _merge_existing_children_from_base(ext_obj.resources, base_obj.resources)
            stats["dimensions_props"] += _merge_existing_children_from_base(ext_obj.dimensions, base_obj.dimensions)
            stats["account_flags_props"] += _merge_existing_children_from_base(ext_obj.account_flags, base_obj.account_flags)
            stats["subconto_flags_props"] += _merge_existing_children_from_base(ext_obj.subconto_flags, base_obj.subconto_flags)
            stats["forms_props"] += _merge_existing_children_from_base(ext_obj.forms, base_obj.forms)
            stats["commands_props"] += _merge_existing_children_from_base(ext_obj.commands, base_obj.commands)
            stats["layouts_props"] += _merge_existing_children_from_base(ext_obj.layouts, base_obj.layouts)
            stats["enum_values_props"] += _merge_existing_children_from_base(ext_obj.enum_values, base_obj.enum_values)
            stats["tabular_props"] += _merge_existing_tabulars_from_base(ext_obj, base_obj)
            stats["journal_graph_props"] += _merge_named_dict_list(ext_obj.properties, base_obj.properties, "ГрафыЖурнала")
            stats["url_template_props"] += _merge_named_dict_list(ext_obj.properties, base_obj.properties, "ШаблоныURL")
            stats["txt_like_empty_defaults"] += _apply_txt_like_empty_defaults(ext_obj)
            if _ensure_owner_attribute_from_object_props(ext_obj):
                stats["owner_attrs_added"] += 1
    return dict(stats)


def _to_configuration(project_name: str, parsed: List[Dict[str, Any]], root: Path) -> Tuple[Configuration, Dict[str, Any]]:
    config_name = root.name
    config_props: Dict[str, Any] = {}
    objects: Dict[Tuple[str, str], Dict[str, Any]] = {}
    warnings = Counter()
    errors = []

    def ensure_object(category: str, owner_name: str) -> Dict[str, Any]:
        key = (category, owner_name)
        return objects.setdefault(key, {
            "category": category,
            "name": owner_name,
            "properties": {"Имя": owner_name},
            "attributes": [], "tabulars": [], "forms": [], "commands": [],
            "layouts": [], "resources": [], "dimensions": [],
            "account_flags": [], "subconto_flags": [], "enum_values": [],
            "characteristic_schemes": [],
        })

    for item in parsed:
        kind = item.get("kind")
        if kind == "configuration":
            config_name = item.get("name") or config_name
            config_props = item.get("properties") or {}
            continue
        if kind == "error":
            errors.append(item)
            continue
        if kind == "child_form":
            category = FOLDER_TO_RU_CATEGORY.get(item.get("owner_folder"))
            owner_name = item.get("owner_name")
            if category and owner_name:
                key = (category, owner_name)
                obj = objects.setdefault(key, {
                    "category": category,
                    "name": owner_name,
                    "properties": {"Имя": owner_name},
                    "attributes": [], "tabulars": [], "forms": [], "commands": [],
                    "layouts": [], "resources": [], "dimensions": [],
                    "account_flags": [], "subconto_flags": [], "enum_values": [],
                    "characteristic_schemes": [],
                })
                _merge_unique_named(obj["forms"], item)
                if item.get("warning"):
                    warnings[item["warning"]] += 1
            continue
        if kind == "child_template":
            category = FOLDER_TO_RU_CATEGORY.get(item.get("owner_folder"))
            owner_name = item.get("owner_name")
            if category and owner_name:
                key = (category, owner_name)
                obj = objects.setdefault(key, {
                    "category": category,
                    "name": owner_name,
                    "properties": {"Имя": owner_name},
                    "attributes": [], "tabulars": [], "forms": [], "commands": [],
                    "layouts": [], "resources": [], "dimensions": [],
                    "account_flags": [], "subconto_flags": [], "enum_values": [],
                    "characteristic_schemes": [],
                })
                _merge_unique_named(obj["layouts"], item)
            continue
        if kind == "child_command":
            category = FOLDER_TO_RU_CATEGORY.get(item.get("owner_folder"))
            owner_name = item.get("owner_name")
            if category and owner_name:
                key = (category, owner_name)
                obj = objects.setdefault(key, {
                    "category": category,
                    "name": owner_name,
                    "properties": {"Имя": owner_name},
                    "attributes": [], "tabulars": [], "forms": [], "commands": [],
                    "layouts": [], "resources": [], "dimensions": [],
                    "account_flags": [], "subconto_flags": [], "enum_values": [],
                    "characteristic_schemes": [],
                })
                _merge_unique_named(obj["commands"], item)
            continue
        if kind == "external_ds_container":
            owner_name = item.get("owner_name")
            if owner_name:
                obj = ensure_object("ВнешниеИсточникиДанных", owner_name)
                tabular = {
                    "name": item.get("name") or "",
                    "properties": item.get("properties") or {},
                    "attributes": item.get("attributes") or [],
                }
                if item.get("container_tag"):
                    tabular["properties"].setdefault("ВидОбъектаВнешнегоИсточникаДанных", item["container_tag"])
                _merge_unique_named(obj["tabulars"], tabular)
            continue
        if kind == "external_ds_container_form":
            owner_name = item.get("owner_name")
            container_name = item.get("container_name")
            if owner_name and container_name:
                obj = ensure_object("ВнешниеИсточникиДанных", owner_name)
                tabular = next((tab for tab in obj["tabulars"] if tab.get("name") == container_name), None)
                if tabular is None:
                    tabular = {
                        "name": container_name,
                        "properties": {
                            "Имя": container_name,
                            "ВидОбъектаВнешнегоИсточникаДанных": item.get("container_tag") or "",
                        },
                        "attributes": [],
                    }
                    if item.get("cube_name"):
                        tabular["properties"]["РодительскийКуб"] = item["cube_name"]
                    obj["tabulars"].append(tabular)
                forms = tabular.setdefault("properties", {}).setdefault("Формы", [])
                form_name = item.get("name")
                if form_name and form_name not in forms:
                    forms.append(form_name)
            continue
        if kind == "external_ds_function":
            owner_name = item.get("owner_name")
            if owner_name:
                obj = ensure_object("ВнешниеИсточникиДанных", owner_name)
                _merge_unique_named(obj["attributes"], {
                    "name": item.get("name") or "",
                    "properties": item.get("properties") or {},
                })
            continue
        if kind == "object":
            key_name = item["name"]
            if item["category"] == "Подсистемы":
                chain = (item.get("properties") or {}).get("ПутьПодсистемы")
                if isinstance(chain, list) and chain:
                    key_name = "/".join(str(part) for part in chain)
            key = (item["category"], key_name)
            obj = objects.setdefault(key, item)
            if obj is not item:
                obj["properties"].update(item.get("properties") or {})
                for list_name in ["attributes", "tabulars", "forms", "commands", "layouts", "resources", "dimensions", "account_flags", "subconto_flags", "enum_values", "characteristic_schemes"]:
                    for child in item.get(list_name) or []:
                        if list_name == "characteristic_schemes":
                            obj[list_name].append(child)
                        else:
                            _merge_unique_named(obj[list_name], child)

    cfg = Configuration(name=config_name, file_path=root / "Configuration.xml", properties=config_props)
    by_category: Dict[str, MetadataCategory] = {}
    for (category, _name), item in sorted(objects.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cat = by_category.setdefault(category, MetadataCategory(name=category))
        mo = MetadataObject(name=item["name"], properties=item.get("properties") or {})
        for child in item.get("attributes") or []:
            mo.attributes.append(Attribute(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("resources") or []:
            mo.resources.append(Attribute(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("dimensions") or []:
            mo.dimensions.append(Attribute(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("account_flags") or []:
            mo.account_flags.append(Attribute(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("subconto_flags") or []:
            mo.subconto_flags.append(Attribute(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("commands") or []:
            mo.commands.append(Command(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("forms") or []:
            mo.forms.append(Form(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("layouts") or []:
            mo.layouts.append(Layout(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("enum_values") or []:
            mo.enum_values.append(EnumValue(name=child["name"], properties=child.get("properties") or {}))
        for child in item.get("characteristic_schemes") or []:
            mo.characteristic_schemes.append(child)
        for child in item.get("tabulars") or []:
            tab = TabularPart(name=child["name"], properties=child.get("properties") or {})
            for attr in child.get("attributes") or []:
                tab.attributes.append(Attribute(name=attr["name"], properties=attr.get("properties") or {}))
            mo.tabular_parts.append(tab)
        cat.metadata_objects.append(mo)
    for cat_name in sorted(by_category):
        cfg.categories.append(by_category[cat_name])

    diagnostics = {
        "object_count": len(objects),
        "category_counts": {name: len(cat.metadata_objects) for name, cat in sorted(by_category.items())},
        "warnings": dict(warnings),
        "errors": errors[:20],
        "error_count": len(errors),
    }
    return cfg, diagnostics


def _set_default_ownership_recursively(cfg: Configuration) -> int:
    """For extension mode: stamp ПринадлежностьОбъекта='Собственный' on every
    node whose XML did not carry <ObjectBelonging>.

    Required by ExtensionRelationshipsBuilder.build_adopted_from (NULL is
    interpreted as 'adopted'). Must cover EVERY label that the builder iterates,
    including dict-backed nodes that rows_builder.py promotes to separate graph
    nodes: Characteristic, UrlTemplate, UrlMethod, JournalGraph.
    """
    own = "Собственный"

    def stamp(props) -> int:
        if not isinstance(props, dict):
            return 0
        if "ПринадлежностьОбъекта" not in props or props.get("ПринадлежностьОбъекта") in (None, ""):
            props["ПринадлежностьОбъекта"] = own
            return 1
        return 0

    touched = 0
    for category in cfg.categories:
        for mo in category.metadata_objects:
            touched += stamp(mo.properties)
            for child_list_name in (
                "attributes", "resources", "dimensions", "account_flags",
                "subconto_flags", "commands", "forms", "layouts", "enum_values",
            ):
                for child in getattr(mo, child_list_name, []) or []:
                    touched += stamp(getattr(child, "properties", None))
            for tab in mo.tabular_parts or []:
                touched += stamp(tab.properties)
                for attr in tab.attributes or []:
                    touched += stamp(attr.properties)

            # Dict-backed nodes — promoted to separate labels by rows_builder.py
            # and explicitly handled by ExtensionRelationshipsBuilder.

            # Characteristic — each scheme dict acts as props directly
            for sch in getattr(mo, "characteristic_schemes", None) or []:
                touched += stamp(sch)

            # UrlTemplate / UrlMethod — nested under mo.properties["ШаблоныURL"]
            for tmpl in (mo.properties.get("ШаблоныURL") or []) if isinstance(mo.properties, dict) else []:
                if not isinstance(tmpl, dict):
                    continue
                touched += stamp(tmpl.get("Свойства"))
                for method in tmpl.get("Методы") or []:
                    if isinstance(method, dict):
                        touched += stamp(method.get("Свойства"))

            # JournalGraph — entries under mo.properties["ГрафыЖурнала"] are
            # serialized verbatim as graph node properties.
            for graph in (mo.properties.get("ГрафыЖурнала") or []) if isinstance(mo.properties, dict) else []:
                if isinstance(graph, dict):
                    touched += stamp(graph)

    return touched


def _finalize_configuration(
    parsed: List[Dict[str, Any]],
    kind_counts: "Counter",
    root: Path,
    project_name: str,
    *,
    is_extension: bool,
    extension_marker: str,
) -> List[Configuration]:
    """Assemble parsed descriptors into a Configuration, apply extension
    post-processing, and log. Shared by parse_files and XmlMetadataParseSession.
    """
    if not parsed:
        logger.warning("XmlMetadataParser: no descriptors parsed for root=%s", root)
        return []

    config, diagnostics = _to_configuration(project_name, parsed, root)

    if diagnostics.get("error_count", 0):
        logger.warning(
            "XmlMetadataParser: %d parse errors in %s (first samples: %s)",
            diagnostics["error_count"], root, diagnostics.get("errors") or [],
        )

    if is_extension:
        base_name = config.name or root.name
        if not base_name.endswith(extension_marker):
            config.name = f"{base_name}{extension_marker}"
        stamped = _set_default_ownership_recursively(config)
        logger.info(
            "XmlMetadataParser[ext]: name=%s, stamped ПринадлежностьОбъекта=Собственный on %d nodes",
            config.name, stamped,
        )
    else:
        for key, value in BASE_CONFIGURATION_DEFAULTS.items():
            config.properties.setdefault(key, value)
        logger.info(
            "XmlMetadataParser: name=%s, categories=%d, objects=%d",
            config.name,
            len(config.categories),
            sum(len(c.metadata_objects) for c in config.categories),
        )

    return [config]


class XmlMetadataParseSession:
    """Streaming variant of XmlMetadataParser: submit descriptor files one by one
    (during a single os.walk) into a ProcessPool, then assemble in finish().

    Used by the orchestrator/extensions to overlap the code-tree walk with XML
    descriptor parsing. ``parse_descriptor`` must stay a top-level function
    (Windows spawn requirement) — the session only fans work out to it.
    """

    def __init__(
        self,
        *,
        workers: int,
        materialize_standard_attrs: bool,
        preserve_listed_standard_attrs: bool,
        root: Path,
        project_name: str = "",
    ):
        self.root = Path(root)
        self.project_name = project_name
        self.materialize = bool(materialize_standard_attrs)
        self.preserve = bool(preserve_listed_standard_attrs)
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx)
        self._futures: List[Any] = []
        self._submitted = 0
        self._closed = False

    def submit(self, path: Path) -> None:
        self._futures.append(
            self._executor.submit(
                parse_descriptor, (str(self.root), str(path), self.materialize, self.preserve)
            )
        )
        self._submitted += 1

    def finish(
        self,
        *,
        is_extension: bool = False,
        extension_marker: str = "$ext$",
    ) -> List[Configuration]:
        try:
            parsed: List[Dict[str, Any]] = []
            kind_counts: Counter = Counter()
            # Worker-level exceptions (BrokenProcessPool, ImportError, NameError,
            # etc.) are FATAL — they mean a broken parsing environment or a lost
            # batch of descriptors, not one bad object. Letting future.result()
            # raise here (the pre-session behavior) aborts the load instead of
            # silently assembling an incomplete Configuration. Per-file parse
            # errors are already returned as {"kind": "error"} items and surfaced
            # via _to_configuration diagnostics. The finally below shuts the pool
            # down (cancel_futures=True) before the exception propagates.
            for future in as_completed(self._futures):
                item = future.result()
                kind_counts[item.get("kind", "?")] += 1
                parsed.append(item)
            logger.info(
                "XmlMetadataParseSession: submitted=%d parsed=%d kinds=%s",
                self._submitted, len(parsed), dict(sorted(kind_counts.items())),
            )
            return _finalize_configuration(
                parsed, kind_counts, self.root, self.project_name,
                is_extension=is_extension, extension_marker=extension_marker,
            )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if not self._closed:
            self._executor.shutdown(cancel_futures=True)
            self._closed = True


class XmlMetadataParser:
    """
    Public XML metadata source. Consumes a ready list of XML descriptor files
    (typically ``CodeFileIndex.metadata_xml_files``) and returns
    ``list[Configuration]`` shaped identically to the TXT pipeline output.

    The parser never walks the filesystem; classification of payload XML
    (Ext/Form.xml, Predefined.xml, Rights.xml, ConfigDumpInfo.xml) is the
    responsibility of the caller (``CodeFileIndexer``).
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        materialize_standard_attributes: bool = False,
        preserve_listed_standard_attributes: bool = True,
    ):
        if workers is None or workers <= 0:
            try:
                from config import settings as _settings  # local import to avoid cycles
                workers = (
                    getattr(_settings, "XML_PROCESS_WORKERS", None)
                    or getattr(_settings, "PROCESS_WORKERS", None)
                    or 4
                )
            except Exception:
                workers = 4
        self.workers = int(workers)
        self.materialize_standard_attributes = bool(materialize_standard_attributes)
        self.preserve_listed_standard_attributes = bool(preserve_listed_standard_attributes)

    def parse_files(
        self,
        files: Sequence[Path],
        root: Path,
        *,
        is_extension: bool = False,
        extension_marker: str = "$ext$",
    ) -> List[Configuration]:
        files = [Path(p) for p in files]
        if not files:
            logger.warning("XmlMetadataParser.parse_files: empty file list for root=%s", root)
            return []

        session = XmlMetadataParseSession(
            workers=self.workers,
            materialize_standard_attrs=self.materialize_standard_attributes,
            preserve_listed_standard_attrs=self.preserve_listed_standard_attributes,
            root=Path(root),
            project_name="",  # project_name is applied later by ConfigLoader
        )
        for p in files:
            session.submit(p)
        return session.finish(is_extension=is_extension, extension_marker=extension_marker)
