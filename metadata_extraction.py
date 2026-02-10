import depthai as dai

from stereo import setup_stereo


def extract_stereo_settings_from_node(stereo_node):
    """
    Extract a JSON-serializable dict of all stereo initialConfig parameters
    from an already-configured StereoDepth node. Call this after setup_stereo()
    and before pipeline.start() to avoid blocking.
    """
    if stereo_node is None:
        return None
    try:
        return _extract_config_dict(stereo_node.initialConfig)
    except Exception:
        return None


def _is_enum_instance(obj):
    return hasattr(obj, "name") and hasattr(obj, "value") and not isinstance(obj, type)


def _format_value(value):
    if value is None:
        return None
    if _is_enum_instance(value):
        return value.name
    if isinstance(value, type) and hasattr(value, "__members__"):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return [str(x) for x in value[:20]] + [f"... ({len(value)} total)"]
        return [_format_value(x) for x in value if _format_value(x) is not None]
    return str(value)


def _extract_config_dict(config_obj, visited=None, max_depth=10):
    if visited is None:
        visited = set()
    result = {}
    if config_obj is None or max_depth <= 0:
        return result
    obj_id = id(config_obj)
    if obj_id in visited:
        return result
    visited.add(obj_id)
    try:
        for attr_name in dir(config_obj):
            if attr_name.startswith("_") or attr_name in (
                "get", "set", "serialize", "deserialize",
                "serializeToJson", "deserializeFromJson", "__members__", "__class__",
            ):
                continue
            try:
                attr_value = getattr(config_obj, attr_name)
                if callable(attr_value):
                    continue
                if isinstance(attr_value, type) and hasattr(attr_value, "__members__"):
                    continue
                if _is_enum_instance(attr_value):
                    result[attr_name] = _format_value(attr_value)
                elif isinstance(attr_value, (str, int, float, bool, type(None))):
                    formatted = _format_value(attr_value)
                    if formatted is not None:
                        result[attr_name] = formatted
                elif isinstance(attr_value, (list, tuple)):
                    if len(attr_value) == 0:
                        result[attr_name] = []
                    else:
                        formatted = _format_value(attr_value)
                        if formatted is not None:
                            result[attr_name] = formatted
                elif hasattr(attr_value, "__class__"):
                    tname = type(attr_value).__name__
                    if tname in ("method", "function", "builtin_function_or_method", "type"):
                        continue
                    try:
                        nested = _extract_config_dict(attr_value, visited.copy(), max_depth - 1)
                        if nested:
                            result[attr_name] = nested
                        else:
                            formatted = _format_value(attr_value)
                            if formatted is not None:
                                result[attr_name] = formatted
                    except (AttributeError, TypeError, RecursionError):
                        formatted = _format_value(attr_value)
                        if formatted is not None:
                            result[attr_name] = formatted
            except (AttributeError, TypeError):
                continue
    finally:
        visited.discard(obj_id)
    return result


def extract_stereo_settings(settings, platform=None):
    """
    Build a StereoDepth node with the same settings as the capture pipeline,
    then return a JSON-serializable dict of all stereo initialConfig parameters.
    Returns None if extraction fails. When no device is available, pass platform
    (e.g. dai.Platform.RVC4); defaults to RVC4.
    """
    if platform is None:
        platform = dai.Platform.RVC4
    try:
        pipeline = dai.Pipeline()
        stereo = setup_stereo(pipeline, settings, platform)
        return _extract_config_dict(stereo.initialConfig)
    except Exception:
        return None
