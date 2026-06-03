FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")

FIELD_TO_ID = {name: idx for idx, name in enumerate(FIELDS)}
ID_TO_FIELD = {idx: name for name, idx in FIELD_TO_ID.items()}


def parse_fields(values):
    if values is None:
        return list(FIELDS)
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]
    unknown = [v for v in values if v not in FIELD_TO_ID]
    if unknown:
        raise ValueError(f"Unknown field strength(s): {unknown}. Expected one of {FIELDS}.")
    return list(values)

