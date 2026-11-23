"""Turn code-flavored interview text into words a voice can read naturally.

Coding prompts are written for the eye: `deps: list[tuple[str, str]]` is clear on screen
and gibberish aloud. The on-screen text is never changed; only the copy handed to the
speech engine is rewritten, the way an interviewer would read it out
("deps, a list of pairs of strings").
"""

from __future__ import annotations

import re

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_APPOSITIVE_RE = re.compile(r", (?:an? |returning )")
_GENERIC_ANGLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*<.*>$", re.DOTALL)
_IDENT_PIECE_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOTTED_NAME = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_CALL_RE = re.compile(rf"^({_DOTTED_NAME})\((.*)\)\s*(?:->\s*(.+))?$", re.DOTALL)
_ANNOTATED_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S.*)$", re.DOTALL)
_FIELD_RE = re.compile(rf"^({_DOTTED_NAME}):$")
_EXPR_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|->|=>|>=|<=|==|!=|\S")
# A snake_case name standing alone in a sentence; file names, paths and addresses are left as written.
_SNAKE_CALL_RE = re.compile(
    r"(?<![\w./@-])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![A-Za-z0-9_])(\(\))?(?![./@-]\w)"
)
_BIG_O_RE = re.compile(r"\bO\(([^()]{1,24})\)")
_PROSE_OPERATORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s*>=\s*"), " greater than or equal to "),
    (re.compile(r"\s*<=\s*"), " less than or equal to "),
    (re.compile(r"\s*!=\s*"), " not equal to "),
    (re.compile(r"\s*==\s*"), " equal to "),
    (re.compile(r"\s*[-=]>\s*"), " to "),
)

# Lowercase name parts engineers say letter by letter ("now M S", "user I D").
_SPELLED_PARTS = frozenset(
    {"ms", "ttl", "lru", "id", "kv", "url", "uri", "api", "db", "sse", "ui", "io", "os", "ip", "sql", "rpc", "cpu"}
)
# All-caps names that are spoken as words rather than spelled.
_UPPERCASE_WORDS = frozenset({"JSON", "REST", "YAML", "NULL", "TRUE", "FALSE", "NONE", "POST", "HEAD"})

_EXPR_WORDS = {
    "-": "minus",
    "+": "plus",
    "*": "times",
    "/": "divided by",
    "%": "modulo",
    "=": "equals",
    "==": "equals",
    "!=": "is not equal to",
    ">=": "is greater than or equal to",
    "<=": "is less than or equal to",
    ">": "is greater than",
    "<": "is less than",
    "->": "returns",
    "=>": "returns",
    ".": "dot",
    ",": ",",
}

# Type name -> (singular, plural) as spoken.
_TYPE_WORDS: dict[str, tuple[str, str]] = {
    "str": ("string", "strings"),
    "string": ("string", "strings"),
    "int": ("integer", "integers"),
    "float": ("float", "floats"),
    "number": ("number", "numbers"),
    "bool": ("boolean", "booleans"),
    "boolean": ("boolean", "booleans"),
    "bytes": ("byte string", "byte strings"),
    "dict": ("dictionary", "dictionaries"),
    "list": ("list", "lists"),
    "tuple": ("tuple", "tuples"),
    "set": ("set", "sets"),
    "any": ("value", "values"),
    "object": ("object", "objects"),
}
_TYPE_ALIASES = {
    "sequence": "list",
    "iterable": "list",
    "array": "list",
    "mapping": "dict",
    "map": "dict",
    "record": "dict",
    "frozenset": "set",
}
_KNOWN_TYPE_NAMES = frozenset({*_TYPE_WORDS, *_TYPE_ALIASES, "optional", "none", "|"})
_GROUP_NAMES = {2: ("pair", "pairs"), 3: ("triple", "triples")}

_TypeNode = tuple[str, list["_TypeNode"]]
# Deeper nesting than anyone says aloud is read token by token instead of being parsed.
_MAX_NESTING = 8
_INTERVAL_ENDS = {"(": "exclusive", "[": "inclusive", ")": "exclusive", "]": "inclusive"}


def _spell(letters: str) -> str:
    return " ".join(letters.upper())


def _speak_part(part: str) -> str:
    if part.isupper() and len(part) > 1:
        if part in _UPPERCASE_WORDS:
            return part.lower()
        vowels = sum(ch in "AEIOU" for ch in part)
        return _spell(part) if len(part) <= 4 or vowels * 3 < len(part) else part.lower()
    if part.lower() in _SPELLED_PARTS:
        return _spell(part)
    return part


def speak_identifier(name: str) -> str:
    """`schedule_layers` -> "schedule layers", `SSEParser` -> "S S E Parser", `now_ms` -> "now M S"."""
    words: list[str] = []
    for chunk in name.replace(".", " dot ").replace("_", " ").split():
        if chunk == "dot":
            words.append(chunk)
            continue
        words.extend(_speak_part(part) for part in _IDENT_PIECE_RE.findall(chunk) or [chunk])
    return " ".join(words)


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == separator and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2 and not any("," in item for item in items):
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _article(phrase: str) -> str:
    return f"{'an' if phrase[:1].lower() in 'aeiou' else 'a'} {phrase}"


def _parse_type(text: str, depth: int = 0) -> _TypeNode | None:
    text = text.strip()
    array_levels = 0
    while text.endswith("[]") and len(text) > 2:
        text = text[:-2].rstrip()
        array_levels += 1
    if depth + array_levels > _MAX_NESTING:
        return None
    if array_levels:
        element = _parse_type(text, depth + array_levels)
        for _ in range(array_levels):
            element = ("list", [element]) if element else None
        return element
    if _GENERIC_ANGLE_RE.match(text):
        text = text.replace("<", "[").replace(">", "]")
    union = _split_top_level(text, "|")
    if len(union) > 1:
        members = [_parse_type(member, depth + 1) for member in union]
        return ("|", members) if all(members) else None  # type: ignore[arg-type]
    if text == "...":
        return ("...", [])
    match = re.fullmatch(rf"({_DOTTED_NAME})(?:\[(.*)\])?", text, re.DOTALL)
    if not match:
        return None
    name = match.group(1).rsplit(".", 1)[-1]
    if match.group(2) is None:
        return (name, [])
    args = [_parse_type(arg, depth + 1) for arg in _split_top_level(match.group(2))]
    return (name, args) if args and all(args) else None  # type: ignore[arg-type]


def _speak_type(node: _TypeNode, *, plural: bool = False) -> str:
    """Speak a parsed type as a bare noun phrase ("list of strings"); callers add the article."""
    name, args = node
    key = _TYPE_ALIASES.get(name.lower(), name.lower())
    if name == "|":
        concrete = [arg for arg in args if arg[0].lower() != "none"]
        spoken = " or ".join(_speak_type(arg, plural=plural) for arg in concrete or args)
        return f"optional {spoken}" if len(concrete) < len(args) and concrete else spoken
    if key == "none":
        return "nothing"
    if key == "optional" and len(args) == 1:
        return f"optional {_speak_type(args[0], plural=plural)}"
    if key == "tuple" and args:
        members = [arg for arg in args if arg[0] != "..."]
        same = all(member == members[0] for member in members)
        if same and len(members) == len(args) and len(args) in _GROUP_NAMES:
            return f"{_GROUP_NAMES[len(args)][plural]} of {_speak_type(members[0], plural=True)}"
        if same:
            return f"{_TYPE_WORDS['tuple'][plural]} of {_speak_type(members[0], plural=True)}"
        return f"{_TYPE_WORDS['tuple'][plural]} of {_join_and([_speak_type(m) for m in members])}"
    if key == "dict" and len(args) == 2:
        return (
            f"{_TYPE_WORDS['dict'][plural]} from {_speak_type(args[0], plural=True)} "
            f"to {_speak_type(args[1], plural=True)}"
        )
    words = _TYPE_WORDS.get(key)
    base = words[plural] if words else speak_identifier(name) + ("s" if plural else "")
    if args:
        return f"{base} of {_join_and([_speak_type(arg, plural=words is not None) for arg in args])}"
    return base


def _speak_expression(text: str) -> str:
    """Read an expression left to right. Brackets are spoken, so `a * (b + c)` and `a * b + c` differ."""
    words: list[str] = []
    previous = ""
    for token in _EXPR_TOKEN_RE.findall(text):
        follows_value = bool(previous) and (previous in ")]" or bool(_IDENT_RE.fullmatch(previous)))
        if _IDENT_RE.fullmatch(token):
            words.append(speak_identifier(token))
        elif token[0].isdigit():
            words.append(token)
        elif token == "[" and follows_value:
            words.append("at")
        elif token == "(":
            words.append("of" if follows_value else "the quantity")
        elif token == ")":
            words.append(",")
        elif token in _EXPR_WORDS:
            words.append(_EXPR_WORDS[token])
        previous = token
    spoken = re.sub(r"\s+,", ",", " ".join(words))
    return re.sub(r",{2,}", ",", spoken).strip(" ,")


def _speak_argument(arg: str, depth: int) -> str:
    arg = arg.lstrip("*")
    annotated = _ANNOTATED_RE.match(arg)
    if annotated:
        parsed = _parse_type(annotated.group(2).split("=", 1)[0])
        if parsed:
            return f"{speak_identifier(annotated.group(1))}, {_article(_speak_type(parsed))}"
    return _speak_code(arg.split("=", 1)[0] if _IDENT_RE.match(arg) else arg, depth)


def _speak_interval(code: str) -> str | None:
    """`(a, b]` -> "from a, exclusive, to b, inclusive". Only mixed brackets are surely an interval."""
    if {code[0], code[-1]} not in ({"(", "]"}, {"[", ")"}):
        return None
    ends = _split_top_level(code[1:-1])
    if len(ends) != 2:
        return None
    return (
        f"from {_speak_expression(ends[0])}, {_INTERVAL_ENDS[code[0]]}, "
        f"to {_speak_expression(ends[1])}, {_INTERVAL_ENDS[code[-1]]}"
    )


def speak_code(code: str) -> str:
    """Read one inline code span aloud: a name, a call ("get of key"), a type, or an expression."""
    return _speak_code(code, 0)


def _speak_code(code: str, depth: int) -> str:
    code = code.strip()
    if not code:
        return ""
    if depth > _MAX_NESTING:
        return _speak_expression(code)
    if code == ":":
        return "a colon"
    if _IDENT_RE.fullmatch(code):
        return speak_identifier(code)
    field = _FIELD_RE.match(code)
    if field:
        return speak_identifier(field.group(1))

    call = _CALL_RE.match(code)
    if call and code.count("(") == code.count(")"):
        spoken = speak_identifier(call.group(1))
        args = [_speak_argument(arg, depth + 1) for arg in _split_top_level(call.group(2))]
        if args:
            spoken += f" of {_join_and(args)}"
        returns = _parse_type(call.group(3)) if call.group(3) else None
        if returns:
            spoken += f", returning {_article(_speak_type(returns))}"
        elif call.group(3):
            spoken += f", returning {_speak_expression(call.group(3))}"
        return spoken

    annotated = _ANNOTATED_RE.match(code)
    if annotated:
        parsed = _parse_type(annotated.group(2))
        if parsed:
            return f"{speak_identifier(annotated.group(1))}, {_article(_speak_type(parsed))}"

    bare_type = _parse_type(code)
    if bare_type and (bare_type[0].lower() in _KNOWN_TYPE_NAMES or _GENERIC_ANGLE_RE.match(code)):
        return _article(_speak_type(bare_type))

    if code[0] in "([" and code[-1] in ")]":
        return _speak_interval(code) or ", ".join(
            _speak_expression(part) for part in _split_top_level(code[1:-1])
        )
    return _speak_expression(code)


def _speak_big_o(match: re.Match[str]) -> str:
    inner = match.group(1).replace("^2", " squared").replace("^3", " cubed").replace("*", " times ")
    return f"O of {' '.join(inner.split())}"


def speak_prose(text: str) -> str:
    """Fix the code that leaks into ordinary sentences: `O(1)`, `>=`, and snake_case names."""
    text = _BIG_O_RE.sub(_speak_big_o, text)
    for pattern, spoken in _PROSE_OPERATORS:
        text = pattern.sub(spoken, text)
    return _SNAKE_CALL_RE.sub(lambda m: speak_identifier(m.group(1)), text)


def speak_text(text: str) -> str:
    """Rewrite a whole message for the voice: inline code spans as speech, the rest as prose."""
    pieces = _INLINE_CODE_RE.split(text)
    spoken: list[str] = []
    for idx, piece in enumerate(pieces):
        if idx % 2 == 0:
            spoken.append(speak_prose(piece))
            continue
        phrase = speak_code(piece)
        following = pieces[idx + 1] if idx + 1 < len(pieces) else ""
        # "deps, a list of pairs of strings" needs a closing pause before the sentence carries on.
        if _APPOSITIVE_RE.search(phrase) and re.match(r"\s+\w", following):
            phrase += ","
        spoken.append(phrase)
    return "".join(spoken)
