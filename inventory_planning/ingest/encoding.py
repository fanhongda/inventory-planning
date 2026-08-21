"""
Reading and writing text exports without mangling non-ASCII names.

Two defects live here, and they are opposites of each other.

**Writing.** `DataFrame.to_csv` defaults to UTF-8 without a byte-order mark. Excel,
opening a `.csv` with no BOM, does not attempt to detect UTF-8 — it decodes using the
system codepage, which on a Chinese Windows install is GBK. A supplier named
深圳市华强电子 is written correctly, read back as 娣卞湷甯傚崕寮虹數瀛, and every report
the planner opens shows that instead. Nothing in the pipeline is wrong; the file is
right and unreadable. Writing the BOM fixes it, costs three bytes, and is ignored by
pandas, Python, R and every database loader.

**Reading.** The encoding ladder every loader used was
`utf-8, utf-8-sig, latin-1, cp1252`, tried until one did not raise. That is not
detection: **latin-1 maps all 256 byte values, so it never raises.** Once UTF-8 fails,
latin-1 is guaranteed to "succeed" and the ladder can never reach anything else — a
GBK export decodes to `ÉîÛÚÊÐ»ªÇ¿µç×Ó` with no error, and the mojibake is then the
data. Adding codecs after latin-1 would have changed nothing; the order is the bug.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

# UTF-8 is self-validating: byte sequences that are not UTF-8 overwhelmingly fail to
# decode, so a successful decode is evidence rather than a shrug. These go first and
# a hit is taken without further question.
_SELF_VALIDATING = ("utf-8-sig", "utf-8")

# Last resort, in this order. Neither can fail, so nothing may follow them.
_WESTERN_CODECS = ("cp1252", "latin-1")

# Han, kana, hangul, and the full-width forms that come with them.
_CJK = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\u1100-\u11ff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]"
)

# Han is shared across all four codepages, so finding it says only "some CJK codec".
# Kana and hangul are not, which makes them decisive — but only for the codepage they
# belong to. Kana appearing in *some* decode proves nothing: Big5 reading GBK bytes
# emits 160 of them, and crediting whichever codec produced kana with having found
# Japanese picked Big5 for a Simplified Chinese file. Kana is evidence for CP932; a
# Han codepage emitting it is noise. Half-width katakana is excluded for the same
# reason in reverse — GBK-encoded Chinese read as CP932 produces it in quantity.
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\u1100-\u11ff\uac00-\ud7af]")

# Legacy codepages, tried only once UTF-8 has ruled itself out, in the order their
# installed base warrants. Each is capable of characters ASCII cannot express, and the
# check below asks whether it actually produced any — that is what separates a real
# GBK file from latin-1's willingness to decode anything at all.
#
# The second element is the script that codepage exists to carry and the others do not,
# which is the evidence that identifies it. GB18030 and Big5 have none: both are Han
# codepages, each reads the other's bytes without complaint, and only the count of
# implausible characters separates them.
_CJK_CODECS = (
    ("gb18030", None),
    ("big5", None),
    ("cp932", _KANA),
    ("cp949", _HANGUL),
)

# Below this share of the decoded text being CJK, a CJK codec is not judged to have
# found CJK. Sparse accented letters in a Western file do pair up into a few valid
# GB18030 characters — `é` followed by an ASCII letter is a legal two-byte sequence —
# so a handful of Han characters is not evidence. A genuine Chinese export clears this
# comfortably even when most of the file is numbers and dates.
_CJK_MIN_SHARE = 0.02
_CJK_MIN_COUNT = 4

# What a wrong CJK codec leaves behind. GB18030 covers enough of the byte space to
# decode a Big5 file without ever raising, but the characters it produces are not the
# ones that were written, and the giveaway is that some of them land in the private use
# area — a range no ERP puts a supplier name in. Counting those separates a codec that
# merely *can* decode the bytes from the one that was used to write them.
_IMPLAUSIBLE = re.compile(r"[-�]")


def decode_bytes(raw: bytes, prefer: str = None) -> Tuple[str, str]:
    """
    Decode `raw`, returning `(text, codec)`.

    `prefer` short-circuits the ladder — an explicit answer from someone who knows the
    source always beats a guess.

    Among the CJK codecs every candidate is scored rather than the first hit taken,
    because "decodes without raising" does not distinguish them either: GB18030 reads a
    Big5, Shift-JIS or EUC-KR file end to end and produces the wrong characters. Three
    things rank them, in order: fewest implausible characters, then most characters from
    the script that codepage alone carries, then the declaration order — GB18030 ahead of
    the rest because mainland installs outnumber them.

    The codec is returned, not just the text, so the caller can report it. A guess
    nobody is told about is indistinguishable from a fact — see `describe_choice`.
    """
    if prefer:
        return raw.decode(prefer), prefer

    for codec in _SELF_VALIDATING:
        try:
            return raw.decode(codec), codec
        except UnicodeDecodeError:
            continue

    scored = []
    for order, (codec, signature) in enumerate(_CJK_CODECS):
        try:
            text = raw.decode(codec)
        except UnicodeDecodeError:
            continue
        hits = len(_CJK.findall(text))
        if hits < _CJK_MIN_COUNT or hits / max(len(text), 1) < _CJK_MIN_SHARE:
            continue
        own_script = len(signature.findall(text)) if signature else 0
        if own_script < _CJK_MIN_COUNT:
            own_script = 0       # a stray kana is a coincidence, not a language
        scored.append(
            (len(_IMPLAUSIBLE.findall(text)), -own_script, order, text, codec)
        )
    if scored:
        *_, text, codec = min(scored)
        return text, codec

    for codec in _WESTERN_CODECS:
        try:
            return raw.decode(codec), codec
        except UnicodeDecodeError:
            continue

    raise ValueError("Cannot decode with any known encoding")


def sniff_encoding(path: Union[str, Path], sample_bytes: int = 1 << 20) -> str:
    """
    The codec a file should be read with, judged from its first megabyte.

    A sample rather than the whole file because the decision is about which codepage
    the source system uses, and that does not change halfway down. The sample is cut
    back to a character boundary first: slicing bytes can split a multi-byte character
    and make a perfectly good UTF-8 file look invalid, which would send it to the
    legacy ladder for no reason.
    """
    raw = Path(path).read_bytes()[:sample_bytes]
    # Drop a possibly-truncated trailing character by cutting at the last newline.
    if len(raw) == sample_bytes and b"\n" in raw:
        raw = raw[: raw.rindex(b"\n")]
    return decode_bytes(raw)[1]


def describe_choice(codec: str, source: str) -> Optional[str]:
    """
    A note naming the codec, or None when there is nothing worth saying.

    UTF-8 is silent because it was verified rather than guessed. Everything else is a
    guess that decides what every Chinese, Japanese or accented name in the file says,
    and a guess nobody is told about is indistinguishable from a fact.
    """
    if codec in _SELF_VALIDATING:
        return None
    return (
        f"  ⚠ {source} is not UTF-8; it was read as {codec}. Non-ASCII names — supplier, "
        f"customer, description — depend on that being right. If they look wrong in the "
        f"output, the file's real encoding is something else and it must be re-exported "
        f"as UTF-8 or read with an explicit codec."
    )


# The BOM is the whole point: it is what makes Excel treat the file as UTF-8 instead of
# as the system codepage. Every other reader skips it.
CSV_ENCODING = "utf-8-sig"


def write_csv(frame, path: Union[str, Path], **kwargs) -> Path:
    """
    Write a CSV that Excel opens correctly under any system locale.

    Exists so the encoding cannot be forgotten one call at a time. It was forgotten in
    all eight places that wrote a CSV, which is what a default nobody states looks like.
    """
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", CSV_ENCODING)
    frame.to_csv(path, **kwargs)
    return Path(path)
