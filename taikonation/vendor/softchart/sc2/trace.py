"""Command-trace parser: authored TJA -> exact rational command stream (slot+ S0).

A trace is the timing fragment of one TJA course represented as an ordered
command stream over the authored-bar rational lattice (SLOTPLUS_DESIGN.md
section 2 and 3): events are NOTE(pos, type), TEMPO(value), DELAY(dur) and
MEASURE(num/den); tempo/meter are persistent state that only events change,
and every bar-internal position is an exact ``fractions.Fraction`` whose
denominator is the authored row's note-character count.  Nothing here goes
through floating point on the position axis; absolute times are derived
quantities (``trace_to_times``) computed with exact rational arithmetic and
converted to float only at the boundary.

Schema
------
Parse layer (lossless per ``#START``/``#END`` play section):

``ChartCommand``
    ``offset``  number of note characters of the bar emitted before the
                command line (0 .. len(digits)),
    ``name``    command name, uppercased, without the leading ``#``,
    ``args``    verbatim argument text.
``ChartBar``
    ``digits``    the bar's note characters with commas removed (may be
                  empty: an authored empty measure),
    ``commands``  list of ChartCommand in source order (all commands are
                  kept here, including GOGO/SCROLL, for losslessness).
``Chart``
    ``headers``           dict of raw header text (keys uppercased),
    ``course``            canonical course name (easy/normal/hard/oni/ura),
    ``level``             int level (0 when absent),
    ``bpm_text``          authored header BPM, verbatim ("120" when absent),
    ``offset_text``       authored header OFFSET, verbatim ("0" when absent);
                          bar 0 starts at absolute time ``-OFFSET`` seconds,
    ``start_args``        arguments of ``#START`` ("" or "P1"/"P2"),
    ``bars``              list of ChartBar,
    ``trailing_commands`` commands after the last completed bar that precede
                          ``#END`` without any note characters.

Trace layer (the command stream itself):

``NoteEvent``   ``bar``, ``pos`` (Fraction in [0, 1)), ``type`` (the raw TJA
                note character, '1'..'9', kept verbatim; see NOTE_CHAR_CLASS).
``TempoEvent``  ``bar``, ``pos``, ``bpm`` (float), ``authored`` (verbatim
                directive argument), and the (k, p) tempo-octave
                factorisation of SLOTPLUS_DESIGN section 2.2: the authored
                beat period ``60/bpm`` equals ``p * 2**k`` with ``p`` inside
                one octave window.
``DelayEvent``  ``bar``, ``pos``, ``seconds`` (signed float), ``authored``.
``MeterEvent``  ``bar``, ``num``, ``den`` (ints), ``authored_num``,
                ``authored_den`` (verbatim).  Meter events sit at bar
                boundaries only (grammar, SLOTPLUS_DESIGN section 3.2).
``BarInfo``     ``index``, ``denom`` (authored row note-character count; 0
                for an empty measure), ``num``/``den`` (meter state effective
                for the bar) and ``k`` (tempo-octave notation state at the
                bar start) -- the per-bar notation coset of section 2.3.
``Trace``       ``course``, ``level``, ``bpm``/``bpm_authored`` (initial
                tempo state), ``offset``/``offset_authored``, ``bars`` (list
                of BarInfo) and ``events`` (the ordered command stream).

Semantics rulings (documented because raw TJA leaves them implicit):

- ``#BPMCHANGE``/``#DELAY`` are legal mid-bar; their position is the exact
  fraction ``written_digits / row_length``.  ``#MEASURE`` is bar-boundary
  only (SLOTPLUS_DESIGN section 3.2): a mid-row ``#MEASURE`` takes effect at
  the next bar's start (the current bar's length is already determined by
  its meter).  Commands written after every note character of a bar but
  before its terminating comma are timing-equivalent to the next bar's
  position 0 and are carried there; if no next bar exists, one empty bar is
  synthesized so the trace stays a lossless verbatim carrier.
- ``#GOGOSTART``/``#GOGOEND``/``#SCROLL`` and other non-timing commands are
  excluded from the trace (SLOTPLUS_DESIGN section 3.1); they remain in the
  Chart layer and never affect reconstructed times.
- Branch commands (``#BRANCHSTART`` etc.) fail closed with
  ``BranchUnsupportedError``: a branched section is three parallel note
  streams and cannot be represented as one digit stream (upstream
  taiko-1000-parsed is already flattened to a single stream).
- ``#DELAY`` is a signed second value added to the timeline at its position.
  A negative delay that makes reconstructed note times non-monotonic raises
  ``NonMonotonicTimeError`` (fail-closed, SLOTPLUS_DESIGN section 3.1).
- Tempo-octave labels: the song's initial k is pinned from the header BPM by
  the canonical window ``[PERIOD_LO, PERIOD_HI)``; each subsequent TEMPO
  event keeps ``p`` log-nearest to the previous canonical period, so k moves
  only on ~x2/x0.5 notation switches (SLOTPLUS_DESIGN section 2.2).  (k, p)
  are annotation: time reconstruction always uses the authored value.

Round-trip contract (SLOTPLUS_DESIGN section 8-S0): the trace carries every
authored TEMPO/MEASURE/DELAY value verbatim and reproduces every note's
absolute time exactly (rational arithmetic; decimal header/directive text is
parsed as exact decimal fractions).
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

# Canonical tempo-octave window (draft values per SLOTPLUS_DESIGN section
# 2.2: [1.0, 2.0) seconds ~ BPM (120, 240]; re-derived and frozen before S1).
PERIOD_LO = 1.0
PERIOD_HI = 2.0

# Timing-bearing TJA commands (the trace's command alphabet besides NOTE).
TIMING_COMMANDS = ("BPMCHANGE", "MEASURE", "DELAY")
# Branch structure diverges into parallel streams: fail closed.
BRANCH_COMMANDS = ("BRANCHSTART", "BRANCHEND", "N", "E", "M")

# Raw TJA note characters -> canonical note classes (vocab.NOTE_CLASSES).
# '9' (kusudama) is balloon-class, mirroring vocab.NOTE_TYPE_MAP.
NOTE_CHAR_CLASS = {"1": "don", "2": "ka", "3": "don_big", "4": "ka_big",
                   "5": "roll", "6": "roll_big", "7": "balloon", "8": "end",
                   "9": "balloon"}

_COURSE_ALIASES = {
    "easy": "easy", "0": "easy",
    "normal": "normal", "1": "normal",
    "hard": "hard", "2": "hard",
    "oni": "oni", "3": "oni",
    "edit": "ura", "ura": "ura", "4": "ura",
}


class TraceError(ValueError):
    """A TJA construct the trace layer refuses to guess about."""


class BranchUnsupportedError(TraceError):
    """#BRANCHSTART sections fail closed (parallel streams, upstream-flattened)."""


class NonMonotonicTimeError(TraceError):
    """A negative #DELAY produced non-monotonic note times (fail-closed)."""


@dataclass(frozen=True)
class ChartCommand:
    offset: int
    name: str
    args: str


@dataclass
class ChartBar:
    digits: str
    commands: list


@dataclass
class Chart:
    headers: dict
    course: str
    level: int
    bpm_text: str
    offset_text: str
    start_args: str
    bars: list
    trailing_commands: list = field(default_factory=list)


@dataclass(frozen=True)
class NoteEvent:
    bar: int
    pos: Fraction
    type: str


@dataclass(frozen=True)
class TempoEvent:
    bar: int
    pos: Fraction
    bpm: float
    authored: str
    k: int
    p: float


@dataclass(frozen=True)
class DelayEvent:
    bar: int
    pos: Fraction
    seconds: float
    authored: str


@dataclass(frozen=True)
class MeterEvent:
    bar: int
    num: int
    den: int
    authored_num: str
    authored_den: str


@dataclass(frozen=True)
class BarInfo:
    index: int
    denom: int
    num: int
    den: int
    k: int


@dataclass
class Trace:
    course: str
    level: int
    bpm: float
    bpm_authored: str
    offset: float
    offset_authored: str
    bars: list
    events: list


def note_class(char):
    """Canonical note class for a raw TJA note character (None if unknown)."""
    return NOTE_CHAR_CLASS.get(char)


def _fraction_text(text):
    """Exact rational value of authored numeric text (decimal-exact when the
    text is decimal; otherwise the exact rational of the parsed float)."""
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return Fraction(float(text))


def _build_chart(headers, start_args, bars, trailing):
    course_text = headers.get("COURSE", "Oni").strip()
    course = _COURSE_ALIASES.get(course_text.lower(), course_text.lower())
    level_text = headers.get("LEVEL", "").strip()
    try:
        level = int(float(level_text)) if level_text else 0
    except ValueError:
        level = 0
    bpm_text = headers.get("BPM", "").strip() or "120"
    offset_text = headers.get("OFFSET", "").strip() or "0"
    return Chart(dict(headers), course, level, bpm_text, offset_text,
                 start_args, bars, trailing)


def parse_tja(text):
    """Parse TJA text into one Chart per #START/#END play section.

    Headers accumulate across the file (TJA semantics: BPM/OFFSET persist
    unless re-declared before the next course section).  Comments (``//``)
    are stripped everywhere; a measure may span several lines and a line may
    hold several comma-terminated measures.
    """
    charts = []
    headers = {}
    in_body = False
    start_args = ""
    bars = commands = None
    digits = ""
    for raw in text.lstrip("\ufeff").splitlines():
        cut = raw.find("//")
        line = (raw[:cut] if cut >= 0 else raw).strip()
        if not line:
            continue
        if not in_body:
            if line.startswith("#"):
                head = line[1:].split(None, 1)
                if head and head[0].upper() == "START":
                    in_body = True
                    start_args = head[1].strip() if len(head) > 1 else ""
                    bars, digits, commands = [], "", []
                # other pre-body commands carry no timing; ignored
            elif ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().upper()] = value.strip()
            continue
        if line.startswith("#"):
            head = line[1:].split(None, 1)
            name = head[0].upper() if head else ""
            args = head[1].strip() if len(head) > 1 else ""
            if name == "END":
                trailing = []
                if digits:
                    bars.append(ChartBar(digits, commands))
                else:
                    trailing = commands
                charts.append(_build_chart(headers, start_args, bars, trailing))
                in_body = False
                bars, digits, commands = None, "", None
                continue
            if name in BRANCH_COMMANDS:
                raise BranchUnsupportedError(
                    f"branch command #{name} is not supported (fail-closed)")
            commands.append(ChartCommand(len(digits), name, args))
            continue
        for ch in line:
            if ch == ",":
                bars.append(ChartBar(digits, commands))
                digits, commands = "", []
            elif ch.isspace():
                continue
            elif ch.isascii() and ch.isalnum():
                digits += ch
            else:
                raise TraceError(
                    f"unexpected character {ch!r} in note line {raw!r}")
    if in_body:
        raise TraceError("unterminated #START section (missing #END)")
    return charts


def parse_tja_course(text, course):
    """Return the first Chart whose canonical course matches ``course``."""
    wanted = _COURSE_ALIASES.get(str(course).strip().lower(),
                                 str(course).strip().lower())
    for chart in parse_tja(text):
        if chart.course == wanted:
            return chart
    raise TraceError(f"no {wanted!r} course section in this TJA")


def chart_to_trace(chart):
    """Convert a parsed Chart into its command Trace.

    Positions are exact ``Fraction(written_digits, row_length)``; authored
    numeric text is kept verbatim on every event.  Non-timing commands are
    excluded (they cannot move any note time); branch charts never reach
    here (parse_tja fails closed).
    """
    try:
        bpm0 = float(chart.bpm_text)
    except ValueError:
        raise TraceError(f"unparseable header BPM {chart.bpm_text!r}") from None
    if not math.isfinite(bpm0) or bpm0 <= 0:
        raise TraceError(f"header BPM must be finite and positive: {chart.bpm_text!r}")
    try:
        offset0 = float(chart.offset_text)
    except ValueError:
        raise TraceError(
            f"unparseable header OFFSET {chart.offset_text!r}") from None
    if not math.isfinite(offset0):
        raise TraceError(f"header OFFSET must be finite: {chart.offset_text!r}")

    state = {"num": 4, "den": 4}
    period0 = 60.0 / bpm0
    # Pin the song's initial octave label from the canonical window
    # (SLOTPLUS_DESIGN section 2.2); with PERIOD_LO = 1.0 this is floor(log2).
    state["k"] = math.floor(math.log2(period0 / PERIOD_LO))
    state["p"] = period0 / 2.0 ** state["k"]

    events = []
    bars = []

    def emit(bar_index, pos, cmd):
        if cmd.name == "BPMCHANGE":
            try:
                value = float(cmd.args)
            except ValueError:
                raise TraceError(
                    f"bar {bar_index}: unparseable #BPMCHANGE {cmd.args!r}"
                ) from None
            if not math.isfinite(value) or value <= 0:
                raise TraceError(
                    f"bar {bar_index}: #BPMCHANGE must be finite and positive: "
                    f"{cmd.args!r}")
            period = 60.0 / value
            # k moves only when the new period leaves the +-half-octave band
            # around the previous canonical period p (~x2/x0.5 switches).
            k = round(math.log2(period / state["p"]))
            state["k"], state["p"] = k, period / 2.0 ** k
            events.append(TempoEvent(bar_index, pos, value, cmd.args, k,
                                     state["p"]))
        elif cmd.name == "MEASURE":
            parts = cmd.args.split("/")
            if len(parts) != 2:
                raise TraceError(
                    f"bar {bar_index}: unparseable #MEASURE {cmd.args!r}")
            authored_num, authored_den = parts[0].strip(), parts[1].strip()
            try:
                num, den = int(authored_num), int(authored_den)
            except ValueError:
                raise TraceError(
                    f"bar {bar_index}: #MEASURE terms must be integers: "
                    f"{cmd.args!r}") from None
            if num <= 0 or den <= 0:
                raise TraceError(
                    f"bar {bar_index}: #MEASURE terms must be positive: "
                    f"{cmd.args!r}")
            state["num"], state["den"] = num, den
            events.append(MeterEvent(bar_index, num, den, authored_num,
                                     authored_den))
        else:  # DELAY
            try:
                seconds = float(cmd.args)
            except ValueError:
                raise TraceError(
                    f"bar {bar_index}: unparseable #DELAY {cmd.args!r}"
                ) from None
            if not math.isfinite(seconds):
                raise TraceError(
                    f"bar {bar_index}: #DELAY must be finite: {cmd.args!r}")
            events.append(DelayEvent(bar_index, pos, seconds, cmd.args))

    carry = []  # timing commands owed to the next bar's position 0
    source_bars = list(chart.bars)
    trailing = [c for c in chart.trailing_commands if c.name in TIMING_COMMANDS]
    index = 0
    while index < len(source_bars) or carry or trailing:
        if index < len(source_bars):
            bar = source_bars[index]
        else:
            # Synthesized empty bar hosting end-of-chart carried commands so
            # their authored values stay in the trace (module docstring).
            bar = ChartBar("", [])
        length = len(bar.digits)
        local, carry_out = [], []
        for cmd in bar.commands:
            if cmd.name not in TIMING_COMMANDS:
                continue
            if (cmd.name == "MEASURE" and cmd.offset > 0) or \
                    (length > 0 and cmd.offset >= length):
                carry_out.append(cmd)
            else:
                local.append(cmd)
        if index >= len(source_bars):
            local = list(trailing)
            trailing = []
        head = carry + [c for c in local if c.offset == 0]
        tail = [c for c in local if c.offset > 0]
        carry = carry_out
        for cmd in head:
            emit(index, Fraction(0), cmd)
        bars.append(BarInfo(index, length, state["num"], state["den"],
                            state["k"]))
        ti = 0
        for i, ch in enumerate(bar.digits):
            while ti < len(tail) and tail[ti].offset <= i:
                emit(index, Fraction(tail[ti].offset, length), tail[ti])
                ti += 1
            if ch != "0":
                events.append(NoteEvent(index, Fraction(i, length), ch))
        # remaining tail commands all had offset == length and were carried
        index += 1

    return Trace(chart.course, chart.level, bpm0, chart.bpm_text,
                 offset0, chart.offset_text, bars, events)


def _iter_timed(trace):
    """Walk the command stream with exact rational time.

    Yields ("edge", bar_index, t) at every bar start plus the terminal edge
    (bar_index == len(bars)), and ("note", NoteEvent, t) for every note, all
    in stream order.  The terminal edge is evaluated from trace state, never
    extrapolated (SLOTPLUS_DESIGN section 8-S0).
    """
    t = -_fraction_text(trace.offset_authored)
    tau = Fraction(60) / _fraction_text(trace.bpm_authored)
    num, den = 4, 4
    bar_i = 0
    cursor = Fraction(0)
    n_bars = len(trace.bars)
    yield "edge", 0, t
    for ev in trace.events:
        while bar_i < ev.bar:
            t += (1 - cursor) * Fraction(4 * num, den) * tau
            bar_i += 1
            cursor = Fraction(0)
            yield "edge", bar_i, t
        if isinstance(ev, MeterEvent):
            if cursor != 0:
                raise TraceError(
                    f"bar {ev.bar}: MEASURE away from the bar boundary is "
                    "outside the grammar")
            num, den = ev.num, ev.den
            continue
        if ev.pos != cursor:
            t += (ev.pos - cursor) * Fraction(4 * num, den) * tau
            cursor = ev.pos
        if isinstance(ev, TempoEvent):
            tau = Fraction(60) / _fraction_text(ev.authored)
        elif isinstance(ev, DelayEvent):
            t += _fraction_text(ev.authored)
        else:
            yield "note", ev, t
    while bar_i < n_bars:
        t += (1 - cursor) * Fraction(4 * num, den) * tau
        bar_i += 1
        cursor = Fraction(0)
        yield "edge", bar_i, t


def trace_to_times(trace, exact=False, check_monotonic=True):
    """Reconstruct every note's absolute time (seconds) and type from a trace.

    Returns a list of (time, type_char) in stream order; times are floats, or
    exact Fractions with ``exact=True``.  Raises NonMonotonicTimeError when a
    negative #DELAY makes note times non-increasing (fail-closed).
    """
    out = []
    last = None
    for kind, payload, t in _iter_timed(trace):
        if kind != "note":
            continue
        if check_monotonic and last is not None and t <= last:
            raise NonMonotonicTimeError(
                f"non-monotonic note time at bar {payload.bar} "
                f"pos {payload.pos}: {float(t)} after {float(last)}")
        last = t
        out.append((t if exact else float(t), payload.type))
    return out


def bar_edge_times(trace, exact=False):
    """Absolute times of every bar start plus the terminal edge.

    Returns ``len(trace.bars) + 1`` values; the terminal edge comes from
    evaluating trace state across the final bar, not from extrapolation.
    """
    edges = []
    for kind, _payload, t in _iter_timed(trace):
        if kind == "edge":
            edges.append(t if exact else float(t))
    return edges
