"""Token vocabulary for chart sequences.

Sequence layout (decoder):
    [BOS] [COURSE_c] [LEVEL_l] [DENS_d] ([TIME_t] [NOTE_e])* [EOS]
Conditions may be UNK (condition dropout / unspecified at inference).
"""

from dataclasses import dataclass

# --- audio / windowing constants (single source of truth) ---
SR = 22050
N_FFT = 2048
HOP = 256
N_MELS = 128
FPS = SR / HOP  # 86.1328125 frames/sec
WINDOW = 1728  # frames per training window (~20.06 s)
MAX_TGT = 1536  # max decoder length (prefix + events + eos)

SLOTS = 96  # TJA lattice: slots per 4/4 measure (LCM of 16ths/triplets/32nds)
MEAS_MAX = WINDOW // SLOTS  # max whole measures per window in slot mode (18)
SLOTS_PER_BEAT = 24  # meter-general lattice (v3): 3/4 -> 72, 5/4 -> 120 slots
BEATS_MAX = WINDOW // SLOTS_PER_BEAT  # 72 beats of token space per window

COURSES = ["easy", "normal", "hard", "oni", "ura"]
N_LEVELS = 12  # levels clamped to 1..12
N_DENS = 16
DENS_BUCKET_NPS = 0.75  # bucket width in notes/sec

# canonical note classes; parsed TJA note_type strings are mapped onto these
NOTE_CLASSES = ["don", "ka", "don_big", "ka_big", "roll", "roll_big", "balloon", "end"]

# mapping from strings observed in the dataset -> canonical class
# (observed in taiko-1000-parsed: Don, Ka, DonBig, KaBig, Roll, RollBig,
#  Balloon, BalloonAlt, EndOf)
NOTE_TYPE_MAP = {
    "Don": "don",
    "Ka": "ka",
    "DonBig": "don_big",
    "KaBig": "ka_big",
    "Roll": "roll",
    "RollBig": "roll_big",
    "Balloon": "balloon",
    "BalloonAlt": "balloon",  # kusudama, treated as balloon-class span
    "EndOf": "end",
}


@dataclass
class Vocab:
    pad: int = 0
    eos: int = 1
    bos: int = 2
    unk_cond: int = 3

    def __post_init__(self):
        base = 4
        self.course = {c: base + i for i, c in enumerate(COURSES)}
        base += len(COURSES)
        self.level = {l: base + l - 1 for l in range(1, N_LEVELS + 1)}
        base += N_LEVELS
        self.dens = {d: base + d for d in range(N_DENS)}
        base += N_DENS
        self.note = {n: base + i for i, n in enumerate(NOTE_CLASSES)}
        base += len(NOTE_CLASSES)
        self.time0 = base
        base += WINDOW
        # Extended condition tokens append after the original token ids.
        self.sep = base  # separates prev-window context from conditions
        self.sib = base + 1  # marks the sibling-chart (easier course) segment
        self.style = {s: base + 2 + s for s in range(8)}  # charting-intent codes
        self.sync = {s: base + 10 + s for s in range(6)}  # LHL syncopation bands
        # plan-realize tokens: [PLAN] marker + block density (8) + block flags (3)
        self.plan = base + 16
        self.pdens = {d: base + 17 + d for d in range(8)}
        self.pflag = {f: base + 25 + f for f in range(3)}  # 0 none / 1 gap / 2 climax
        self.mask = base + 28  # type-infill placeholder
        self.cplx = {c: base + 29 + c for c in range(6)}  # rhythmic-complexity band
        # dual-mode models: explicit output-semantics switch (v3). mode "slot"
        # = positions are lattice indices on a supplied grid; "time" = frames.
        self.mode = {"slot": base + 35, "time": base + 36}
        self.size = base + 37

        self.id2note = {v: k for k, v in self.note.items()}
        self.id2course = {v: k for k, v in self.course.items()}

    def time(self, frame: int) -> int:
        assert 0 <= frame < WINDOW
        return self.time0 + frame

    def is_time(self, tok: int) -> bool:
        return self.time0 <= tok < self.time0 + WINDOW

    def is_note(self, tok: int) -> bool:
        return tok in self.id2note

    def dens_bucket(self, nps: float) -> int:
        return min(N_DENS - 1, max(0, int(nps / DENS_BUCKET_NPS)))


VOCAB = Vocab()


SIB_EVENTS = 12  # fixed number of sibling-chart events in the prefix (24 tokens)


PLAN_SLOTS = 5  # fixed number of plan blocks in the prefix (2 tokens each)


def complexity_band(frames):
    """IOI-class entropy of a window's hit sequence -> band 0-5 (rhythmic
    complexity independent of raw density)."""
    import math
    hits = sorted(f for f, c in frames if NOTE_CLASSES[c] in ("don","ka","don_big","ka_big"))
    if len(hits) < 4:
        return 0
    iois = [b - a for a, b in zip(hits, hits[1:]) if b > a]
    if not iois:
        return 0
    from collections import Counter
    cls = Counter(int(round(math.log2(max(i,1)) * 2)) for i in iois)
    tot = sum(cls.values())
    ent = -sum((n/tot) * math.log2(n/tot) for n in cls.values())
    return min(5, int(ent / 0.5))


def encode_window(vocab, course, level, notes, cond_drop=0.0, rng=None, ctx_types=None,
                  sib_pairs=None, style=None, sync_band=None, plan_slice=None,
                  complexity=None, mode=None):
    """Build a token sequence for one window.

    notes: list of (frame_idx, note_class_id) sorted by frame, frame in [0, WINDOW).
    ctx_types: optional list of note-class ids from the previous window's tail
        (pattern continuity context, v2). Encoded as [BOS] ctx.. [SEP] conds..
    sib_pairs: optional list of (frame, class_id) events from an easier course of
        the same song (skeleton hint, easy⊂hard). Fixed SIB_EVENTS slots,
        missing slots filled with UNK. Encoded after [SIB].
    Returns (tokens, prefix_len) where loss should be applied after the prefix.
    """
    n_hits = sum(1 for _, c in notes if NOTE_CLASSES[c] not in ("end",))
    nps = n_hits / (WINDOW / FPS)
    d = vocab.dens_bucket(nps)

    def maybe(tok):
        if cond_drop > 0 and rng is not None and rng.random() < cond_drop:
            return vocab.unk_cond
        return tok

    lvl = max(1, min(N_LEVELS, level if level and level > 0 else 1))
    seq = [vocab.bos]
    if mode is not None:  # dual-mode: explicit output-semantics token (never dropped)
        seq.append(vocab.mode[mode])
    if ctx_types is not None:  # fixed-length, left-padded with UNK (id < 0 = pad)
        seq += [vocab.note[NOTE_CLASSES[c]] if c >= 0 else vocab.unk_cond
                for c in ctx_types]
        seq.append(vocab.sep)
    if sib_pairs is not None:
        seq.append(vocab.sib)
        pairs = list(sib_pairs)[:SIB_EVENTS]
        for f, c in pairs:
            seq += [vocab.time(int(f)), vocab.note[NOTE_CLASSES[c]]]
        seq += [vocab.unk_cond] * (2 * (SIB_EVENTS - len(pairs)))
    seq += [
        maybe(vocab.course[course]),
        maybe(vocab.level[lvl]),
        maybe(vocab.dens[d]),
    ]
    if style is not None:  # charting-intent code (style >= 0; -1 = unknown)
        seq.append(maybe(vocab.style[style]) if style >= 0 else vocab.unk_cond)
    if sync_band is not None:  # LHL syncopation band (groove-intensity control)
        seq.append(maybe(vocab.sync[sync_band]) if sync_band >= 0 else vocab.unk_cond)
    if complexity is not None:  # rhythmic-complexity band (difficulty beyond density)
        seq.append(maybe(vocab.cplx[complexity]) if complexity >= 0 else vocab.unk_cond)
    if plan_slice is not None:  # song-level plan blocks overlapping this window
        seq.append(vocab.plan)
        blocks = list(plan_slice)[:PLAN_SLOTS]
        for d8, fl in blocks:
            seq += [maybe(vocab.pdens[min(7, max(0, d8))]),
                    maybe(vocab.pflag[min(2, max(0, fl))])]
        seq += [vocab.unk_cond] * (2 * (PLAN_SLOTS - len(blocks)))
    prefix_len = len(seq)
    for f, c in notes:
        seq.append(vocab.time(int(f)))
        seq.append(vocab.note[NOTE_CLASSES[c]])
    seq.append(vocab.eos)
    return seq, prefix_len


def decode_tokens(vocab, tokens):
    """Token ids -> list of (frame_idx, note_class_name). Ignores malformed pairs."""
    out = []
    cur_t = None
    for tok in tokens:
        if vocab.is_time(tok):
            cur_t = tok - vocab.time0
        elif tok in vocab.id2note and cur_t is not None:
            out.append((cur_t, vocab.id2note[tok]))
        elif tok == vocab.eos:
            break
    return out
