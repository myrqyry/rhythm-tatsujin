"""SoftChart 2.0 serving code, vendored verbatim into its own namespace.

The Space already ships a SoftChart 1.x library at ``softchart.*``.  The 2.0
modules carry the same file names but an incompatible token table
(``VOCAB.size`` 2071 vs 1.x, ``MAX_TGT`` 960 vs 1536), so they cannot share a
package: overwriting ``softchart/vocab.py`` or ``softchart/model.py`` would
break every 1.x checkpoint the Space serves.  They live here instead, as a
sibling subpackage.  Both trees are importable in one process and neither can
see the other -- every module below uses package-relative imports only, so
``from .vocab import VOCAB`` inside ``softchart.sc2.generate`` resolves to
``softchart.sc2.vocab`` and never to ``softchart.vocab``.

The eight modules are BYTE-IDENTICAL copies of the frozen shipping tree at
``.codex/worktrees/v18-8h/src/softchart/``.  Their md5s are the ones the
release package recorded in ``config.json:code.module_md5``:

    barscript.py   9afc715a56ac28e9ec5675d64cd0536c
    checkpoint.py  10e215e86ac77ada9046a164bffa0644
    generate.py    73b6e49b00fb60f3229c8539bc7a1029
    meter.py       18ccd6a003086f81276f2a916ada4c64
    model.py       eace5d89abeebf0edcf28d10b5ad3e68
    tja.py         360b95e19dabe925bf8eab03fa3fd110
    trace.py       e707f7204f0d86deacca6b6325f031f1
    vocab.py       0f948a0efc52f5ce60e8219fcaeb136c

Verbatim is the point: ``softchart/sc2_loader.py`` checks these md5s against
the package's own record at load time, so serving code that has drifted from
the code the checkpoint was evaluated under is refused rather than served.
Any behaviour change belongs upstream in the shipping tree, not here.

Not copied, because the BarScript serving path never reaches them:
``data.py`` (``generate.generate_song`` imports ``song_summary`` lazily; that is
the 1.x time-mode decoder), ``hf.py`` (``generate.load_hf``; this Space loads
2.0 from a local directory and must not need a Hub client), and the training,
evaluation and grid-fitting modules.
"""
