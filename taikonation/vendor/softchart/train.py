"""Training loop.

    python -m softchart.train --cache /workspace/softchart/cache --out runs/base
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ChartWindowDataset, collate, load_split_ids
from .model import ChartModel, count_params
from .vocab import VOCAB


# class-balanced weights for rare note types (~1/sqrt(train frequency), capped)
TYPE_WEIGHTS = {"don": 1.0, "ka": 1.23, "don_big": 3.19, "ka_big": 5.29,
                "roll": 7.18, "roll_big": 8.0, "balloon": 8.0, "end": 5.0}


def build_class_weights(device, cap=8.0):
    w = torch.ones(VOCAB.size, device=device)
    for name, weight in TYPE_WEIGHTS.items():
        w[VOCAB.note[name]] = min(weight, cap)
    return w


def loss_fn(logits, tgt, loss_mask, label_smoothing=0.1, class_weight=None):
    # logits: (B, L-1, V) predicting tgt[:, 1:]
    gold = tgt[:, 1:]
    mask = loss_mask[:, 1:]
    ls = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        gold.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
        weight=class_weight,
    ).reshape(gold.shape)
    return (ls * mask).sum() / mask.sum().clamp(min=1)


def corrupt_events(tgt, loss_mask, rng_gen, hard=False):
    """Synthetic negative for DPO. hard=True keeps corruptions NEAR the snap
    tolerance (small jitter, few swaps) so negatives resemble the policy's own
    mistakes rather than being trivially separable (fixes dpo_loss->0 collapse)."""
    from .vocab import WINDOW

    rej = tgt.clone()
    ev = loss_mask.clone()
    t0 = VOCAB.time0
    is_time = (rej >= t0) & (rej < t0 + WINDOW) & ev
    if hard:  # +-1..3 frames (~12-35ms, straddles the snap grid) on 60% of notes
        jit = torch.randint(-3, 4, rej.shape, device=rej.device, generator=rng_gen)
        jit = torch.where(jit == 0, torch.ones_like(jit), jit)
        do_t = torch.rand(rej.shape, device=rej.device, generator=rng_gen) < 0.6
        jit = torch.where(do_t, jit, torch.zeros_like(jit))
        swap_p = 0.12
    else:
        jit = torch.randint(-5, 6, rej.shape, device=rej.device, generator=rng_gen)
        jit = torch.where(jit.abs() < 2, jit.sign() * 2, jit)
        swap_p = 0.25
    rej = torch.where(is_time, (rej + jit).clamp(t0, t0 + WINDOW - 1), rej)
    hit0 = VOCAB.note["don"]
    is_hit = (rej >= hit0) & (rej < hit0 + 4) & ev
    swap = torch.randint(0, 4, rej.shape, device=rej.device, generator=rng_gen) + hit0
    do = torch.rand(rej.shape, device=rej.device, generator=rng_gen) < swap_p
    rej = torch.where(is_hit & do, swap, rej)
    return rej


def seq_logprob(model, mel, tgt, mask):
    import torch.nn.functional as F

    logits = model(mel, tgt)
    lp = F.log_softmax(logits.float(), dim=-1)
    gold = tgt[:, 1:]
    g = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    return (g * mask[:, 1:]).sum(1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for mel, tgt, mask, _aux, gsum, posb, _beat, _in in loader:
        mel, tgt, mask = mel.to(device), tgt.to(device), mask.to(device)
        gsum = gsum.to(device) if gsum is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(mel, tgt, gsum=gsum, pos_bucket=posb.to(device))
            l = loss_fn(logits, tgt, mask, label_smoothing=0.0)
        tot += l.item() * mel.shape[0]
        n += mel.shape[0]
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--cond-drop", type=float, default=0.15)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--balanced", action="store_true", help="class-balanced type loss")
    ap.add_argument("--balance-cap", type=float, default=8.0,
                    help="cap for class weights (v3: 2.5 after v2 over-corrected)")
    ap.add_argument("--ctx", action="store_true", help="prev-window context tokens")
    ap.add_argument("--aux", action="store_true", help="onset-heatmap auxiliary head")
    ap.add_argument("--global-ctx", action="store_true",
                    help="whole-song summary + window-position context (v4)")
    ap.add_argument("--importance-sampling", action="store_true",
                    help="bias window sampling toward informative regions (v4)")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--func-time", action="store_true",
                    help="functional (sinusoidal-basis) TIME token embeddings")
    ap.add_argument("--emb-factor", type=int, default=0,
                    help="ALBERT-style factorized embeddings (e.g. 64); 0 = off")
    ap.add_argument("--distill", default=None,
                    help="teacher checkpoint for knowledge distillation")
    ap.add_argument("--use-aug", action="store_true",
                    help="include waveform speed-augmented variants in training")
    ap.add_argument("--init-from", default=None,
                    help="initialize matching weights (decoder/tok_emb) from a checkpoint")
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="CE weight (rest = KL)")
    ap.add_argument("--kd-tau", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--tempo-aug", action="store_true",
                    help="rhythm-preserving tempo/pitch augmentation in mel domain")
    ap.add_argument("--sibling", action="store_true",
                    help="easier-course skeleton hint in the prefix (easy subset-of hard)")
    ap.add_argument("--style", action="store_true",
                    help="charting-intent style token (cache/styles.json)")
    ap.add_argument("--align", action="store_true",
                    help="pointer alignment loss: notes must attend to their onset frame")
    ap.add_argument("--beat-head", action="store_true",
                    help="beat/downbeat auxiliary head (labels from TJA measures)")
    ap.add_argument("--beat-weight", type=float, default=0.1,
                    help="beat-head loss weight (raise for a dedicated beat model)")
    ap.add_argument("--beat-hires", action="store_true",
                    help="frame-resolution beat head (11.6 ms bins instead of 46 ms)")
    ap.add_argument("--sync-token", action="store_true",
                    help="LHL syncopation-band condition token")
    ap.add_argument("--plan", action="store_true",
                    help="song-level plan-block conditioning (cache/plans.json)")
    ap.add_argument("--mask-infill", type=float, default=0.0,
                    help="probability of type-mask (skeleton->color) curriculum per window")
    ap.add_argument("--complexity", action="store_true",
                    help="rhythmic-complexity band token (density-independent difficulty)")
    ap.add_argument("--slot", action="store_true",
                    help="slot-token mode: notes as exact TJA lattice indices "
                         "(measure*96+slot), measure-aligned windows, grid-phase "
                         "input channels — zero quantization error by construction")
    ap.add_argument("--dual-slot-p", type=float, default=0.65,
                    help="dual mode: probability of a slot window (rest = time)")
    ap.add_argument("--dual", action="store_true",
                    help="dual-mode: mixed slot/time windows with an explicit "
                         "MODE token; one model serves both the grid-exact path "
                         "and the gridless time path")
    ap.add_argument("--dpo", default=None,
                    help="reference checkpoint: switch to DPO finetuning with synthetic negatives")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-hard", action="store_true",
                    help="near-tolerance negatives (resemble policy errors)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="fraction of train songs to use (data-scaling ablation); "
                         "val/test splits are unaffected")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ids, val_ids, _ = load_split_ids(args.cache, use_aug=args.use_aug,
                                           frac=args.data_frac)
    train_ds = ChartWindowDataset(
        args.cache, train_ids, train=True, cond_drop=args.cond_drop,
        spec_augment=not args.no_augment, use_ctx=args.ctx, aux=args.aux,
        global_ctx=args.global_ctx, importance_sampling=args.importance_sampling,
        tempo_aug=args.tempo_aug, sibling=args.sibling, style=args.style,
        beat_head=args.beat_head, sync_token=args.sync_token,
        plan=args.plan, mask_infill=args.mask_infill, complexity=args.complexity,
        slot=args.slot, dual=args.dual, dual_slot_p=args.dual_slot_p,
        beat_hires=args.beat_hires,
    )
    val_ds = ChartWindowDataset(args.cache, val_ids, train=False, windows_per_chart=1,
                                use_ctx=args.ctx, aux=args.aux, global_ctx=args.global_ctx,
                                sibling=args.sibling, style=args.style,
                                beat_head=args.beat_head, sync_token=args.sync_token,
                                plan=args.plan, complexity=args.complexity,
                                slot=args.slot, dual=args.dual,
                                beat_hires=args.beat_hires)
    print(f"train charts {len(train_ds.items)}, val charts {len(val_ds.items)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=4,
    )

    from .vocab import N_MELS as _NM
    model = ChartModel(
        d_model=args.d_model, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        ffn=args.ffn, aux=args.aux, global_ctx=args.global_ctx,
        func_time=args.func_time, in_ch=(_NM + 2) if (args.slot or args.dual) else None,
        emb_factor=args.emb_factor or None,
    ).to(device)
    if args.align:
        model.enable_ptr()
    if args.beat_head:
        model.enable_beat(hires=args.beat_hires)
    model.to(device)
    print(f"params: {count_params(model)/1e6:.1f}M, vocab {VOCAB.size}")
    if args.init_from and os.path.exists(args.init_from):
        src = torch.load(args.init_from, map_location=device)["model"]
        own = model.state_dict()
        hit = {k: v for k, v in src.items()
               if k in own and own[k].shape == v.shape
               and k.startswith(("decoder", "tok_emb", "out", "dec_pos"))}
        own.update(hit)
        model.load_state_dict(own)
        print(f"initialized {len(hit)} tensors from {args.init_from}")
    class_weight = build_class_weights(device, args.balance_cap) if args.balanced else None

    teacher = None
    if args.distill:
        from .generate import load_model

        teacher = load_model(args.distill, device=device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_vocab = teacher.tok_emb.weight.shape[0]
        print(f"distilling from {args.distill} (teacher vocab {t_vocab})")
    dpo_ref = None
    if args.dpo:
        from .generate import load_model as _lm

        dpo_ref = _lm(args.dpo, device=device)
        for p_ in dpo_ref.parameters():
            p_.requires_grad_(False)
        # initialize the policy from the reference (vocab may have grown since:
        # copy overlapping embedding rows, everything else matches the 12M arch)
        ref_sd = dpo_ref.state_dict()
        own = model.state_dict()
        for k, v in ref_sd.items():
            if k in own:
                if own[k].shape == v.shape:
                    own[k] = v
                elif k == "tok_emb.weight":
                    own[k][: v.shape[0]] = v
        model.load_state_dict(own)
        # disable dropout for DPO: logprob-margin noise from dropout on long
        # sequences swamps the beta-scaled objective (verified: init loss 3.3
        # instead of log 2)
        import torch.nn as _nn

        for m_ in model.modules():
            if isinstance(m_, _nn.Dropout):
                m_.p = 0.0
        print(f"DPO: policy initialized from {args.dpo}, beta={args.dpo_beta}, dropout off")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
        print(f"resumed from {args.resume} at step {step}")

    log_path = os.path.join(args.out, "train_log.jsonl")
    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    done = False
    while not done:
        for mel, tgt, mask, aux_t, gsum, posb, beat_t, in_tgt in train_loader:
            step += 1
            if step > args.steps:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            mel, tgt, mask = mel.to(device, non_blocking=True), tgt.to(device), mask.to(device)
            gsum = gsum.to(device) if gsum is not None else None
            posb = posb.to(device)
            in_tgt = in_tgt.to(device)
            if dpo_ref is not None:
                gen_rng = torch.Generator(device=device)
                gen_rng.manual_seed(args.seed * 100003 + step)
                rej = corrupt_events(tgt, mask, gen_rng, hard=args.dpo_hard)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pc = seq_logprob(model, mel, tgt, mask)
                    pr = seq_logprob(model, mel, rej, mask)
                    with torch.no_grad():
                        rc = seq_logprob(dpo_ref, mel, tgt, mask)
                        rr = seq_logprob(dpo_ref, mel, rej, mask)
                    loss = -F.logsigmoid(
                        args.dpo_beta * ((pc - pr) - (rc - rr))).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += loss.item()
                run_n += 1
                if step % 100 == 0:
                    msg = {"step": step, "dpo_loss": round(run_loss / run_n, 4)}
                    print(json.dumps(msg), flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(msg) + "\n")
                    run_loss, run_n = 0.0, 0
                if step % args.val_every == 0 or step == args.steps:
                    ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                          "step": step, "best_val": 0.0, "args": vars(args)}
                    torch.save(ck, os.path.join(args.out, "last.pt"))
                    torch.save(ck, os.path.join(args.out, "best.pt"))
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.align or args.beat_head:
                    logits, aux_logits, ptr_logits, beat_logits = model(
                        mel, tgt, return_aux=True, return_extras=True,
                        gsum=gsum, pos_bucket=posb, in_tgt=in_tgt)
                elif args.aux and aux_t is not None:
                    logits, aux_logits = model(mel, tgt, return_aux=True,
                                               gsum=gsum, pos_bucket=posb, in_tgt=in_tgt)
                    ptr_logits = beat_logits = None
                else:
                    logits = model(mel, tgt, gsum=gsum, pos_bucket=posb, in_tgt=in_tgt)
                    aux_logits = ptr_logits = beat_logits = None
                aux_loss = 0.0
                if args.aux and aux_t is not None and aux_logits is not None:
                    aux_loss = F.binary_cross_entropy_with_logits(
                        aux_logits.float(), aux_t.to(device))
                extra = 0.0
                if ptr_logits is not None:
                    # pointer alignment: positions whose GOLD token is TIME must
                    # point at that frame's encoder position
                    gold = tgt[:, 1:]
                    is_t = (gold >= VOCAB.time0) & (gold < VOCAB.time0 + 1728) & mask[:, 1:]
                    if is_t.any():
                        target_pos = ((gold - VOCAB.time0) // 4).clamp(0, ptr_logits.shape[-1] - 1)
                        pl = F.cross_entropy(
                            ptr_logits.reshape(-1, ptr_logits.shape[-1]).float(),
                            target_pos.reshape(-1), reduction="none",
                        ).reshape(gold.shape)
                        extra = extra + 0.2 * (pl * is_t).sum() / is_t.sum()
                if beat_logits is not None and beat_t is not None:
                    bt = beat_t.to(device)
                    bm = bt >= 0  # -1 marks masked (grid-visible) windows
                    if bm.any():
                        extra = extra + args.beat_weight * F.binary_cross_entropy_with_logits(
                            beat_logits.float()[bm], bt[bm])
                loss = loss_fn(logits, tgt, mask, class_weight=class_weight) + 0.1 * aux_loss + extra
                if teacher is not None:
                    with torch.no_grad():
                        # teacher may have a smaller vocab (no SIB/STYLE tokens):
                        # clamp unseen ids to UNK and distill over the shared slice
                        t_tgt = torch.where(tgt >= t_vocab,
                                            torch.full_like(tgt, VOCAB.unk_cond), tgt)
                        t_logits = teacher(mel, t_tgt)
                    V = min(t_logits.shape[-1], logits.shape[-1])
                    tau = args.kd_tau
                    kl = F.kl_div(
                        F.log_softmax(logits[..., :V].float() / tau, dim=-1),
                        F.softmax(t_logits[..., :V].float() / tau, dim=-1),
                        reduction="none",
                    ).sum(-1)
                    kl = (kl * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * (tau * tau) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            run_n += 1

            if step % 100 == 0:
                msg = {
                    "step": step,
                    "loss": round(run_loss / run_n, 4),
                    "lr": round(lr_at(step), 6),
                    "sec": round(time.time() - t0, 1),
                }
                print(json.dumps(msg), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                run_loss, run_n = 0.0, 0

            if step % args.val_every == 0 or step == args.steps:
                vl = validate(model, val_loader, device)
                msg = {"step": step, "val_loss": round(vl, 4)}
                print(json.dumps(msg), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                      "step": step, "best_val": best_val, "args": vars(args)}
                torch.save(ck, os.path.join(args.out, "last.pt"))
                if vl < best_val:
                    best_val = vl
                    ck["best_val"] = best_val
                    torch.save(ck, os.path.join(args.out, "best.pt"))

    print(f"done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
