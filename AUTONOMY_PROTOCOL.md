# LiteAttention CI Autonomy Protocol

This file defines how the main agent and subagents should operate for the
wheel-CI optimization task.

## Objective

Optimize the total wall time of the full GitHub wheel matrix.

Priority order:
1. Whole-matrix wall time
2. Safe workflow/build-system changes
3. Compile-time improvements that preserve supported wheel coverage
4. Research and profiling that sharpen the next experiment

Non-goals:
- Optimizing only a single wheel at the expense of the matrix
- Pruning supported wheel coverage without explicit approval
- Re-running fixed baselines unless the benchmark tuple or workflow shape changes materially

## Control Loop

The main agent must always maintain:
- one active top-priority task
- one queued next experiment per host when practical
- one live status payload at `/tmp/liteattention_status.txt`
- one email notifier path
- one active assignment for every available subagent

If any subagent completes, the main agent should re-task it immediately with a
non-overlapping task.

If no benchmark is running, the main agent should automatically switch to one of:
- workflow implementation
- profiling harness work
- cache/toolchain research
- experiment design and queue preparation

## Status Cadence

Primary status channel:
- hourly email update

Status email requirements:
- subject contains current best ETA and speedup vs unoptimized baseline
- mobile-friendly formatting
- unique TL;DR
- focused on current work, not repeated history
- includes current-task ETA
- ends with a dense list of active idea lanes

Manual status checks should not reset the experiment queue.

## Benchmark Policy

Fixed baselines stay fixed unless:
- the tuple changes materially, or
- the workflow shape changes materially

Each experiment should be categorized as one of:
- whole-matrix / workflow-shape
- compile-side
- cache-side
- profiling-only
- research-only

Every completed experiment must be recorded as:
- winner
- loser
- inconclusive

Blind stacking is not allowed after two losses in the same lane. At that point,
switch to profiling or workflow-shape changes.

## Host Policy

Use `nebius61` and `beast` opportunistically.

Rules:
- do not disturb an active benchmark checkout
- do not mix unrelated workflow states on the same host during an A/B
- track disk headroom before launching large ACT/Docker runs
- prefer workflow-shape validation on the host that matters for ETA estimates

## Subagent Policy

Every subagent should own a bounded, concrete task.

Good subagent lanes:
- workflow review
- profiling design
- ccache strategy
- notifier design
- alternative toolchain triage
- public-signal synthesis

Avoid duplicate lanes. If two agents overlap, the main agent must narrow one.

## Change Policy

Prefer:
- workflow changes
- build-flag changes
- cache-key changes
- notifier/process changes

Avoid:
- kernel-surface pruning
- risky source changes
- unrelated refactors

Commit when:
- a workflow/profiling/notifier change is coherent and validated enough to keep
- a benchmark harness improvement removes repeated operational friction

## Blocker Policy

A blocker is one of:
- missing permissions
- broken infrastructure
- destructive choice requiring explicit approval
- contradictory user requirement

When blocked:
1. record the blocker in the status payload
2. switch all idle capacity to research or adjacent implementation
3. keep subagents busy on non-blocked lanes

## Current Standing Direction

Current leaders:
- whole-matrix: ABI-pair batching
- compile-side: `nvcc-resource-usage=0`

Current next steps:
- port batching into the real workflow path
- build deep NVCC timing/trace breakdown
- improve notifier reliability and formatting
- investigate realistic ccache improvements
