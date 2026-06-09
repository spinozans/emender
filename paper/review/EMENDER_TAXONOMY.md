# Emender head-type taxonomy — canonical naming reference

**Status:** canonical. This document fixes the architecture vocabulary for Emender
head-types. Future code, tasks, papers, and docs **must** use these names. The
experiment-number tags (E97, E98, E99) are **retired as architecture names** and kept
**only** as historical run identifiers (which CMA run / leaderboard entry produced a
result), never as the name of a head-type.

The paper anchor is the Background subsection *"A dynamics taxonomy of head-types:
eigenvalue placement × saturation"* and `@fig_taxonomy` in `paper/main.typ`.

---

## The two axes

A head-type is one point in a 2-axis grid. The two axes are independent properties of
the per-token state map `S_t = f(A_t · S_{t-1} + B_t)` and are read from the **dynamics**,
not from the experiment that first produced the head.

### Axis 1 — eigenvalue placement in the unit disk

Where the **along-key eigenvalue** of the per-token state transition `A_t` sits in the
complex **unit disk**:

| Placement                | Name      | Dynamics                                        |
|--------------------------|-----------|-------------------------------------------------|
| real, positive           | `decay`   | stored value fades toward zero along the key    |
| real, negative           | `reflect` | sign flips each step — the tracking lever (S₅)  |
| complex (conjugate pair) | `rot`     | rotation / oscillation of the stored value      |

`decay` and `reflect` are the two **ends of the real diameter**. `rot` **walks off the
real axis** into the rest of the disk. The negative-eigenvalue (reflection) lever is what
lets a recurrence track non-solvable group structure such as S₅; a transition pinned
real-positive cannot.

### Axis 2 — state map: linear vs saturated

Whether the state update is left **linear** or passed through a bounded **saturating**
nonlinearity:

| State map     | Suffix     | Canonical fn | Dynamics                                      |
|---------------|------------|--------------|-----------------------------------------------|
| linear        | *(none)*   | identity     | unbounded along its eigendirections           |
| + saturation  | `-nonlin`  | `hardtanh`   | latches a driven slot → finite-state regime   |

The canonical saturating map of the `-nonlin` axis is `hardtanh`. The 1.3 B E88 instance
uses the smooth variant `tanh`; both are the same bounded-saturation axis. Turning this
axis on is *"turning on saturation."*

---

## The grid (named cells)

|                    | **decay** (real > 0) | **reflect** (real < 0) | **rot** (complex)        |
|--------------------|----------------------|------------------------|--------------------------|
| **linear**         | `decay`              | `reflect`              | `rot`                    |
| **+ hardtanh**     | `nonlin`             | `nonlin`               | `rot-nonlin` *(future)*  |

Cell definitions:

| Name         | Axis 1 (eigenvalue) | Axis 2 (state map) | Identity                                            |
|--------------|---------------------|--------------------|-----------------------------------------------------|
| `decay`      | real, positive      | linear             | **vanilla GDN**                                     |
| `reflect`    | real, negative      | linear             | GDN + negative eigenvalue                           |
| `nonlin`     | real (any sign)     | + hardtanh         | the saturated delta-correcting head (**E88**)       |
| `rot`        | complex             | linear             | the complex-eigenvalue oscillator head              |
| `rot-nonlin` | complex             | + hardtanh         | nonlinear oscillator — **future**, only if axes earn it |

`rot-nonlin` is reserved: add it only once both axes have independently demonstrated their
value, not speculatively.

---

## Key claim — GDN-2 is a special case of the Emender

> **Proposition (GDN-2 as a special case of the Emender).** Gated DeltaNet-2 — the
> gated-delta linear-recurrent baseline that admits a negative eigenvalue — is exactly the
> restriction of the Emender to the **real diameter** of the unit disk with **no
> saturation**: the **{`decay`, `reflect`} × linear** sub-grid. The Emender generalizes
> GDN-2 along two axes:
>
> 1. **real diameter → full unit disk** — adding the complex-eigenvalue head `rot`;
> 2. **linear state map → saturated state map** — adding the `nonlin` head.
>
> The **Emender layer is the within-layer pool** (a mixture over this grid of head-types
> inside a single layer); **GDN-2 is the linear, real-axis corner** of that pool.

Plain-language reading of the two generalization moves:

- the negative-eigenvalue result is *"walking to the far end of the real diameter"*
  (`decay` → `reflect`);
- the complex-rotation head is *"walking off the real axis"* (real → `rot`);
- `hardtanh` is *"turning on saturation"* (linear → `nonlin`).

---

## Naming map — old (experiment tag) → new (dynamics name)

| Old name / tag          | New head-type name | Notes                                              |
|-------------------------|--------------------|----------------------------------------------------|
| **E97**                 | `nonlin`           | saturated delta-correcting head (real × hardtanh)  |
| **complex-eig**         | `rot`              | complex-eigenvalue oscillator (complex × linear)   |
| vanilla GDN             | `decay`            | real-positive × linear                             |
| GDN + negative eigval   | `reflect`          | real-negative × linear                             |
| GDN-2                   | {`decay`,`reflect`} × linear | the real-diameter, no-saturation sub-grid |
| E88 (the 1.3 B instance) | `nonlin` head, carried as the 1.3 B Emender | E88 is a *run/instance* tag, not a head-type name |
| E98, E99                | *(not head-types)* | historical CMA-run / mixture-study tags only       |

**Rule of use.** When naming an architecture or head-type — in code identifiers, task
titles, figures, or prose — use the dynamics name (`decay` / `reflect` / `nonlin` / `rot`
/ `rot-nonlin`). Use E97/E98/E99/E88 **only** to refer to a specific historical run,
checkpoint, or leaderboard entry.

---

## Scope notes

- This document is **paper + naming only**. No code files or live tasks are renamed by the
  task that introduced it (`emender-taxonomy`); kernels are mid-build. A mechanical
  code-rename pass is a separate, later task.
- "GDN-2" here is the gated-delta linear-recurrent baseline of the paper (Gated DeltaNet,
  `@gated_deltanet2024`), the variant that can reach a negative along-key eigenvalue and
  therefore spans the full real diameter {`decay`, `reflect`}. Vanilla GDN occupies only
  the real-positive `decay` corner.
