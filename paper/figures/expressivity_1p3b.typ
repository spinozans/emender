// Current Section 6 tables read frozen result JSONs directly.
#let result-root = "../results/expressivity_1p3b/"
#let s5-e97 = json(result-root + "capacity/cap_e97_s5_sweep.json")
#let s5-gdn2 = json(result-root + "capacity/cap_gdn2_s5_sweep.json")
#let invertible = json("../results/invertible_quadratic/stage4_13b.json")
#let fixed(value, digits: 4) = {
  let parts = str(calc.round(value, digits: digits)).split(".")
  if digits == 0 { parts.first() } else {
    let decimals = if parts.len() > 1 { parts.at(1) } else { "" }
    parts.first() + "." + decimals + "0" * (digits - decimals.len())
  }
}
#let percent(value) = fixed(100 * value, digits: 2) + "%"
#let invertible-table = figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right), stroke: 0.5pt, inset: 6pt,
    table.header([*Seed*], [*E97 accuracy*], [*GDN2 accuracy*], [*E97 − GDN2 (pp)*]),
    ..range(3).map(seed => {
      let e = invertible.runs.at("headline_e97_seed" + str(seed)).test.test_1x.accuracy
      let g = invertible.runs.at("headline_gdn2_seed" + str(seed)).test.test_1x.accuracy
      (str(seed), percent(e), percent(g), fixed(100 * (e - g), digits: 3))
    }).flatten(),
    [Mean ± SD],
    [#fixed(100 * invertible.aggregate.e97.test_1x_accuracy.mean, digits: 3)% ± #fixed(100 * invertible.aggregate.e97.test_1x_accuracy.sample_sd, digits: 3)%],
    [#fixed(100 * invertible.aggregate.gdn2.test_1x_accuracy.mean, digits: 3)% ± #fixed(100 * invertible.aggregate.gdn2.test_1x_accuracy.sample_sd, digits: 3)%],
    [#fixed(100 * invertible.paired_aggregate.test_1x_accuracy.mean, digits: 3) ± #fixed(100 * invertible.paired_aggregate.test_1x_accuracy.sample_sd, digits: 3)],
  ),
  caption: [
    *Invertible quadratic tracking on the released 1.3 B pair.*
    Trained-depth full-state test accuracy at matched learning rate 1e-3;
    65,536 scored driver positions per seed. Mean and sample SD over three
    paired seeds, with validation-only checkpoint selection frozen before
    test. Running positions within a sequence are correlated. E97 leads
    on seeds 0/1 and GDN2 on seed 2; the paired difference shows no robust
    separation. Source: `paper/results/invertible_quadratic/stage4_13b.json`.
  ],
)
#let growth-root = "../results/invertible_quadratic/growth/"
#let growth-table = figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right), stroke: 0.5pt, inset: 5pt,
    table.header([*Model*], [*Seed*], [*Position 1*], [*Position 2*], [*Position 3*], [*Position 4*]),
    ..("e97", "gdn2").map(arm => range(3).map(seed => {
      let receipt = json(growth-root + arm + "_seed" + str(seed) + "/positions.json")
      (if arm == "e97" { "E97" } else { "GDN2" }, str(seed),
       ..receipt.positions.map(position => str(position.correct)))
    }).flatten()).flatten(),
  ),
  caption: [
    *Four-driver curriculum: joint correct out of 512 at each position.*
    Existing best-aggregate-validation checkpoints; one frozen positional
    evaluation per checkpoint, with no training or re-selection. Each
    position adds one nonlinear-map application. GDN2 reaches the 90%
    tracking threshold through position 3 in two seeds and position 2
    in one; E97 through position 1 in all three. None reaches position 4.
    Best and final checkpoints differ: these counts are not final scores
    or independent test results. Source:
    `paper/results/invertible_quadratic/growth/`.
  ],
)
#let s5-table = figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right), stroke: 0.5pt, inset: 6pt,
    table.header([*Length*], [*E97*], [*GDN2*]),
    ..s5-e97.acc_vs_T.pairs().map(pair => {
      let (length, result) = pair
      (length + if length == "512" { "*" } else { "" },
       fixed(result.acc_all, digits: 3), fixed(s5-gdn2.acc_vs_T.at(length).acc_all, digits: 3))
    }).flatten(),
  ),
  caption: [
    *S5 capacity/long-horizon accuracy on the released pair.*
    Full-vocabulary argmax accuracy, seed #s5-e97.seed.
    GDN2 leads at every reported length on permutation composition,
    a task admitting linear state updates; both curves decay.
    \*GDN2 completed one
    optimizer step at length 512 before the curriculum stage proved
    OOM-infeasible even at sub-batch 1; the length-512 point includes
    that completed update.
    Source: `capacity/cap_{e97,gdn2}_s5_sweep.json` and
    `cap_gdn2_s5_train.json` under `paper/results/expressivity_1p3b/`
    (E97 `f1ed5615`, GDN2 salvage `6f916843`).
  ],
)
