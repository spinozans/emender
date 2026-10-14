// Section 6 tables are computed from the committed result JSONs at build time.
#let result-root = "../results/expressivity_1p3b/"
#let wall-root = result-root + "iterated-map/"
#let wall-e97 = json(wall-root + "ft_e97_modular_quadratic_sweep.json")
#let wall-gdn2 = json(wall-root + "ft_gdn2_modular_quadratic_sweep.json")
#let train-e97 = json(wall-root + "ft_e97_modular_quadratic.json")
#let train-gdn2 = json(wall-root + "ft_gdn2_modular_quadratic.json")
#let budget = json(wall-root + "budget.json")
#let s5-e97 = json(result-root + "capacity/cap_e97_s5_sweep.json")
#let s5-gdn2 = json(result-root + "capacity/cap_gdn2_s5_sweep.json")
#let fixed(value, digits: 4) = {
  let parts = str(calc.round(value, digits: digits)).split(".")
  if digits == 0 { parts.first() } else {
    let decimals = if parts.len() > 1 { parts.at(1) } else { "" }
    parts.first() + "." + decimals + "0" * (digits - decimals.len())
  }
}
#let percent(value) = fixed(100 * value, digits: 2) + "%"
#let wall-acc(model, length) = model.acc_vs_T.at(str(length)).acc
#let wall-table = figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right), stroke: 0.5pt, inset: 6pt,
    table.header([*Length*], [*Train-length multiple*], [*E97*], [*GDN2-MLP*]),
    ..wall-e97.acc_vs_T.pairs().map(pair => {
      let (length, result) = pair
      (length, str(int(length) / train-e97.recipe.train_len) + "×",
       percent(result.acc), percent(wall-acc(wall-gdn2, length)))
    }).flatten(),
  ),
  caption: [
    *Modular-quadratic length extrapolation on the released 1.3 B pair.*
    Full-vocabulary argmax accuracy at the pre-registered plateau stops;
    trained length #train-e97.recipe.train_len, seed #wall-e97.seed,
    #wall-e97.acc_vs_T.at("128").n scored token positions per length.
    The common cap is #budget.cap_steps steps. E97 stopped after
    #train-e97.steps_trained steps and GDN2-MLP after #train-gdn2.steps_trained.
    E97 accuracy declines with length; GDN2-MLP approaches a roughly 50%
    plateau, above the #percent(wall-gdn2.random_baseline_acc) random baseline. Source: `paper/results/expressivity_1p3b/iterated-map/`,
    `ft_{e97,gdn2}_modular_quadratic{,_sweep}.json` (commit `9dc3afd7`);
    `budget.json` (pre-registration `1989656c`).
  ],
)
#let s5-table = figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right), stroke: 0.5pt, inset: 6pt,
    table.header([*Length*], [*E97*], [*GDN2-MLP*]),
    ..s5-e97.acc_vs_T.pairs().map(pair => {
      let (length, result) = pair
      (length + if length == "512" { "*" } else { "" },
       fixed(result.acc_all, digits: 3), fixed(s5-gdn2.acc_vs_T.at(length).acc_all, digits: 3))
    }).flatten(),
  ),
  caption: [
    *S5 group-state tracking on the current 1.3 B pair.*
    Full-vocabulary argmax accuracy, seed #s5-e97.seed;
    both curves decay and GDN2-MLP stays ahead. \*GDN2-MLP completed one
    optimizer step at length 512 before the curriculum stage proved
    OOM-infeasible even at sub-batch 1.
    Source: `capacity/cap_{e97,gdn2}_s5_sweep.json` and
    `cap_gdn2_s5_train.json` under `paper/results/expressivity_1p3b/`
    (E97 `f1ed5615`, GDN2 salvage `6f916843`).
  ],
)
