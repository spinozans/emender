import ElmanProofs.PaperCore
import Lean.Util.CollectAxioms

/-! Reproducible, fail-closed audit of all declarations in the operator modules.
Run from formal/lean: `lake env lean scripts/audit_operator_algebra.lean`.
Writes OPERATOR_ALGEBRA_AUDIT.json, including every source theorem and its axioms.
-/

open Lean Elab

def operatorModules : Array Name := #[
  `ElmanProofs.Architectures.MemoryOperator,
  `ElmanProofs.Architectures.DiagonalMemory,
  `ElmanProofs.Architectures.EmenderMemory,
  `ElmanProofs.Architectures.EmenderToken,
  `ElmanProofs.Architectures.GDNMemoryOperator,
  `ElmanProofs.Architectures.LiftedMemory,
  `ElmanProofs.Architectures.ConditionalScan,
  `ElmanProofs.Expressivity.MemoryScan,
  `ElmanProofs.Architectures.AttentionOperator,
  `ElmanProofs.Architectures.AttentionMixture,
  `ElmanProofs.Architectures.AttentionScan,
  `ElmanProofs.Architectures.AttentionObstructions]

def attentionModules : Array Name := operatorModules.extract 8 12

#eval show CoreM Unit from do
  let env ← getEnv
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let mut checked := 0
  let mut attentionChecked := 0
  let mut attentionTheorems := 0
  let mut census : Array (Name × Nat × Nat) := operatorModules.map (·, 0, 0)
  let mut theorems : Array (String × Json) := #[]
  for (name, info) in env.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let moduleName := env.allImportedModuleNames[idx.toNat]!
    unless operatorModules.contains moduleName do continue
    checked := checked + 1
    if attentionModules.contains moduleName then attentionChecked := attentionChecked + 1
    census := census.map fun (m, d, t) => (m, d + if m == moduleName then 1 else 0, t)
    if let .axiomInfo _ := info then
      throwError "new axiom declaration: {name}"
    let axioms ← Lean.collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "unapproved axiom dependency: {name} -> {ax}"
    let .thmInfo _ := info | continue
    if name.isInternal then continue
    let some range ← findDeclarationRanges? name | continue
    if attentionModules.contains moduleName then attentionTheorems := attentionTheorems + 1
    census := census.map fun (m, d, t) => (m, d, t + if m == moduleName then 1 else 0)
    let file := moduleName.toString.replace "." "/" ++ ".lean"
    let record := Json.mkObj [
      ("name", toJson name.toString),
      ("file", toJson file),
      ("line", toJson range.range.pos.line),
      ("axioms", toJson ((axioms.map Name.toString).qsort (· < ·)))]
    theorems := theorems.push (name.toString, record)
  if checked == 0 || theorems.isEmpty then throwError "empty operator audit"
  for (m, d, t) in census do
    if d == 0 || t == 0 then throwError "empty module audit: {m}"
  theorems := theorems.qsort (fun a b => a.1 < b.1)
  let result := Json.mkObj [
    ("status", toJson ("passed" : String)),
    ("modules", toJson (operatorModules.map Name.toString)),
    ("declarationsChecked", toJson checked),
    ("sourceTheorems", toJson theorems.size),
    ("attentionDeclarationsChecked", toJson attentionChecked),
    ("attentionSourceTheorems", toJson attentionTheorems),
    ("moduleCensus", Json.arr (census.map fun (m, d, t) => Json.mkObj [
      ("module", toJson m.toString), ("declarations", toJson d), ("sourceTheorems", toJson t)])),
    ("sorryCount", toJson (0 : Nat)),
    ("sorryReasons", toJson (#[] : Array String)),
    ("newAxioms", toJson (#[] : Array String)),
    ("allowedFoundationalAxioms", toJson (allowed.map Name.toString)),
    ("theorems", Json.arr (theorems.map (·.2)))]
  IO.FS.writeFile "OPERATOR_ALGEBRA_AUDIT.json" (result.pretty ++ "\n")
  IO.println s!"Operator audit passed: {checked} declarations, {theorems.size} source theorems; no placeholders or extra axioms."
  IO.println s!"Attention addition: {attentionChecked} declarations, {attentionTheorems} source theorems."
