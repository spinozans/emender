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
  `ElmanProofs.Expressivity.MemoryScan]

#eval show CoreM Unit from do
  let env ← getEnv
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let mut checked := 0
  let mut theorems : Array (String × Json) := #[]
  for (name, info) in env.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let moduleName := env.allImportedModuleNames[idx.toNat]!
    unless operatorModules.contains moduleName do continue
    checked := checked + 1
    if let .axiomInfo _ := info then
      throwError "new axiom declaration: {name}"
    let axioms ← Lean.collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "unapproved axiom dependency: {name} -> {ax}"
    let .thmInfo _ := info | continue
    if name.isInternal then continue
    let some range ← findDeclarationRanges? name | continue
    let file := moduleName.toString.replace "." "/" ++ ".lean"
    let record := Json.mkObj [
      ("name", toJson name.toString),
      ("file", toJson file),
      ("line", toJson range.range.pos.line),
      ("axioms", toJson ((axioms.map Name.toString).qsort (· < ·)))]
    theorems := theorems.push (name.toString, record)
  if checked == 0 || theorems.isEmpty then throwError "empty operator audit"
  theorems := theorems.qsort (fun a b => a.1 < b.1)
  let result := Json.mkObj [
    ("status", toJson ("passed" : String)),
    ("modules", toJson (operatorModules.map Name.toString)),
    ("declarationsChecked", toJson checked),
    ("sourceTheorems", toJson theorems.size),
    ("sorryCount", toJson (0 : Nat)),
    ("sorryReasons", toJson (#[] : Array String)),
    ("newAxioms", toJson (#[] : Array String)),
    ("allowedFoundationalAxioms", toJson (allowed.map Name.toString)),
    ("theorems", Json.arr (theorems.map (·.2)))]
  IO.FS.writeFile "OPERATOR_ALGEBRA_AUDIT.json" (result.pretty ++ "\n")
  IO.println s!"Operator audit passed: {checked} declarations, {theorems.size} source theorems; no placeholders or extra axioms."
