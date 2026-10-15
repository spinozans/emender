import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {verifyArmor} from './armor-audit.mjs';
const read = p => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const source = read('../main.typ'), captions = read('../figures/expressivity_1p3b.typ');
test('P18 current source keeps live scopes and findings', () => {
  assert.equal(verifyArmor(source, captions).claimColumns, 3);
});
test('P18 rejects retired expectation armor, including old S5 wording', () => {
  for (const armor of ['The negative transfer is:', 'The 8M S5 separation does NOT replicate',
    'reverses the predicted ordering', 'Non-claim', 'not a strawman']) {
    assert.throws(() => verifyArmor(source + '\n' + armor, captions), /Retired armor/);
  }
});
test('P18 rejects losing any claim-row scope', () => {
  for (const phrase of ['One E97 run, single seed', '150 B pair is pending',
    'both nonlinear\n     updates plateau below ceiling', 'different successful-seed subsets',
    'unequal realized step counts', 'TC#super[0] versus NC#super[1]']) {
    assert(source.includes(phrase), phrase);
    assert.throws(() => verifyArmor(source.replace(phrase, ''), captions), /boundary missing/);
  }
});
test('P18 rejects losing measured/current-pair or formal ceilings', () => {
  for (const phrase of ['2,500 versus\n2,200 steps', 'after one completed optimizer step',
    '0.79, GDN 0.36', '0.010 nats/token', 'the reported mean is conditional on that subset',
    'This synthetic model excludes the actual dense right-multiplication',
    'No strict separation of TC#super[0]', 'not uniformly bounded away from zero',
    "both models'\naccuracy declines toward random with length, and GDN2 leads E97 at\nevery tested length"]) {
    assert(source.includes(phrase), phrase);
    assert.throws(() => verifyArmor(source.replaceAll(phrase, ''), captions), /boundary missing/);
  }
});
test('P18 rejects caption loss of common-cap/unequal-step/OOM/above-random facts', () => {
  for (const phrase of ['unequal realized', 'above the declared', 'one\n    optimizer step', 'OOM-infeasible even at sub-batch 1']) {
    assert(captions.includes(phrase), phrase);
    assert.throws(() => verifyArmor(source, captions.replace(phrase, '')), /Caption boundary missing/);
  }
});
test('P18 rejects restored fourth claim-table column', () => {
  assert.throws(() => verifyArmor(source.replace('columns: (1.45fr, 2.05fr, 3.3fr)',
    'columns: (1.45fr, 2.05fr, 1.35fr, 1.95fr)'), captions), /three columns/);
});
