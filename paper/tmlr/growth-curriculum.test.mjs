// P25: independent numerical replay of the frozen growth/positional evidence.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const read = p => fs.readFileSync(new URL(p, import.meta.url));
const evidence = '../results/invertible_quadratic/growth/';
const json = p => JSON.parse(read(evidence + p));
const names = ['e97_seed0','e97_seed1','e97_seed2','gdn2_seed0','gdn2_seed1','gdn2_seed2'];
const counts = [[512,10,2,0],[512,0,0,1],[511,1,0,1],[512,512,512,0],[512,512,508,0],[512,512,4,0]];
test('P25 growth evidence retains exact source-byte hashes and result bindings', () => {
  for (const {file,sha256} of json('provenance.json').files) {
    assert.equal(createHash('sha256').update(read(evidence + file)).digest('hex'),sha256,file);
  }
  for (const name of names) {
    const p = json(name + '/positions.json');
    const binding = Object.entries(p.sources).find(([path]) => path.endsWith('/result.json'))[1];
    assert.equal(createHash('sha256').update(read(evidence + name + '/result.json')).digest('hex'),binding);
  }
});
test('P25 delayed one-step ignition and archived-rule rejection are measured', () => {
  const curve = json('e97_T1_validation.json');
  const accuracy = row => row.validation_by_depth['1'].accuracy;
  assert.equal(accuracy(curve.find(r => r.step === 20000)),15/512);
  assert(curve.filter(r => r.step <= 20000).every(r => accuracy(r) < 0.04));
  assert.equal(curve.find(r => accuracy(r) >= 0.9).step,48500);
  const ignition = curve.find(r => accuracy(r) >= 0.97);
  assert.equal(ignition.step,57000); assert.equal(accuracy(ignition),501/512);
  assert.equal((100*accuracy(ignition)).toFixed(2),'97.85');
  const replay = json('replay.json');
  for (const threshold of [0.9,0.97]) {
    assert.equal(replay.rows.find(r => r.arm === 'e97' && r.threshold === threshold && r.rule === 'archived_X5b_no_floor').first_stop,10000);
    assert.equal(replay.rows.find(r => r.arm === 'e97' && r.threshold === threshold && r.rule === 'shipped_X5c_100k_floor').first_stop,null);
  }
  assert.equal(createHash('sha256').update(read(evidence + 'e97_T1_validation.json')).digest('hex'),
    Object.entries(replay.input_sha256).find(([path]) => path.endsWith('e97_T1/validation.json'))[1]);
});
test('P25 all six final stops are T4 stagnation, not safety, with final accuracy ranges and zero terminal', () => {
  const stops = [190000,104000,104000,324000,372000,204000];
  const finalCounts = [515,512,511,1536,1532,1025];
  names.forEach((name,i) => {
    const result = json(name + '/result.json'), stage = result.stages[0];
    assert.equal(result.stages.length,1); assert.equal(result.stopping_depth,4);
    assert.equal(result.stop_reason,'stagnation'); assert.equal(result.total_steps,stops[i]);
    assert.equal(stage.safety_stop,false); assert.equal(stage.decision,'stagnation');
    const window = stage.stagnation_window;
    assert(window.comparison_eligible); assert(window.stagnated);
    assert.equal(window.consecutive_joint_plateaus,2);
    assert(window.stage_step >= 104000);
    assert(window.best_val_gain < 0.001); assert(window.relative_train_loss_decrease < 0.001);
    assert.equal(stage.retention['4'].correct,finalCounts[i]);
    assert.equal(stage.retention['4'].count,2048);
    assert.equal(stage.retention['4'].terminal_correct,0);
    assert.equal(stage.retention['4'].terminal_count,512);
    assert.equal(stage.final_accuracy,finalCounts[i]/2048);
  });
  assert.deepEqual([511,515,1025,1536].map(n => (100*n/2048).toFixed(2)),['24.95','25.15','50.05','75.00']);
});
test('P25 positional counts independently locate depth 1 vs 3/3/2 and reconcile best, not final', () => {
  names.forEach((name,i) => {
    const p = json(name + '/positions.json'), result = json(name + '/result.json');
    assert.equal(p.depth,4); assert.equal(p.examples_per_position,512);
    assert.deepEqual(p.positions.map(x => x.correct),counts[i]);
    assert.deepEqual(p.positions.map(x => x.driver_position),[1,2,3,4]);
    assert.deepEqual(p.positions.map(x => x.token_index),[2,3,4,5]);
    assert(p.positions.every(x => x.count === 512 && x.out_of_range === 0));
    const depth = p.positions.filter(x => x.correct/512 >= 0.9).length;
    assert.equal(depth,[1,1,1,3,3,2][i]);
    const sum = p.positions.reduce((n,x) => n+x.correct,0);
    assert.equal(sum,result.stages[0].best.correct);
    assert.equal(p.positions[3].correct,result.stages[0].best.terminal_correct);
    assert.equal(p.checkpoint.sha256,result.stages[0].best.checkpoint.sha256);
    assert.deepEqual(p.reconciliation.recorded_final,result.stages[0].retention['4']);
    assert.equal(p.reconciliation.best_loss_delta,0);
    assert.deepEqual(p.reconciliation.best_count_differences,{});
    const tex = read('main.tex').toString();
    const [arm,seed] = name.split('_seed');
    assert(tex.includes(`${arm === 'e97' ? 'E97' : 'GDN2'} & ${seed} & ${counts[i].join(' & ')} \\\\`));
  });
  for (const seed of [0,1,2]) assert.equal(json(`e97_seed${seed}/positions.json`).data.sha256,json(`gdn2_seed${seed}/positions.json`).data.sha256);
  assert.equal(json('e97_seed0/positions.json').positions[1].x_correct,338);
  assert.equal(json('gdn2_seed2/positions.json').positions[2].x_correct,105);
});
test('P25 all three PDFs retain the new signature, checkpoint caveat and learned-solution scope', () => {
  for (const variant of ['canonical','submission','preprint']) {
    const text = read(`validation/${variant}-text.txt`).toString()
      .replaceAll('Under review as submission to TMLR','').replace(/(?:^|\n)\d+\s*(?=\n|\f|$)/g,'\n')
      .replace(/[^\p{L}\p{N}]/gu,'').toLowerCase();
    for (const fragment of ['silent-then-ignite','97.85% at 57,000 updates','Position 1 requires one application',
      'four-application position remains unsolved in every run (0–1/512)',
      'E97 best seeds 1/2 each have one terminal hit',
      'learned solutions under this training regime and protocol',
      'not an architectural capacity limit or impossibility result']) {
      assert(text.includes(fragment.replace(/[^\p{L}\p{N}]/gu,'').toLowerCase()),`${variant}: ${fragment}`);
    }
  }
});
