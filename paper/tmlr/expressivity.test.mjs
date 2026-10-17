// P23: independently replay the scoped X4R findings from frozen receipts.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const read = p => fs.readFileSync(new URL(p, import.meta.url));
const json = p => JSON.parse(read('../results/invertible_quadratic/' + p));
const x4 = json('stage4_13b.json'), x3 = json('stage3_8m.json');
const mean = xs => xs.reduce((a,b) => a+b, 0)/xs.length;
const sd = xs => Math.sqrt(xs.reduce((a,b) => a+(b-mean(xs))**2, 0)/(xs.length-1));
const close = (a,b) => assert(Math.abs(a-b)<1e-12, `${a} != ${b}`);
test('X-series frozen evidence retains source-byte hashes', () => {
  for (const {file,sha256} of json('provenance.json').files) {
    assert.equal(createHash('sha256').update(read('../results/invertible_quadratic/'+file)).digest('hex'),sha256,file);
  }
});
test('X4 matched 1e-3: all three counts, paired uncertainty, seed direction and extrapolation CE', () => {
  const accuracies = {};
  for (const [arm,counts] of [['e97',[153,131,243]],['gdn2',[59,20,368]]]) {
    accuracies[arm] = counts.map((count,seed) => {
      const run = x4.runs[`headline_${arm}_seed${seed}`];
      assert(run.frozen); assert.equal(run.test.test_1x.correct,count);
      assert.equal(run.test.test_1x.count,65536);
      for (const panel of Object.values(run.test)) assert.equal(panel.terminal_correct,0);
      return count/65536;
    });
    close(mean(accuracies[arm]),x4.aggregate[arm].test_1x_accuracy.mean);
    close(sd(accuracies[arm]),x4.aggregate[arm].test_1x_accuracy.sample_sd);
  }
  const paired = accuracies.e97.map((e,i)=>e-accuracies.gdn2[i]);
  assert.deepEqual(paired.map(d=>d>0),[true,true,false]);
  close(mean(paired),x4.paired_aggregate.test_1x_accuracy.mean);
  close(sd(paired),x4.paired_aggregate.test_1x_accuracy.sample_sd);
  assert.equal((100*mean(paired)).toFixed(3),'0.041');
  assert.equal((100*sd(paired)).toFixed(3),'0.201');
  for (const seed of [0,1,2]) {
    for (const depth of ['test_4x','test_8x']) {
      assert(x4.runs[`headline_e97_seed${seed}`].test[depth].loss < x4.runs[`headline_gdn2_seed${seed}`].test[depth].loss);
    }
  }
  close(Math.log(3721),x4.marginal_optimal_reference.nats);
  close(1/3721,x4.suffix_floor);
});
test('X2b 5e-5 GDN2 mean is 1.73%, with explicit stopping-rule confound', () => {
  const scores = [0,1,2].map(seed => json(`stage2b_gdn2_seed${seed}.json`).panels.test_1x.accuracy);
  assert.equal((100*mean(scores)).toFixed(2),'1.73');
  const source = read('../main.typ').toString().replace(/\s+/g,' ');
  assert(source.includes('earlier train-window threshold was 1%, compared with 0.1% here'));
  assert(source.includes('observational with a protocol confound'));
});
test('X3 GDN2 >= E97 is exactly the nine primary matched test-1x cells', () => {
  let cells = 0;
  for (const lr of ['0.001','0.0003','0.0001']) for (const seed of [0,1,2]) {
    const e = x3.runs[`primary_p61_e97_lr${lr}_seed${seed}`];
    const g = x3.runs[`primary_p61_gdn2_lr${lr}_seed${seed}`];
    assert(g.result.test.test_1x.accuracy >= e.result.test.test_1x.accuracy); cells++;
  }
  assert.equal(cells,9);
});
test('X3 ladder and one-step exact counts retain their matched-rate scope', () => {
  for (const [name,count,total] of [
    ['difficulty_p13_e97',1970,65536],['difficulty_p13_gdn2',2974,65536],
    ['difficulty_p31_e97',515,65536],['difficulty_p31_gdn2',1092,65536],
    ['one_step_p61_e97',500,512],['one_step_p61_gdn2',512,512]]) {
    const panel = x3.runs[`${name}_lr0.001_seed0`].result.test.test_1x;
    assert.equal(panel.correct,count); assert.equal(panel.count,total);
  }
});
test('converted map arguments and uncertainty retain mathematical glyphs and scope', () => {
  const tex = read('main.tex').toString();
  assert(tex.replace(/\s+/g,'').includes('{F}_{c}(x,y)'));
  assert(tex.replace(/\s+/g,'').includes('{F}_{c}^{-1}(u,v)'));
  assert(tex.includes('0.041 \\ensuremath{\\pm} 0.201'));
  assert(!tex.includes('0.041 ± 0.201'));
  for (const file of ['canonical','submission','preprint']) {
    const text = read(`validation/${file}-text.txt`).toString();
    assert(text.includes('0.041 ± 0.201'),file);
    assert(!text.includes('0.041 ś 0.201'),file);
  }
});
test('retired scalar headline and display stay absent from every active manuscript variant', () => {
  for (const file of ['../main.typ','../figures/expressivity_1p3b.typ','main.tex',
    'validation/canonical-text.txt','validation/submission-text.txt','validation/preprint-text.txt']) {
    const text = read(file).toString();
    assert.doesNotMatch(text,/99\.96|76\.09|76\.07|53\.36|tab_depth_wall|graceful length transfer/,file);
  }
  const source = read('../main.typ').toString().replace(/\s+/g,' ');
  assert(source.includes('erase the difference modulo 256. This task property'));
  assert(source.includes('It does not identify which shortcut, if any, the trained models used'));
  assert(source.includes('Every finite-state task admits a sufficiently large one-hot linear-state representation'));
});
