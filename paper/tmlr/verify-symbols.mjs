// Independent math oracle: compare complete evaluated equations, without
// stripping operators, primes, or script boundaries. Never import convert.mjs.
import assert from 'node:assert/strict';
const symbols={'λ':'\\lambda','≥':'\\geq','×':'\\times','≈':'\\approx','−':'-','…':'\\ldots','≠':'\\neq','∈':'\\in','ℝ':'\\mathbb{R}','γ':'\\gamma','δ':'\\delta','⊙':'\\odot','←':'\\leftarrow','‖':'\\Vert','σ':'\\sigma','⋅':'\\cdot','↓':'\\downarrow','⋯':'\\cdots','±':'\\pm','α':'\\alpha','τ':'\\tau','β':'\\beta','Δ':'\\Delta','→':'\\rightarrow','θ':'\\theta','φ':'\\varphi','μ':'\\mu','ε':'\\epsilon','ρ':'\\rho','η':'\\eta','≔':':=','≤':'\\leq','∗':'\\ast','∼':'\\sim',' ':'~','%':'\\%','{':'\\{','}':'\\}'};
const compact=s=>s.replace(/\s/g,'');
function expectedMath(x){
  switch(x.func){
    case 'symbol':return symbols[x.text]??x.text;
    case 'text':return /^[\d.,]+$/.test(x.text)?x.text:'\\text{'+x.text.replaceAll('−','\\ensuremath{-}')+'}';
    case 'sequence':return x.children.map(expectedMath).join('');
    case 'styled':return expectedMath(x.child);
    case 'space':return '';
    case 'op':return '\\operatorname{'+x.text.text+'}';
    case 'lr':return expectedMath(x.body);
    case 'frac':return '\\frac{'+expectedMath(x.num)+'}{'+expectedMath(x.denom)+'}';
    case 'align-point':return '&';
    case 'linebreak':return '\\\\';
    case 'h':return '\\quad';
    case 'primes':assert(Number.isInteger(x.count)&&x.count>0);return "'".repeat(x.count);
    case 'attach':{
      const fields=Object.keys(x).filter(k=>k!=='func').sort();
      assert(fields.every(k=>['base','b','t','tr'].includes(k)),`Unaudited attachment fields: ${fields}`);
      assert(!x.tr||(x.tr.func==='primes'&&!x.t),'Unaudited top-right attachment');
      return '{'+expectedMath(x.base)+'}'+(x.tr?expectedMath(x.tr):'')+(x.b?'_{'+expectedMath(x.b)+'}':'')+(x.t?'^{'+expectedMath(x.t)+'}':'');
    }
    default:throw Error('Unaudited math node '+x.func);
  }
}
export function verifySymbols(ast,tex,pdfTexts={}){
  const expected=new Map(),attachmentVariants={};let equations=0,attachments=0,primes=0;
  function visit(x){
    if(!x||typeof x!=='object')return;
    // Vector figure bodies are imported PDFs, not converted math. Captions
    // remain in this audit; asset hashes independently guard the two drawings.
    if(x.func==='figure'&&x.kind==='image'){visit(x.caption);return;}
    if(x.func==='equation'){
      const key=compact(expectedMath(x.body));expected.set(key,(expected.get(key)||0)+1);equations++;
    }
    if(x.func==='attach'){
      attachments++;const key=Object.keys(x).filter(k=>k!=='func').sort().join(',');
      attachmentVariants[key]=(attachmentVariants[key]||0)+1;
    }
    if(x.func==='primes')primes+=x.count;
    Object.values(x).forEach(v=>Array.isArray(v)?v.forEach(visit):visit(v));
  }
  visit(ast);
  const actual=new Map();
  for(const match of tex.matchAll(/(?<!\\)\$([^$]*)\$|\\\[\\begin\{aligned\}([\s\S]*?)\\end\{aligned\}\\\]/g)){
    const key=compact(match[1]??match[2]);actual.set(key,(actual.get(key)||0)+1);
  }
  for(const [key,count] of expected)assert.equal(actual.get(key)||0,count,`Math expression or attachment lost/changed: ${key}`);
  assert.equal([...actual.values()].reduce((a,b)=>a+b,0),equations,'Unexpected math expressions');
  for(const [variant,text] of Object.entries(pdfTexts)){
    // The current canonical census is three primed S attachments; preserve
    // the prime glyph rather than the punctuation-stripping prose normalizer.
    const normalized=text.normalize('NFKC').replace(/\s/g,'');
    assert.equal((normalized.match(/S[′']/g)||[]).length,primes,`S prime lost in ${variant} PDF`);
  }
  return {equations,attachments,attachmentVariants,primes,pdfVariantsChecked:Object.keys(pdfTexts)};
}
