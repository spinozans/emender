// Packaging only: convert evaluated canonical Typst content, never paraphrase it.
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkCanonicalSource} from './canonical-guard.mjs';
const dir=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(dir,'../..');
checkCanonicalSource(root);
const json=execFileSync('typst',['query','--root',root,path.join(dir,'serialize.typ'),'<manuscript>','--field','value'],{maxBuffer:16*1024*1024});
const doc=JSON.parse(json)[0];
const bib=fs.readFileSync(path.join(dir,'../refs.bib'),'utf8');
const keys=new Set([...bib.matchAll(/@\w+\s*\{\s*([^,]+),/g)].map(m=>m[1].trim()));
const stripLabel=s=>s.replace(/^<|>$/g,'');
const symbols={'λ':'\\lambda','≥':'\\geq','×':'\\times','≈':'\\approx','−':'-','…':'\\ldots','≠':'\\neq','∈':'\\in','ℝ':'\\mathbb{R}','γ':'\\gamma','δ':'\\delta','⊙':'\\odot','←':'\\leftarrow','‖':'\\Vert','σ':'\\sigma','⋅':'\\cdot','↓':'\\downarrow','⋯':'\\cdots','±':'\\pm','α':'\\alpha','τ':'\\tau','β':'\\beta','Δ':'\\Delta','→':'\\rightarrow','θ':'\\theta','φ':'\\varphi','μ':'\\mu','ε':'\\epsilon','ρ':'\\rho','η':'\\eta','ω':'\\omega','ℓ':'\\ell','≔':':=','≤':'\\leq','∗':'\\ast','∼':'\\sim',' ':'~','%':'\\%','{':'\\{','}':'\\}'};
function esc(s){s=s.replaceAll('—','---').replaceAll('–','--').replaceAll('“','``').replaceAll('”',"''").replaceAll('’',"'");return [...s].map(c=>({'′':'\\ensuremath{^{\\prime}}','·':'\\textperiodcentered{}','…':'\\ldots{}','\\':'\\textbackslash{}','{':'\\{','}':'\\}','%':'\\%','&':'\\&','#':'\\#','_':'\\_','$':'\\$','^':'\\textasciicircum{}','~':'\\textasciitilde{}','²':'\\textsuperscript{2}','³':'\\textsuperscript{3}','×':'\\ensuremath{\\times}','≤':'\\ensuremath{\\leq}','≥':'\\ensuremath{\\geq}','≈':'\\ensuremath{\\approx}','→':'\\ensuremath{\\rightarrow}','−':'\\ensuremath{-}','∞':'\\ensuremath{\\infty}','Δ':'\\ensuremath{\\Delta}','λ':'\\ensuremath{\\lambda}','τ':'\\ensuremath{\\tau}','μ':'\\ensuremath{\\mu}','σ':'\\ensuremath{\\sigma}','ε':'\\ensuremath{\\epsilon}'}[c]??c)).join('')}
let section='',formalSet='',figureNumber=0,enumGroup=0;
const inventory={headings:[],figures:[],tables:[],theoremEnvironments:[],citations:[],anonymizedURLs:[]};
function plain(x){if(!x)return '';if(typeof x.text==='string')return x.text;return (x.child?plain(x.child):'')+(x.body?plain(x.body):'')+(x.children?x.children.map(plain).join(''):'')}
function flatten(xs){return xs.flatMap(x=>x.func==='sequence'?flatten(x.children):x.func==='styled'?flatten([x.child]):[x])}
function raw(s){if(s.includes('…'))return s.split('…').map(raw).join('\\ldots{}');if(s.includes('|'))throw Error('Unexpected raw delimiter');if(/https?:\/\/(?:huggingface\.co\/spinozans|hypervolu\.me\/)/.test(s)){inventory.anonymizedURLs.push(s);return '\\artifactlink{'+s+'}'}return '\\protect\\path|'+s+'|'}
function math(x){if(!x)return '';switch(x.func){
case 'sequence':return x.children.map(math).join('');
case 'styled':return math(x.child);
case 'space':return ' ';
case 'text':return /^[\d.,]+$/.test(x.text)?x.text:'\\text{'+esc(x.text)+'}';
case 'symbol':return (symbols[x.text]??x.text)+' ';
case 'op':return '\\operatorname{'+esc(plain(x.text))+'}';
case 'attach':{
  for(const field of Object.keys(x))if(!['func','base','b','t','tr'].includes(field))throw Error('Unsupported attachment field '+field);
  if(x.tr && (x.tr.func!=='primes'||x.t))throw Error('Unsupported top-right attachment');
  return '{'+math(x.base)+'}'+(x.tr?math(x.tr):'')+(x.b?'_{'+math(x.b)+'}':'')+(x.t?'^{'+math(x.t)+'}':'');
}
case 'lr':return math(x.body);
case 'frac':return '\\frac{'+math(x.num)+'}{'+math(x.denom)+'}';
case 'align-point':return '&';
case 'linebreak':return '\\\\';
case 'primes':return "'".repeat(x.count);
case 'h':return '\\quad ';
default:throw Error('Unsupported math '+x.func);
}}
function theorem(body,env='theorem'){inventory.theoremEnvironments.push({environment:env,statement:body.slice(0,150)});return '\\begin{'+env+'}\n'+body+'\n\\end{'+env+'}'}
function seq(xs){xs=flatten(xs);let out='';for(let i=0;i<xs.length;i++){
const x=xs[i];
// Sets B/D/E/H state their results in prose rather than bullet lists. Keep
// the complete original paragraph in an unheaded amsthm environment.
if(section==='Formal Results'&&/^Theorem set [BDEH]:/.test(formalSet)&&!['space','parbreak','heading'].includes(x.func)){
  let end=i;while(end<xs.length&&!['parbreak','heading'].includes(xs[end].func))end++;
  const paragraph=xs.slice(i,end),text=paragraph.map(plain).join('');
  if(paragraph.some(e=>e.func==='raw')&&!/^(What to take|Open item)/.test(text)){
    out+='\n'+theorem(paragraph.map(render).join(''),formalSet.startsWith('Theorem set E:')?'definition':'theorem')+'\n';i=end-1;continue;
  }
}
if(['enum-item','list-item'].includes(x.func)){
const group=[x];let j=i+1;for(;j<xs.length;j++){if(['space','parbreak'].includes(xs[j].func))continue;if(xs[j].func===x.func)group.push(xs[j]);else break;}i=j-1;
const env=x.func==='enum-item'?'enumerate':'itemize';const opt=x.func==='enum-item'?(enumGroup++===0?'[label=\\arabic*.]':section==='Background'?'[label=\\arabic*.]':'[label=(\\alph*)]'):'';
if(section==='Formal Results'&&/^Theorem set [ACGF]:/.test(formalSet)){
  out+='\n'+group.map(e=>theorem(render(e.body),'theorembullet')).join('\n')+'\n';
}else out+='\n\\begin{'+env+'}'+opt+'\n'+group.map(e=>'\\item '+render(e.body)).join('\n')+'\n\\end{'+env+'}\n';
}else out+=render(x);
}return out}
function table(x){const cols=x.columns.length;let weights=x.columns.map(c=>parseFloat(c)||1);if(x.columns.every(c=>c==='auto'))weights=Array.from({length:cols},(_,i)=>i===0?2:1);if(section==='Architecture'&&cols===5)weights=[2,1,1,1,3];const sum=weights.reduce((a,b)=>a+b,0);let cells=[],headerCount=0;for(const e of x.children){if(e.func==='header'){headerCount=e.children.length;cells.push(...e.children)}else cells.push(e)}if(cells.length%cols)throw Error('Partial table row');inventory.tables.push({section,columns:cols,rows:cells.length/cols});
const spec=weights.map(w=>'>{\\hsize='+((w/sum)*cols).toFixed(5)+'\\hsize\\linewidth=\\hsize\\raggedright\\arraybackslash}X').join('|');
let out='\\begingroup\\setlength{\\tabcolsep}{3pt}\n\\begin{tabularx}{\\linewidth}{|'+spec+'|}\\hline\n';
for(let i=0;i<cells.length;i+=cols){out+=cells.slice(i,i+cols).map(e=>render(e.body)).join(' & ')+' \\\\ \\hline\n'}return out+'\\end{tabularx}\n\\endgroup\n';}
function render(x){if(!x)return '';switch(x.func){
case 'sequence':return seq(x.children);
case 'styled':return render(x.child);
case 'text':return esc(x.text);
case 'symbol':return '\\ensuremath{'+math(x)+'}';
case 'space':return ' ';
case 'parbreak':return '\n\n';
case 'strong':case 'emph':return '\\emph{'+render(x.body)+'}';
case 'raw':{
  if(x.block){
    if(x.text.includes('\\end{verbatim}'))throw Error('Unsupported verbatim terminator');
    return '\n\\begin{verbatim}\n'+x.text+'\n\\end{verbatim}\n';
  }
  return raw(x.text);
}
case 'smartquote':return x.double?'"':"'";
case 'super':return '\\textsuperscript{'+render(x.body)+'}';
case 'equation':return x.block?'\n\\[\\begin{aligned}'+math(x.body)+'\\end{aligned}\\]\n':'$'+math(x.body)+'$';
case 'heading':{const title=render(x.body);const level=x.level??1;if(level===1){section=plain(x.body);formalSet='';}else formalSet=plain(x.body);inventory.headings.push({section,level,title});const appendix=x.label==='<sec:appendix>'?'\n\\clearpage\\appendix\n':'';return appendix+'\n\\'+(level===1?'section':level===2?'subsection':'subsubsection')+(x.numbering===null?'*':'')+'{'+title+'}'+(x.label?'\\label{'+stripLabel(x.label)+'}':'')+'\n';}
case 'ref':{const key=stripLabel(x.target);if(keys.has(key)){inventory.citations.push(key);return '\\citep{'+key+'}'}return (key.startsWith('sec:')?'Section~':key.startsWith('fig')?'Figure~':'Table~')+'\\ref{'+key+'}';}
case 'figure':{const isImage=x.kind==='image'||x.body.func==='image';let body;if(x.kind==='image')body='\\includegraphics[width=\\linewidth]{assets/figure-'+(++figureNumber)+'.pdf}';else body=render(x.body);const env=isImage?'figure':'table';inventory.figures.push({label:x.label,kind:env,caption:plain(x.caption.body)});return '\n\\begin{'+env+'}[htbp]\n\\centering\n'+body+'\n\\caption{'+render(x.caption.body).replace(/\n\s*\n/g,' ')+'}\\label{'+stripLabel(x.label)+'}\n\\end{'+env+'}\n';}
case 'image':{const file=path.basename(x.source);fs.copyFileSync(path.join(dir,'..',x.source),path.join(dir,'assets',file));return '\\includegraphics[width=0.95\\linewidth]{assets/'+file+'}';}
case 'table':return table(x);
case 'align':case 'pad':return render(x.body);
case 'block':{const body=render(x.body);if(section==='Formal Results'&&formalSet.startsWith('Theorem set C′'))return '\n'+theorem(body)+'\n';return '\n'+body+'\n';}
case 'h':return '\\quad ';
case 'v':return '\n';
case 'linebreak':return '\\newline{}';
case 'counter-update':return '';
case 'pagebreak':return '\n\\clearpage\n';
case 'bibliography':return '\n\\bibliographystyle{tmlr}\n\\bibliography{refs}\n';
default:throw Error('Unsupported content '+x.func);
}}
// The evaluated template has a title/author/abstract/keywords front matter and
// one nested body. Select these structurally, never recover them from PDF text.
const template=doc.children.find(x=>x.func==='styled').child;
const abstract=template.children[5].body.body.children.find(x=>x.func==='styled').child;
const keywords=template.children[9];
const body=template.children[10].child;
let abstractTex=render(abstract).replace(/\\section\*\{Abstract\}/,'').trim();
let bodyTex=render(body);
// Only these identifying public-repository mentions are variant-gated.
bodyTex=bodyTex.replace(/public \\protect\\path\|emender\| repo/g,'public \\artifactrepo{}');
const tex=String.raw`% Generated packaging layer; canonical source: paper/main.typ (P13 baseline).
% Pre-revision accepted snapshot: eb2b9636; see canonical-baseline.json.
\documentclass[10pt]{article}
\ifdefined\EmenderPreprint
  \usepackage[preprint]{tmlr}
\else
  \usepackage{tmlr}
\fi
\usepackage{amsmath,amssymb,amsthm,graphicx,tabularx,array,enumitem}
\usepackage[hyphens]{url}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
% Unnumbered amsthm statements retain the original statement headings verbatim.
\newtheoremstyle{canonical}{3pt}{3pt}{\normalfont}{}{}{}{0pt}{}
\theoremstyle{canonical}
\newtheorem*{theorem}{}
\newtheorem*{lemma}{}
\newtheorem*{definition}{}
\newtheoremstyle{canonicalbullet}{3pt}{3pt}{\normalfont}{1em}{}{}{0.5em}{#1}
\theoremstyle{canonicalbullet}
\newtheorem*{theorembullet}{\textbullet}
\ifdefined\EmenderPreprint
  \author{\name \href{http://orcid.org/0000-0003-3821-631X}{Erik Garrison} \email erik.garrison@gmail.com\\
  \addr Poietic PBC / UTHSC, Memphis, TN, USA}
  \hypersetup{pdfauthor={Erik Garrison}}
  \DeclareRobustCommand{\artifactlink}[1]{\path|#1|}
  \DeclareRobustCommand{\artifactrepo}{\path|emender| repo}
\else
  \author{Anonymous}
  \hypersetup{pdfauthor={Anonymous}}
  \newcommand{\artifactlink}[1]{the artifact repository (anonymized)}
  \newcommand{\artifactrepo}{artifact repository (anonymized)}
\fi
\title{Emending Nonlinear Recurrence}
\hypersetup{pdftitle={Emending Nonlinear Recurrence}}
\begin{document}
\maketitle
\begin{abstract}
`+abstractTex+'\n\\end{abstract}\n\\emph{Keywords} '+render(keywords)+'\n'+bodyTex+'\n\\end{document}\n';
fs.writeFileSync(path.join(dir,'main.tex'),tex.replace(/[ \t]+$/gm,''));
fs.writeFileSync(path.join(dir,'refs.bib'),bib);
fs.writeFileSync(path.join(dir,'conversion-inventory.json'),JSON.stringify(inventory,null,2)+'\n');
console.log(`Converted ${inventory.headings.length} headings, ${inventory.figures.length} figures/tables, ${inventory.theoremEnvironments.length} amsthm statements; bibliography copied byte-for-byte.`);
