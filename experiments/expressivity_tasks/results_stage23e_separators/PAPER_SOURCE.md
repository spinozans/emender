# RNNs More Parallelizable?

# Why AreLinear

2 3 **William Merrill¹ Hongjian Jiang² Yanhong Li¹ Anthony Lin** **Ashish Sabharwal¹**

## Abstract

The community is increasingly exploring linear RNNs (LRNNs) as language models, motivated by their expressive power and parallelizability. While prior work establishes the expressivity ben- efits of LRNNs over transformers, it is unclear what makes LRNNs—but not traditional,*nonlinear*RNNs—as easy to parallelize in practice as transformers. We answer this question by providing a tight connection between types of RNNs and standard complexity classes. We show that LRNNs can be viewed as log-depth (bounded fan-in) arithmetic circuits, which represents only a slight depth overhead relative to log-depth boolean circuits that transformers admit. Further-more, we show that nonlinear RNNs can solve L-complete problems (and even P-complete ones, under polynomial precision), revealing a fundamental barrier to parallelizing them as efficiently as transformers. Our theory also identifies fine-grained expressivity differences between recent popular LRNN variants: permutation-diagonal LRNNs are NC¹-complete whereas diagonal-plus- low-rank LRNNs are more expressive (PNC¹- complete). We provide further insight by asso- ciating each type of RNN with a corresponding automata-theoretic model that it can simulate. Together, our results reveal fundamental tradeoffs between nonlinear RNNs and different variants of LRNNs, providing a foundation for designing LLM architectures that achieve an optimal balance between expressivity and parallelism.

(Merrill & Sabharwal,2023). RNN architecture design re- flects this tradeoff: while older RNNs were nonlinear and highly sequential (Elman,1990;Hochreiter & Schmidhuber,

1997), more recent RNN architectures prefer a*linear*state update (Bradbury et al.,2017;Katharopoulos et al.,2020; Gu & Dao,2024, inter alia) to enable parallel processing of long sequences (Blelloch,1990). Linear RNNs (LRNNs) have been shown to be surprisingly theoretically expressive relative to transformers (Merrill et al.,2024;Grazzi et al.,2025) and practically performant (Beck et al.,2024; Yang et al.,2025;Peng et al.,2025), but it remains unclear how LRNNs and nonlinear RNNs compare in expressive power, or, conversely, whether fundamental barriers prevent parallelizing nonlinear RNNs to the same degree as LRNNs. In this work, we theoretically compare the parallelizability and expressive power of nonlinear RNNs and LRNNs through the lens of computational complexity theory. A core result is a conditional separation in expressive power between the two classes: whereas LRNNs can only solve
1 tasks that are in the circuit complexity class PNC (Theorem3), nonlinear RNNs can express P-complete problems with polynomial precision (Corollary2) or L-complete problems when restricted to log precision (Theorem2). In either regime, this suggests nonlinear RNNs can represent substantially less parallelizable computation that’s beyond the 1 capabilities of LRNNs, assuming 1 PNC *̸*=P with poly precision, orPNC *̸*=Lwith log precision.

Conversely, these expressivity results have implications for the parallelizability of different types of RNNs. It follows that LRNNs—regardless of precision—can be simulated by NC circuits of depth *O*(log*n*log *∗*

*n*), i.e., with a negligible
depth overhead of *O*(log *∗*

*n*) beyond the wall-clock time we
would expect for simulating a transformer. In contrast, our results imply that nonlinear RNNs with log precision likely require Ω(log² *n*) depth, constituting a *O*(log*n*) overhead relative to a transformer. Moreover, with polynomial precision, nonlinear RNNs require more than polylog depth to simulate with bounded fan-in circuits, assumingNC*̸*=P.

Beyond comparing nonlinear RNNs and LRNNs, we also compare the fine-grained expressivity of different LRNN 1 variants within PNC. In particular, we consider differences between several different parameterizations that can express NC¹-complete problems, showing that diagonal-plus-low-

## 1. Introduction

Parallelism and expressive power are both desirable properties in an LLM architecture that are, unfortunately, at odds 1 Allen Institute for AI 2 Rheinland-Pfalzische Technische Uni-¨ 3 versitat¨ Max-Planck Institute for Software Systems. Correspon- dence to: William Merrill*<*willm@allenai.org*>*.

*Proceedings of the 43* *rd* *International Conference on Machine* *Learning*, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

<u>Why are Linear RNNs More Parallelizable?</u>

|Complexity class:|TC|NC¹|PNC¹|L|P|
|---|---|---|---|---|---|
|RNN model class:|Ungated / Diag.|PD LRNN|DPLR LRNN|Nonlinear RNN|Nonlinear RNN|
|Specific models:|(S4, Mamba)|(PD-SSM)|(DeltaNet, RWKV-7)|with log precision||

*∗* NCcircuit depth (upper bound): log(*n*) log(*n*) log(*n*) log

(*n*) log²(*n*) beyondpolylog(*n*)
Automaton class (lower bound): — DWFA WFA Counter machine Turing machine

*Figure 1.*Main results, summarized as a hierarchy of increasingly expressiveRNN classeswith popular*models*within each class. Each

RNN shown is “tight” for the respective complexity class C (PNC¹, L, etc.) in the sense that both falls in C and can solve a C-complete *∗* problem. LRNNs are in PNC¹, implying they can be nearly as efficiently parallelized as transformers, incurring only a small *O*(log

(*n*))
depth overhead in terms of bounded fan-in boolean circuits. Nonlinear RNNs, in contrast, can solve L-complete and even P-complete problems, but this comes at the cost of being less parallelizable, requiring notably deeper circuits. Bottom row lists theclasses of automata that each RNN model class can simulate, where WFA stands for weighted finite automaton and DWFA stands for deterministic WFA.

rank (DPLR) LRNNs like DeltaNet (Yang et al.,2024; Grazzi et al.,2025) and RWKV-7 (Peng et al.,2025) can express PNC¹-complete problems, whereas permutation-diagonal (PD) LRNNs (Terzic et al.,2025) are in NC¹. These results, combined with our other results about nonlinear RNNs, yield a comprehensive hierarchy of the expressivity landscape for RNNs that is summarized in Figure1.

Beyond our complexity characterization, we also associate each RNN and LRNN class with a corresponding automata-theoretic model from prior work (Peng et al.,2018;Weiss et al.,2018;Merrill et al.,2020) that this class can simulate. 1 Finally, we find empirically that our results about RNNs’ expressivity predict their behavior when trained on synthetic tasks. On an L-complete graph connectivity task, only nonlinear RNNs achieve good performance, as theory predicts. In addition, both nonlinear RNNs and DPLR LRNNs learn iterated matrix multiplication problem (PNC¹- complete), but transformers and Mamba cannot, in line with theory. These tasks provide potential synthetic benchmarks for LRNN architecture development beyond the traditional state tracking and recall tasks that are currently popular (e.g., Arora et al.,2024;Beck et al.,2024;Peng et al.,2025).

Overall, our theoretical results reveal the fundamental tradeoffs in expressivity and parallelism between linear and nonlinear RNNs. Moreover, we show fine-grained differences in expressivity between LRNNs. Taken together, we hope our results can help the community continue to refine LRNN architectures to push the frontier between expressivity and parallelism as much as possible up to fundamental barriers. 1 Code link: [https://arg-git.informatik.uni-k](https://arg-git.informatik.uni-k)

l.de/pub/LinearRNN
## 2. Preliminaries

We first introduce the datatypes we use to model bounded-precision arithmetic, as well as various LRNN architectures. We then turn to introducing computational models from circuit complexity and and formal language theory that will be relevant in our analysis of LRNNs.

### 2.1. Datatypes

Analyzing LLM architectures via circuit complexity requires making assumptions about the underlying datatype. Prior work has considered various options while trying to ensure nice numerical properties like associativity and com- mutativity (Merrill et al.,2024). In this work, guided by the theory of weighted finite automata, we make these desider- ata more explicit by asserting that the numeric datatype should be a semiring, denoted *⟨*K*,⊕,⊗⟩*. That is, addition and multiplication should be associative with identities, addition should commute, and multiplication should distribute over addition This makes the output of LRNNs more cleanly defined in an implementation-agnostic way, and also makes WFA and matrix multiplication computations well-defined for these datatypes (which will be important).

In particular, we will work with the standard*rational*semiring used in prior analysis, (i.e., polynomial-precision rational numbers;Chiang,2025), though other semirings are possible in principle.

**Definition 1.**Rational numbers, Q, form the**rational semiring**with the standard arithmetic semantics for*⊕*and*⊗*.

We assume Q admits an**order** *>* in the standard way. Additionally, we equip Q with division in order to capture layer *√* normalization Finally, we define non-linear functions *·* and exp via their Taylor series approximation in terms of

*⊕* and *⊗* to *O*(poly(*n*)) terms (cf.Chiang,2025). Each of these operations can be given poly- or log-precision variants, which return an (*n* *c* )- or (*c*log*n*)-bit approximation using this technique, respectively (see Definition10).

A natural notion for describing “simple” arithmetic functions will be the following:

**Definition 2**(Arithmetic Closed Form)**.** *f*:K *k* *→*K has an*arithmetic closed form*if it can be written as a finite *√* composition of*⊕,⊗,*max*,*exp, and *·*as defined forK.

The following property about closed forms will be useful:

**Lemma 1**(Chiang,2025)**.***If f*:Q *k* *→*Q *has an arithmetic* 0 *closed form, thenfcan be computed in*FO*-uniform*TC*.*

### 2.2. Nonlinear RNNs

An RNN over a datatype K (generally Q) is a neural network component that takes as input a sequence of vectors *x₁,...,xn∈*K *d* and outputs a new sequence *y₁,...,yn∈* K *d* that sequentially mixes them. Classical RNNs (Elman, 1990;Hochreiter & Schmidhuber,1997) were defined by updating a recurrent state vector via the application of some (generally nonlinear) function:

**Definition 3**(Nonlinear RNN)**.**Given x₁*,...,*x*n∈* K *d*, an RNN computes a a sequence of recurrent states h₁*,...,*h*n∈*K *d* via h*t*=*f*(h*t−*1*,*x*t*), where *f* is a closed-form arithmetic function (cf. Definition2).

We consider two nonlinear RNN parameterizations. First, *ReLU RNNs*are nonlinear RNNs with *f*(h*t−*1*,*x*t*) = max*{*Ah*t−*1+Bx*t,*0*}*, where A and B are matrices K *d×d*. More generally, we will sometimes consider the more general*MLP RNN*where *f* can be a multi-layer feedforward network using ReLU activations.

### 2.3. Linear RNNs

Going beyond classical RNNs, there has recently been inter- est in RNNs with*linear*state updates due to the advantages for parallelizing on the sequence dimension. These linear RNNs (LRNNs) can be defined as follows:

**Definition 4**(LRNN)**.**Given x₁*,...,*x*n∈*K *d*, an LRNN head computes a sequence of matrix valued states S₁*,...,*S*n∈*K *d×d* via

S*t*=*At*(x*t*)S*t−*1+*bt*(x*t*)*,*

where *At*(x*t*)*∈*K *d×d* and *bt*(x*t*)*∈*K *d* have arithmetic closed forms independent of*t*.

Definition4leaves open the specific parameterization of *At*and *bt*as a function of x*t*. The choice of parameterization is crucial for theoretical expressivity (Merrill et al.,

2024) as well practical concerns like memory. For instance,
constrained parameterizations (time-invariant or diagonal) were explored in early LRNNs, but these were shown to limit expressivity to TC⁰ (Merrill et al.,2024). Since then, newer LRNNs have adopted richer parameterizations that are non-diagonal but still efficient. Two broad classes of “slightly non-diagonal” parameterizations are diagonal-plus- 2 low-rank (DPLR) and permutation-diagonal (PD) LRNNs. **Definition 5**(DPLR)**.**An LRNN is*DPLR*if it satisfies *⊤* A*t*=D*t−*k*t*v*t*, where D*t*is diagonal, k*t,*v*t*are vectors, and all have an arithmetic closed form in terms ofx*t*.

In particular, we consider specific DPLR variants:

### •RWKV-7(Peng et al.,2025):

A*t*= diag(w*t*)*−λtκt*(a*t⊙κt*) *⊤* b*t*=v*t*k˜ *⊤* *t,*

for per-token vectors w*t,*a*t,κt,*v*t,λt,* k˜*t∈*K *d* are produced fromx*t*via linear projection. •**DeltaNet**(Yang et al.,2024):

A*t*=I*−βt*k *⊤* *t*k*t*b*t*=v*t*k *⊤* *t,*

### wherekt,vt,βtdepend onxtvia projection.

A key difference between RWKV-7 and DeltaNet is that the state update is symmetric in DeltaNet. Related to this, Grazzi et al.(2025) show how, for fixed-precision DeltaNet, the range of *βt*is crucial for expressivity: restricting *βt∈* (0*,*1) prevents recognizes parity, but *βt∈*(0*,*2) allows it. Both architectures have recently been trained at large scale (Peng et al.,2025;Kimi Team,2025). **Definition 6**(PD;Terzic et al.,2025)**.**An LRNN is*PD*if it satisfies A*t*=P*t*D*t*for a column one-hot 0-1 matrix P*t* and a diagonal matrix D*t*, both with an arithmetic closed form in terms ofx*t*.

P*t*represents a function *π*:*{*1*,*2*,...,d}→{*1*,*2*,...,d}*.
Despite the name, P*t*need not represent a permutation matrix, though it will be a permutation when P*t*is full rank.

### 2.4. Multilayer RNNs

In line with current practice (Yang et al.,2025), we will imagine that these RNN components (Sections2.2and2.3, respectively) are used as “heads” within a larger architecture resembling the transformer. A head is computed by pass- ing x₁*,...,*x*n*to each head; the specific parameterizations mentioned above will compute quantities resembling keys and values in the transformer. After this, the head output y*t* is determined as to be h*t*in a nonlinear RNN and q *⊤* *t*S*t−*1 in an LRNN, whereq*t*=Qx*t*is analogous to a query.

The overall architecture of a multilayer RNN then works as follows. The first layer maps each token to an embedding.

Henceforth, for brevity, we will write A*t*, k*t*, etc. to mean *At*(x*t*),*kt*(x*t*), etc., leaving the dependence onx*t* implicit.

This is followed by alternating LRNN sublayers, which mix information across tokens, and feedforward sublayers, which apply local computation at each token. More formally, these sublayers have the form: **Definition 7**(Multihead RNN Sublayer)**.**Given x₁*,...,*x*n∈*K *d*, we first apply layer-norm, compute heads y₁*,...,*y*h*in parallel, and then aggregate the outputs. Formally, the sublayer computes the mapping x*t7→*x*t*+O*·⟨*y₁*,...,*y*h⟩*. **Definition 8**(Feedforward Sublayer)**.**Given x₁*,...,*x*n∈* K *d*, we first apply layer-norm and then apply a standard two-layer ReLU network. Letting x˜= lnorm(x), the feedforward sublayer computesx*t7→*x*t*+W*·*ReLU(Ux˜*t*).

A full RNN network consists of serial composition of multihead RNN and feedforward sublayers. **Definition 9**(Language Recognition)**.**Given a datatype K, we define the language recognized by an RNN as follows. Let $*∈*Σ be a beginning-of-sequence symbol. For every string *w∈*Σ *n*, we pass $*w* through the RNN and apply a linear transformation o to the final layer output y*n* *ℓ* at the last token and accept if and only ifo *⊤* y*n* *ℓ* *>*0.

*∗ d* **Precision.**Let *f*: Σ *→*K be a representation in a neural sequence model (e.g., a transformer or RNN). We can define the precision of this representation as follows: **Definition 10.**Let encode be the function that serializes each value in a vector x*∈*K *d* in binary. The precision of h: Σ *∗* *→*K *d* is*p*h(*n*) = max*w∈*Σ*n |*encode(h(*w*))*|*.

We define the precision of an RNN as the maximum precision over all components in its computation graph. By definition, all RNN over Q have a poly-size computation *c* graph and hence at most*poly-precision*, i.e., *O*(*n*) for some

*c*. Some RNNs attain*log precision*, i.e., satisfy *O*(log*n*). When talking about log-precision networks, we will assume
*√* that log-precision variants of exp and *·* are used in the computation graph, so that extra precision is not added to intermediate values needlessly.

### 2.5. Circuit Complexity

We will leverage circuit complexity (Vollmer,1999) to for- malize the tradeoff between parallelizability and expressivity of RNNs. We first recall standard circuit complexity classes. A*circuit*is a directed acyclic graph where leaf nodes represent input variables and internal nodes represent logic gates (e.g., AND, OR, NOT) applied to their children; a circuit *Cn*define a boolean function *{*0*,*1*}* *n* *→ {*0*,*1*}*, *n* or, more generally, a function Σ *→ {*0*,*1*}* for some finite vocabulary Σ, in the standard way. A*circuit family* *C*=*{Cn}* *∞* *n*=0is then a collection of circuits indexed by *∗* input length that defines a formal language *L⊆*Σ where *w∈L* if and only if *C|w|*(*w*) = 1. NC *d* is the class of

languages recognized by bounded-fan-in boolean circuits of polynomial size and depth S *O*(log *d*

*n*). Write NC for
*d≥*0NC *d*. AC *d* is the extension of NC *d* that allows unbounded fan in, and TC *d* extends AC *d* by allowing majority (MAJ) gates in addition to AND, OR, and NOT.

In general, circuit families are a non-uniform model of computation: since there is a different circuit for each input length, there is no finite description of the overall computation. To rectify this issue, we can define*uniform variants* of circuit classes where the structure of *Cn*is determined as as a function of *n* via some kind of computable procedure. There are different natural variants of circuit classes with different levels of uniformity; unless otherwise specified, we work with first-order (FO) uniformity. SeeStrobl et al. (2024) for more background on these classes.

We will focus on the*bounded fan-in*classes, which are more aligned with real hardware. The expressivity of transformers (Hao et al.,2022;Merrill & Sabharwal,2023) and simple LRNNs like Mamba ( 0 Merrill et al.,2024) is upper bounded by TC, which is an unbounded fan-in class but known to be contained in the log-depth bounded fan-in class 1 NC. This latter class is**efficiently parallelizable**in the sense that problems in it can be solved by an appropriate hardware implementation in time proportional to the depth of the corresponding circuits, i.e., in logarithmic wall-clock time. With today’s LLMs often trained for context lengths of 64K to 1M tokens, log depth is proportional to 16 to 20 whereas the log-squared depth of NC² circuits is proportional to 256 to 400. Thus, log-squared depth increases sequential runtime by a factor of 16 to 20 relative to log depth, rendering the resulting computation significantly less parallelizable. We thus treat log-depth as the*boundary of* *efficient parallelization*.

A more liberal notion of efficient parallelizability allows for*arithmetic circuits*instead of boolean ones. For our purpose, an arithmetic circuit is defined with respect to a ring (*R,⊕,⊖,⊗,*0*,*1). In particular, it uses gates that correspond to *⊕,⊖,⊗*: *R×R→R*. Such a circuit gives rise to a function *f*: Σ *n* *→R*, where *n* is the number of input nodes.

Central to our results will be the class PNC, which cor- 1

responds to the class of languages *L* for which there is a family *C₁* =*{Cn}n≥*0of log-depth arithmetic circuits with *Cn*represents a function with *n* input nodes such that, for all *w∈*Σ *n*, we have *w∈L* iff *C* (*w*)*>*0 (*positivity* *n* check). If we replace the positivity check *Cn*(*w*)*>*0 with the zero or*equality*check *Cn*(*w*) = 0, we obtain the class ENC. 1 3 The following relations are known between these classes (assuming some level of uniformity for the circuit

While this class is often denoted as C=NC, we use the sim- plified notationENC¹ for succinctness.

### classes when compared withLorP):

TC⁰ *⊆*NC¹ *⊆*ENC¹ *⊆*PNC¹ *⊆*L 2 *⊆*NC *⊆···⊆*NC*⊆*P*.*

Whether these relations are proper is a long-standing open question in complexity theory (just as the question of P vs. NP). That said, we will proceed as in complexity theory, namely, associate a neural model with one of the above complexity class by showing that it*captures a complete problem* *in the class*and is*subsumed by the class*. We define complete problems for these circuit classes in the standard way,

i.e., under first-order (FO) reductions. For NC¹, natural complete problems include simulating a DFA (i.e., the word problem for *S₅*), multiplying matrices over the boolean semiring, and evaluating boolean formulas (Buss,1987). For PNC¹, complete problems can be defined that, infor- mally, extend these NC¹-complete problems with arithmetic values (cf. Proposition3;Caussinus et al.,1998). We note that PNC¹ is generally thought to be “just above” NC¹ in the sense that it can be simulated by NC circuits of depth *O*(log*n*log
*∗*

*n*). 4 For sequence of length 64K to
1M, the overhead factor in parallel runtime relative to NC¹ is just 3. We will therefore think of the class PNC¹ as nearly as efficiently parallelizable as NC¹ in general and as transformers in particular.

## 3. Parallelizability Limits of Nonlinear RNNs

We start with a folklore upper bound on the expressivity of nonlinear RNNs (proof in AppendixAfor completeness): **Proposition 1**(Upper Bound)**.***Let M be an RNN over* Q *with s*=*O*(poly(*n*)) *precision. Then the language* *recognized by M is also recognizable by a Turing machine* *that usesO*(max*{s,*log*n}*)*space and*poly(*n*)*time.* **Corollary 1.***Poly-precision RNNs can be simulated in* P *and log-precision RNNs can be simulated in*L*.*

This result gives a*fully sequential*simulation for poly-precision RNNs in P. On the other hand, since L*⊆*NC², log-precision nonlinear RNNs can be parallelized to depth *O*(log² *n*), incurring a factor of *O*(log*n*) in circuit depth compared to transformers. We will next show that both of these upper bounds are essentially tight, barring major breakthroughs in complexity theory. That is, poly-precision RNNs likely cannot be parallelized at all to poly-logarithmic depth, and, for log-precision RNNs, the *O*(log*n*) penalty relative to transformers likely cannot be removed. 4 Mereghetti & Palano(2000) mention that the iterated matrix multiplication problem is in NC¹. However, they do not provide a proof or reference, and this turns out to be not quite accurate. The best known simulation of PNC¹ by NC circuits recursively uses the Chinese Remainder Theorem and incurs a small overhead of *O*(log *∗*

*n*), requiring total depth*O*(log*n*log
*∗*

*n*)(Jung,1985).
**3.1. Poly-Precision Nonlinear RNNs Are**P**-Complete** We first show that nonlinear RNNs with polynomial precision can simulate Turing machines, taking inspiration from the classical construction ofSiegelmann & Sontag(1995). It will follow that they can solve P-complete problems, which means they cannot be parallelized (to poly-log depth) unless NC=P. Towards showing this, we prove that RNNs whose gating function is an MLP can simulate a multi-stack machine, which is Turing-complete (proof in AppendixA): **Theorem 1**(Turing-Completeness)**.***For any multi-stack machine M, there exists a one-layer MLP RNN that recognizes* *the same language asM.* *Proof sketch.* The idea followsSiegelmann & Sontag (1995). We use a poly-precision scalar in the RNN state to represent each stack and leverage the fact that a ReLU network (used in the recurrence) can update the stack by pushing and popping. Since a multi-stack machine is Turing complete, it follows from Theorem1that poly-precision nonlinear RNNs can solve aP-complete problem (proof in AppendixA): **Corollary 2**(P-Completeness)**.***There is a one-layer MLP* *RNN whose language is*P*-complete under*FO*reductions.* It follows that, under standard complexity conjectures, poly-precision nonlinear RNNs cannot be parallelized effectively: **Corollary 3**(Poly-Precision Depth Lower Bound)**.***If* NC*̸*= P*, there is a nonlinear RNN that, for any k, cannot be*
*k* *simulated by an*NC*circuit family of depthO*(log *n*)*.*

**3.2. Log-Precision Nonlinear RNNs Are**L**-Complete** We have established that poly-precision nonlinear RNNs cannot be parallelized effectively assumingNC*̸*=P. However, in practice, LLMs use bounded precision, which motivates similarly analyzing nonlinear RNNs with log precision. Corollary1shows such RNNs are in L; we now show they can also solve an L-complete problem, establishing that they likely cannot be parallelized as effectively as transformers or linear RNNs under standard conjectures. Prior work (Merrill & Sabharwal,2025b) has used the*graph* *connectivity*problem as a prototypical L-complete problem when analyzing models such as log-depth transformers. This problem, however, does not naturally lend to a left-to-right single-pass solution that would be suitable for formalizing the additional expressive power of nonlinear RNNs. To work around this hurdle, we consider the following*sorted*and *deterministic*variant of the problem where each graph node has exactly one outgoing edge and the nodes are presented in a topologically sorted order: **Definition 11.***Sorted deterministic graph connectivity*is the problem that takes as input a source node *s*, sequence

of directed edges (*i₁,j₁*)*,...*(*in,jn*), and target node *t*, all encoded in unary. The edges are*deterministic*— meaning that each *i* has at most one outgoing edge (*i,j*) — and *topologically sorted*— meaning *i₁ <···< in*. The output is 1 if and only if there is a path of edges from*s*to*t*.

Perhaps surprisingly, this simpler variant of deterministic graph connectivity remainsL-complete: **Proposition 2.***Sorted deterministic graph connectivity is* L*-complete under*FO*-reductions.*

We defer the proof to AppendixA.2. Intuitively, this is achieved by reduction from deterministic (but not necessarily sorted) graph connectivity. The latter is well-known to be L-complete (cf.Jones,1975;Cook & McKenzie, 1987;Sipser,1997), and is in fact also complete under FO-reductions just like the general graph connectivity problem (Immerman,1998). The reduction makes *O*(*n*) copies of the deterministic graph and connects the *i*-th copy to (*i*+ 1)-st copy. In this way, the resulting deterministic graph is polynomial in the size of the original graph and at the same time topologically sorted.

The topological sortedness condition allows a logspace machine to solve the sorted deterministic graph connectivity problem by reading the graph in one direction (from left to right). This is also the reason that a single-layer nonlinear RNN can solve the sorted deterministic graph connectivity problem. Since the indices of the vertices (given in unary) can be encoded in binary with *O*(log*n*) bits, nonlinear RNNs can solve this problem with only log precision. **Theorem 2**(L-Complete)**.***There exists a one-layer log-precision MLP RNN that solves sorted deterministic graph* *connectivity, an*L*-complete problem.*

*Proof sketch.* The proof leverages the computational model of counter machines (Fischer et al.,1968;Weiss et al., 2018;Merrill,2021), a generalization of finite automata augmented with a fixed number of integer-valued registers that has been heavily used to analyze RNNs (Weiss et al., 2018;Merrill,2019). Lemma2shows a counter machine can solve sorted deterministic graph connectivity. Moreover, Lemma3shows a log-precision MLP RNN can simulate a counter machine. 5 Thus, we conclude that an MLP RNN can also solve sorted deterministic graph connectivity.

It is a longstanding open problem (Allender,2023) if any L-complete problem has a boolean circuit family of depth *o*(log² *n*). Thus, we have: **Corollary 4**(Log-Precision Parallelism Limit)**.***Unless every problem in* L *admits an* NC *circuit family of o*(log² *n*) 5 While counter machines have been leveraged extensively to analyze RNNs in the past (Weiss et al.,2018;Merrill et al.,2020), Lemma3is novel and interesting in its own right.

*depth, there exists a log-precision nonlinear RNN that requires*NC*circuit families of depth*Ω(log² *n*)*to simulate.*

Corollary4relates to recent work on parallelizing nonlinear RNNs via Newton’s method, i.e., exactly computing the nonlinear RNN through iterated linear approximation (Danieli et al.,2026;Gonzalez et al.,2026).Gonzalez et al. (2025) show that this method runs in time *O*(log² *n*) for predictable dynamical systems, matching our theoretical expectation for parallel runtime of nonlinear RNNs.

## 4. Simulating LRNNs with Parallel Circuits

Having established the limits of parallelizing nonlinear RNNs, we turn to understanding the degree to which linear RNNs can be parallelized and, conversely, their limitations for expressing inherently sequential computation. Prior theoretical work has also studied the expressivity of linear RNNs from a circuit complexity perspective, revealing expressivity advantages of*some*linear RNNs over transformers (Merrill et al.,2024). In particular, whereas linear RNNs with simple parameterizations like S4 and Mamba fall into TC⁰, the same complexity class as transformers, more complex parameterizations like DeltaNet (Grazzi et al.,2025) and 1 RWKV (Peng et al.,2025) can express NC -complete finite state tracking problems. This past work suggests a key advantage of well-parameterized linear RNN architectures over transformers, but leaves open a precise understanding of whether this additional expressivity implies drawbacks in parallelism, i.e, additional overhead in the sequential runtime required to simulate LRNNs relative to transformers. 0 Transformers only compute functions in 1 TC, which implies they can be simulated by NC circuits of depth *O*(log*n*). It is known that linear RNNs can be parallelized to near log depth using the parallel SCAN algorithm (Blelloch,1990), but this does not immediately yield a circuit complexity characterization comparable to that of transformers. We address this gap by showing that all linear RNNs, independent of parameterization details, can be simulated in the 1 complexity class PNC of bounded fan-in log-depth circuits with arithmetic gates:

**Theorem 3**(LRNN Upper Bound)**.***The language recognized by any LRNN over*Q*is in*PNC¹*.*

*Proof.* Since FO-uniform polynomial-size circuit families are closed under finite composition (Merrill & Sabharwal, 2025a, Proposition 3), the proof proceeds by simulating each component in the network inFO-uniformPNC¹.

As a building block, we need gates that take as input bit representations for (*s,t*) and output the integer-pair representation (*s,t*) for the rational number *s/t*. This can be done by an FO-uniform poly-sized log-depth arithmetic Z-circuit (here, the measurement is with respect to the num-

ber of bits of *s* and *t*), as follows: P if *s* is represented by *∗ n−*1 *i* *b* *n−*1*...b₀ ∈{*0*,*1*}*, then *s*=*i*=0*bi·*2. Iterated sum, fast exponentiation, and multiplication by a constant can all be done by a uniform arithmetic Z-circuit with depth *O*(log*n*)and size polynomial in*n*.

Next, we write the output in a*convolutional form*(Merrill & Sabharwal,2023;Merrill et al.,2024):

X*i*Y *j* !

y*t*=x*t*+ q *⊤* *t*A*j−k*b*i−j.*(1) *j*=0 *k*=0

0 1 By Lemma1, there is an FO-uniform TC, and hence NC, circuit family that maps the inputs ¯*x* to each A*t*and b*t*. Using the previously described bits-to-rational conversion gates, we turn these into rational representations. Each of these matrices has bounded dimension *d*. Iterated matrix multiplication, as well as iterated matrix addition, for matrices of a bounded dimension over any semiring K (here K=Q) can be done using a log depth poly-sized K-arithmetic circuit (Vollmer,1999). Since Q-arithmetic circuits can be simulated using Z-circuits incurring only a constant number of operations for each gate (see Proposition

4), we obtain a GapNC¹ circuit for outputting w
*⊤* y*t*. The last step is to perform a positivity check, w *⊤* y*t>*0, which puts the problem inPNC¹.

Theorem3applies generally to LRNNs over Q (i.e., poly-precision LRNNs). For the subclass of log-precision LRNNs, it is in fact possible to obtain a tighter characterization. Let AC⁰[ENC¹] denote languages that can be solved by boolean AC⁰ circuits that can make use (as a blackbox) of any boolean circuit that comes from an ENC¹- arithmetic circuit. It is known (cf.Datta et al.,2012) that ENC¹ *⊆*AC⁰[ENC¹]*⊆*PNC¹.

**Theorem 4**(Log-Precision LRNN Upper Bound)**.***The language recognized by any log-precision LRNN over* Q *is in* AC⁰[ENC¹]*.*

*Proof.* Let *C* denote the FO-uniform GapNC¹ circuit con- structed in the proof of Theorem3that computes w *⊤* y*t>*0. Owing to log precision of the LRNN, this output can take only polynomially many possible values *v*. Thus, instead of performing a positivity check on w *⊤* y*t*(as in construction in Theorem3), we leverage log-precision to simulate the positivity check with polynomially many*equality*checks. Specifically, for each possible *v*, we extend *C* to build an FO-uniform ENC¹-circuit *Cv*that checks whether the output w *⊤* y*t*equals *v*. The positivity check is then simply an unbounded fan-in OR, and hence an AC⁰ circuit, whose inputs are the outputs of*Cv*for*v >*0.

As noted in Section2.5, PNC¹ can be simulated by *O*(log*n*log *∗*

*n*)depthNCcircuits. We therefore have:
**Corollary 5.***The language recognized by any LRNN over* Q *is also recognized by an* NC *circuit family of depth* *O*(log*n*log *∗*

*n*)*.*
It follows that LRNNs are nearly as**efficiently paralleliz-** *∗* **able**on hardware as transformers, paying only an *O*(log *n*) depth penalty beyond the *O*(log*n*) depth needed to implement transformers with NC circuits. On the other hand, we will show next that well-parameterized LRNNs significantly gain (over transformers) on the side of expressivity.

As a by-product of our above results, we also obtain the following boundary between linear RNNs and nonlinear RNNs. Recall that (sorted, deterministic) graph connectivity problem isL-complete. **Corollary 6.***Assuming* *∗* L *does not admit* NC *circuits of* *depth O*(log*n*log *n*)*, LRNNs over* Q *cannot solve (sorted,* *deterministic) graph connectivity, but nonlinear RNNs can.*

It is only known that L*⊆*NC². It is still a long-standing open problem (Allender,2023) whether any L-complete problem admitsNCcircuits with depth*o*(log² *n*).

## 5. Expressivity Differences Between LRNNs

In Section4, we showed all linear RNNs can only solve problems in PNC¹. A natural question is whether this upper bound is tight, i.e., whether each linear RNN variant can solve hard problems in PNC¹. For some variants like S4 and Mamba, we already know a tighter upper bound of TC⁰ can be given (Merrill et al.,2024), meaning these architectures do not achieve full expressivity in PNC¹. However, several newer LRNN parameterizations of LRNNs that can express NC¹-complete problems have been proposed: in particular several flavors of DLPR LRNNs as well as PD LRNNs (Section2.3). We now analyze the expressivity of these different parameterizations, showing a conditional separation: while DPLR LRNNs attain PNC¹-complete expressivity, we can show that PD LRNNs are contained withinNC¹.

## 5.1. DPLR LRNNs arePNC¹-Complete

Two common popular DPLR LRNNs are RWKV-7 (Peng et al.,2025) and DeltaNet (Schlag et al.,2021 1 ). We will show both of these architectures can solve PNC -complete problems, achieving the upper bound on LRNN expressivity 1 from Section4. We begin by defining a PNC -complete problem, namely*iterated matrix multiplication*: **Definition 12**(Iterated 3*×*3 Matrix Multiplication)**.**The input is a token stream of length 9*n* over *{−*1*,*0*,*1*}*, partitioned into *n* consecutive blocks of length 9. Block *t* encodes a matrix A *∈*K³ *×*3. The iterated matrix multipli- *t*Q *n* 3*×*3 cation problem is to computeP*n*=*t*=1A*t∈*Z. **Proposition 3**(Caussinus et al.,1998, Corollary 3.4)**.***The* 3*×*3 *iterated multiplication problem is* GapNC¹*-complete*

*under* FO *reductions, and thus checking if the* (0*,*0) *entry* *of*P*nis positive is*PNC¹ *complete under*FO*reductions.*

We show RWKV-7 and DeltaNet can express this iterated matrix multiplication problem, establishing that they can solvePNC¹-complete problems: 1 **Theorem 5**(DPLR PNC-Completeness)**.***There exist a* 4*-layer RWKV-7 and a* 4*-layer DeltaNet that solves the iterated* 3*×*3 *matrix multiplication problem and, in particular,* *checks if the*(0*,*0)*entry of*P*nis positive.*

*Proof sketch.* For a full proof, see AppendixB.3for RWKV- 7 and AppendixB.5for DeltaNet. The goal is to multiply a sequence of arbitrary matrices. If the transition matrices were allowed to have an arbitrary form, the construction would be easy with a single RNN layer. However, since the transition matrices are constrained to have DPLR form, we need a few extra layers to factor blocks of the given 3*×*3 matrices into blocks of DPLR matrices. To this end, we generalize an idea fromPeng et al.(2025): use layers 1–3 to factor blocks of arbitrary matrices into DPLR matrices with the same product, and then multiply all of these DPLR matrices to get the overall productP*n*.

Theorem5shows that RWKV-7 and DeltaNet, two DPLR 1 LRNNs, can indeed solve a PNC-complete problem, providing a lower bound on their expressivity that matches the upper bound from Theorem3. This however leaves open the possibility that even such LRNNs may not be able to solve much simpler problems within PNC¹. To address this gap, we extend the argument to also derive a more*general lower* *bound*, stating that RWKV-7 can, in fact, solve*every*problem solvable by a*Weighted Finite Automaton (WFA)*over the semiring Z (WFA; see Definition13for a definition). WFA subsume the problem of iterated 3-by-3 matrix multi- plications with matrices over*{−*1*,*0*,*1*}*, which we already remarked (see Exercise 5.10 ofVollmer,1999andCaussinus et al.,1998) to be GapNC¹-complete. In other words, the set of functions*f*: Σ *∗* *→*Zdefinable by WFAs contains a GapNC¹-complete problem. Thus, considering “acceptance” by a WFA of an input string *w* with the condition *f*(*w*)*>*0, there is aPNC¹-complete WFA language.

**Theorem 6**(DPLR WFA Simulation)**.***For any n-state WFA* *A over* Q*, there exist a* 4*-layer RWKV-7 and a 4-layer* *DeltaNet that both computefA*(*w*)*.*

A proof sketch and details are deferred to AppendixB.2for RWKV-7 and to AppendixB.4for DeltaNet.

**5.2. PD LRNNs are**NC**-Complete** PD LRNNs are another recently proposed LRNN variant that can succinctly represent the NC¹-complete state tracking problem (Terzic et al.,2025). We analyze the expres-
sive power, showing that, in contrast to DPLR LRNNs, PD LRNNs are constrained toNC¹:

**Theorem 7.***Let M be a multi-layer PD LRNN over* Q*.* 1 *Then the language recognized by M is in* FO*-uniform* NC*.*

*Proof Sketch.* The theorem reduces to showing that the product of a sequence of matrices P D1 1*,...,*P*n*D*n*can 1 be computed in FO-uniform NC .Terzic et al.(2025) observe PD matrices are closed under multiplication, so this Q*n* ˜ ˜ product also has PD form. Let*i*=1P*i*D*i*= P*n*D*n*. We derive a closed form for P˜*i*and D˜*i*, and show these can be 1 computed in uniformNC; full proof in AppendixC.

Theorem7establishes a conditional expressivity gap between DPLR and PD LRNNs: while DPLR architec- 1 tures like DeltaNet and RWKV-7 can express some 1 PNC-complete problems, PD LRNNs are restricted to NC. Moreover, since it has previously been shown that PD LRNNs can recognize regular languages ( 1 Terzic et al.,2025), we see that they can solveNC -complete problems as well.

In the one-layer case, we can show that deterministic WFAs can be simulated by PD LRNNS in the following sense:

**Theorem 8.***Any language recognized by a deterministic* *WFA with a zero threshold over* Q *can also be recognized* *by a one-layer PD LRNN over*Q*.*

Proof in AppendixD.2. From Theorems7and8, we have:

**Corollary 7.***Any language defined by a deterministic WFA* 1 *with a zero threshold can be simulated in* 1 NC*. Moreover,* *there exists an* NC*-complete language that can be recognized by a deterministic WFA with a zero threshold.*

## 6. Experiments

We have provided a concise theoretical characterization of the expressive strengths and limitations of (non)linear RNNs. We now empirically evaluate the learnable capacity of recurrent models on the Deterministic Graph Connectivity and Iterated3*×*3Matrix Multiplication. Specifically, we compare nonlinear RNNs with linear RNN models (DeltaNet and RWKV-7), the linear RNN variant Mamba, and include Transformers as an additional baseline.

**Experimental Setup.**To better evaluate learning capacity, we construct balanced datasets with length generalization. The train set contains 70K samples uniformly drawn from *N∈*[1*,*100] and the validation set contains 20K samples drawn from the same range. We evaluated on three test splits, each containing 10K samples, drawn from [1*,*100], [101*,*200], and [201*,*300]. Here, *N* denotes the number of nodes in the graph task and the size of the matrix in the matrix multiplication task.

[1,100] [101,200] [201,300]

(a) Sorted Deterministic Graph Connectivity
100 75 50 25 Accuracy(%) 0 RNN RWKV-7 DeltaNet Mamba Transformer

(b) Matrix Multiplication overZ*m*
100 75 50 25 Accuracy(%) 0 RNN RWKV-7 DeltaNet Mamba Transformer

(c) Unbounded Matrix Multiplication
100 75 50 25 Accuracy(%) 0 RNN RWKV-7 DeltaNet Mamba Transformer

*Figure 2.*Accuracy across size/length ranges. (a) Deterministic

graph connectivity over graph-size ranges. (b) Iterated 3*×*3 matrix multiplication over Z*m*. (c) Iterated 3*×*3 matrix multiplication over Z. Models are evaluated on in-distribution ranges [1*,*100] and out-of-distribution ranges[101*,*200]and[201*,*300].

We used the nonlinear RNN and transformer (Sarrof et al.,

2024), and linear models fromYang & Zhang(2024). All models are trained using the BCEWithLogitsLoss function. All models are trained using AdamW with learning rate 1*e* *−*4 and weight decay 10
*−*4, batch size 64, gradient clipping at norm 1.0, and early stopping either when the accuracy of the in-distribution data reaches 100% for three consecutive evaluations or when a maximum of 60K training steps are reached. Full benchmark details and training configurations are provided in the Appendix.

**Sorted Deterministic Graph Connectivity.**As shown in Figure2(a), all models achieve high in-distribution accuracy, indicating that sorted deterministic graph connectivity is learnable under matched training and test lengths. However, performance diverges markedly under length ex- trapolation. The nonlinear RNN maintains near-perfect accuracy across all bins, demonstrating strong length generalization. In contrast, Transformer, RWKV-7, Mamba, and DeltaNet exhibit substantial degradation as graph size increases, despite competitive performance. These results highlight the importance of recurrent inductive bias for length-generalizable algorithmic reasoning.

**Iterated Matrix Multiplication.**RWKV-7, the nonlinear RNN, and DeltaNet achieve near-perfect accuracy on the in-distribution data and degrade only moderately on the longer out-of-distribution sequences in (b). In contrast, Transformer and Mamba perform poorly across all lengths, suggesting limited capacity to capture the underlying algebraic structure even within the training regime. Architectural differences are more pronounced in the non-modular

(c). RWKV-7, the nonlinear RNN, and DeltaNet retain perfect or near-perfect accuracy across both in-distribution and out-of-distribution lengths, despite unbounded integer growth. Mamba improves with increasing sequence length but remains substantially below the top-performing models, while the Transformer degrades sharply beyond training lengths. Overall, these results indicate that architectures with explicit recurrence or state-space structure are substantially better suited for iterated linear-algebraic computations than standard attention-based Transformers.
## 7. Conclusion

We have given a comprehensive characterization of the expressive power and parallelizability of different RNNs and LRNNs. First, we reveal fundamental expressivity gaps between linear and nonlinear RNNs, highlighting the deterministic graph connectivity problem as a simple separator. This circuit complexity perspective refines an older line of work on rational recurrences (Peng et al.,2018;Merrill et al.,2020), formalizing a fundamental tradeoff between the expressivity of the state update and the parallelizability of an RNN. Second, we compare the expressivity of different types of linear RNN. Whereas both PD and DPLR parameterizations have been shown to be capable of regular language recognition in prior work, our theory reveals fine-grained differences between these variants, with PD being 1 1 NC -complete while DPLR LRNNs are PNC -complete. Experiments agree with our theoretical predictions about the expressivity differences between nonlinear RNNs and LRNNs and different LRNN variants.

Hybrid transformer–LRNN models have recently seen wide adoption (e.g.,Waleffe et al.,2024;Kimi Team,2025;Qwen Team,2026) informed by emerging understanding of LRNN expressivity (Merrill et al.,2024;Grazzi et al.,2025;Peng et al.,2025;Merrill et al.,2026). Thus, we hope our theory can guide the design of LRNN and hybrid architectures that effectively balance expressivity and parallelism. Our results also suggest recent methods to parallelize*nonlinear*RNNs (Danieli et al.,2026;Gonzalez et al.,2025) are one pathway to more expressive hybrid models, though nonlinear RNNs must fundamentally pay a Θ(log*n*) cost in parallel runtime under standard conjectures (cf. Corollary4). It is an open question whether the additional expressivity of nonlinear RNNs justifies this overhead in practice.

## Impact Statement

This paper presents foundational work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## Acknowledgments

WM thanks Mehryar Mohri for insightful discussions related to RNNs and WFAs as well as Brian Kitano and Alek- sandar Terzic for finding issues in early drafts. ´

This material is based in part upon work supported by the European Union (ERC, LASD, 101089343, https: //doi.org/10.3030/101089343).

## References

Allender, E. Discussion on parallelizing logspace problems. [https://cstheory.stackexchange.com/q](https://cstheory.stackexchange.com/q) uestions/52459/what-circuit-depth-i s-enough-to-compute-a-log-space-com plete-problem, 2023. [Online; accessed 27 Jan 2026].

Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti,

S., Zou, J., Rudra, A., and Re, C. Simple linear attention language models balance the recall-throughput tradeoff. In*ICLR 2024 Workshop on Mathematical and Empirical* *Understanding of Foundation Models*, 2024. URL http s://openreview.net/forum?id=qRlcoPhE oD.
Barcelo, P., Chiang, D., Cybenko, G., Strobl, L., and Yang,

A. Theory of Neural Language Models (Dagstuhl Semi- nar 25282).*Dagstuhl Reports*, 15(7):22–52, 2026. ISSN 2192-5283. doi: 10.4230/DagRep.15.7.22. URL [https://drops.dagstuhl.de/entities/d](https://drops.dagstuhl.de/entities/d) ocument/10.4230/DagRep.15.7.22.
Beck, M., Poppel, K., Spanring, M., Auer, A., Prudnikova, ¨

O., Kopp, M. K., Klambauer, G., Brandstetter, J., and Hochreiter, S. xLSTM: Extended long short-term memory. In*NeurIPS*, 2024. URL [https://openreview](https://openreview) .net/forum?id=ARAxPPIAhq.
Blelloch, G. E. Prefix sums and their applications. Technical Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon University, November 1990.

Bradbury, J., Merity, S., Xiong, C., and Socher, R. Quasi- recurrent neural networks. In*ICLR*, 2017. URL https:

|//openreview.net/forum?id=H1zJ-v5xl.||K. L., and Linderman, S.|
|---|---|---|
|||lelization of nonlinear state space models. InNeurIPS,|
|Buss, S. R.|The boolean formula value problem is in|2025. URL [https://openreview.net/forum|](https://openreview.net/forum|)
|alogtime.|InProceedings of the Nineteenth Annual|?id=7AGXSlXcK6.|

//openreview.net/forum?id=H1zJ-v5xl.

*ACM Symposium on Theory of Computing*, STOC ’87, pp. 123–131, New York, NY, USA, 1987. Association for Computing Machinery. ISBN 0897912217. doi:

10.1145/28395.28409. URL [https://doi.org/10](https://doi.org/10)
.1145/28395.28409.
Carlyle, J. W. and Paz, A. Realizations by stochastic finite automata.*JCSS*, 5(1):26–40, 1971.

Caussinus, H., McKenzie, P., Therien, D., and Vollmer, H. ´ Nondeterministic NC¹ computation.*Journal of Computer* *and System Sciences*, 57(2):200–212, 1998. ISSN 0022-

0000. doi: [https://doi.org/10.1006/jcss.1998.1588](https://doi.org/10.1006/jcss.1998.1588). URL [https://www.sciencedirect.com/scienc](https://www.sciencedirect.com/scienc) e/article/pii/S0022000098915884.
Chiang, D. Transformers in uniform TC⁰.*TMLR*, 2025. ISSN 2835-8856. URL [https://openreview.n](https://openreview.n) et/forum?id=ZA7D4nQuQF.

Cook, S. A. and McKenzie, P. Problems complete for deterministic logarithmic space.*Journal of Algorithms*, 8

(5):385–394, September 1987. ISSN 0196-6774. doi:
10.1016/0196-6774(87)90018-6. URL [https://doi](https://doi). org/10.1016/0196-6774(87)90018-6.
Danieli, F., Rodriguez, P., Sarabia, M., Suau, X., and Zap- pella, L. ParaRNN: Unlocking parallel training of nonlinear RNNs for large language models. In*The Fourteenth* *International Conference on Learning Representations*,

2026. URL [https://openreview.net/forum](https://openreview.net/forum) ?id=mX8b64iUaa.
Datta, S., Mahajan, M., Rao, B. V. R., Thomas, M., and Vollmer, H. Counting classes and the fine structure between NC¹ and L.*Theor. Comput. Sci.*, 417:36–49, 2012. doi: 10.1016/J.TCS.2011.05.050. URL https: //doi.org/10.1016/j.tcs.2011.05.050.

Elman, J. L. Finding structure in time.*Cognitive Science*, 14(2):179–211, 1990. doi: [https://doi.org/10.1207/s155](https://doi.org/10.1207/s155) 16709cog1402*\* 1. URL [https://onlinelibrar](https://onlinelibrar)

y.wiley.com/doi/abs/10.1207/s1551670 9cog1402_1.
Fischer, P. C., Meyer, A. R., and Rosenberg, A. L. Counter machines and counter languages.*Mathematical systems* *theory*, 2:265–283, 1968. URL [https://api.sema](https://api.sema) nticscholar.org/CorpusID:13006433.

Gonzalez, X., Kozachkov, L., Zoltowski, D. M., Clarkson,

K. L., and Linderman, S. Predictability enables paral-

Gonzalez, X., Buchanan, E. K., Lee, H. D., Liu, J. W., Wang,

K. A., Zoltowski, D. M., Kozachkov, L., Re, C., and Linderman, S. A unifying framework for parallelizing sequential models with linear dynamical systems.*TMLR*, April 2026. URL [https://openreview.net/f](https://openreview.net/f) orum?id=fw6GgAIGur.
Grazzi, R., Siems, J., Franke, J. K., Zela, A., Hutter, F., and Pontil, M. Unlocking state-tracking in linear RNNs through negative eigenvalues. In*ICLR*, 2025. URL ht tps://openreview.net/forum?id=UvTo3t VBk2.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL [https://op](https://op) enreview.net/forum?id=AL1fq05o7H.

Hao, Y., Angluin, D., and Frank, R. Formal language recognition by hard attention transformers: Perspectives from circuit complexity.*TACL*, 10:800–810, 2022.

Hochreiter, S. and Schmidhuber, J. Long short-term memory. *Neural Comput.*, 9(8):1735–1780, November 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL [https://doi.org/10.1162/neco.1997.9.8](https://doi.org/10.1162/neco.1997.9.8).

1735.
Immerman, N.*Descriptive Complexity*. Springer Verlag,

1998.
Immerman, N. and Landau, S. The complexity of iterated multiplication. Technical report, University of Mas- sachusetts, USA, 1992.

Jones, N. D. Space-bounded reducibility among combinato- rial problems.*JCSS*, 11(1):68–85, August 1975. ISSN 0022-0000. doi: 10.1016/S0022-0000(75)80050-X. URL [https://doi.org/10.1016/S0022-0000(75](https://doi.org/10.1016/S0022-0000(75)

)80050-X.
Jung, H. Depth efficient transformations of arithmetic into boolean circuits. In*Proceedings of Fundamentals of* *Computation Theory*. Springer, 1985. doi: 10.1007/BF b0028801. URL [https://link.springer.com/](https://link.springer.com/) chapter/10.1007/BFb0028801.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are RNNs: fast autoregressive transformers with linear attention. In*ICML*, 2020.

Kimi Team. Kimi Linear: An expressive, efficient attention architecture, 2025. URL [https://arxiv.org/ab](https://arxiv.org/ab) s/2510.26692.

Mereghetti, C. and Palano, B. Threshold circuits for iterated matrix product and powering.*RAIRO-Theoretical* *Informatics and Applications*, 34(1):39–46, 2000. doi:

10.1051/ita:2000105.
Merrill, W. Sequential neural networks as automata. In Eis- ner, J., Galle, M., Heinz, J., Quattoni, A., and Rabusseau, ´

G. (eds.),*Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges*, pp. 1– 13, Florence, August 2019. Association for Computational Linguistics. doi: 10.18653/v1/W19-3901. URL [https://aclanthology.org/W19-3901/](https://aclanthology.org/W19-3901/).
Merrill, W. On the linguistic capacity of real-time counter automata, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/)

2004.06866.
Merrill, W. and Sabharwal, A. The parallelism tradeoff: Limitations of log-precision transformers.*TACL*, 11:531– 545, 2023. doi: 10.1162/tacl a 00562. URL https: //aclanthology.org/2023.tacl-1.31/.

Merrill, W. and Sabharwal, A. Exact expressive power of transformers with padding. In*NeurIPS*, 2025a. URL [https://openreview.net/forum?id=O1ab](https://openreview.net/forum?id=O1ab) xStFcy.

Merrill, W. and Sabharwal, A. A little depth goes a long way: The expressive power of log-depth transformers. In *NeurIPS*, 2025b. URL [https://openreview.net](https://openreview.net) /forum?id=5pHfYe10iX.

Merrill, W., Weiss, G., Goldberg, Y., Schwartz, R., Smith,

N. A., and Yahav, E. A formal hierarchy of RNN architectures. In Jurafsky, D., Chai, J., Schluter, N., and Tetreault,
J. (eds.),*Proceedings of the 58th Annual Meeting of the* *Association for Computational Linguistics*, pp. 443–459, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.43. URL [https://aclanthology.org/2020.acl-mai](https://aclanthology.org/2020.acl-mai)
n.43/.
Merrill, W., Petty, J., and Sabharwal, A. The illusion of state in state-space models. In*Forty-first International* *Conference on Machine Learning*, 2024. URL https: //openreview.net/forum?id=QZgo9JZpLq.

Merrill, W., Li, Y., Romero, T., Svete, A., Costello, C., Dasigi, P., Groeneveld, D., Heineman, D., Kuehl, B., Lambert, N., Li, C., Lo, K., Malik, S., Matusz, D., Minix- hofer, B., Morrison, J., Soldaini, L., Timbers, F., Walsh,

P., Smith, N. A., Hajishirzi, H., and Sabharwal, A. Olmo hybrid: From theory to practice and back, 2026. URL [https://arxiv.org/abs/2604.03444](https://arxiv.org/abs/2604.03444).
Mohri, M. Rational transductors, 2026. URL https: //arxiv.org/abs/2602.07599.

Peng, B., Zhang, R., Goldstein, D., Alcaide, E., Du, X., Hou, H., Lin, J., Liu, J., Lu, J., Merrill, W., Song, G., Tan,

K., Utpala, S., Wilce, N., Wind, J. S., Wu, T., Wuttke, D., and Zhou-Zheng, C. RWKV-7 ”goose” with expressive

dynamic state evolution. In*COLM*, 2025. URL https: //openreview.net/forum?id=ayB1PACN5j.

Peng, H., Schwartz, R., Thomson, S., and Smith, N. A. Rational recurrences. In*EMNLP*, October-November

2018. doi: 10.18653/v1/D18-1152. URL https: //aclanthology.org/D18-1152/.
Qwen Team. Qwen3.5-omni technical report, 2026. URL [https://arxiv.org/abs/2604.15804](https://arxiv.org/abs/2604.15804).

Sarrof, Y., Veitsman, Y., and Hahn, M. The expressive capacity of state space models: a formal language perspective. In*Proceedings of the 38th International Conference on* *Neural Information Processing Systems*, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

Schlag, I., Irie, K., and Schmidhuber, J. Linear transformers are secretly fast weight programmers. In*ICML*, pp. 9355– 9366, 2021. URL [https://proceedings.mlr](https://proceedings.mlr). press/v139/schlag21a.html.

Siegelmann, H. and Sontag, E. On the computational power of neural nets.*Journal of Computer and System Sciences*, 50(1):132–150, 1995. ISSN 0022-0000. doi: https: //doi.org/10.1006/jcss.1995.1013. URL [https://www](https://www). sciencedirect.com/science/article/pi i/S0022000085710136.

Sipser, M.*Introduction to the Theory of Computation*. PWS Publishing Company, 1997.

Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin,

D. What formal languages can transformers express? a survey.*TACL*, 12:543–561, 2024. doi: 10.1162/tacl a 0
0663. URL [https://aclanthology.org/2024](https://aclanthology.org/2024). tacl-1.30/.
Terzic, A., Menet, N., Hersche, M., Hofmann, T., and Rahimi, A. Structured sparse transition matrices to enable state tracking in state-space models. In*The Thirty-ninth* *Annual Conference on Neural Information Processing* *Systems*, 2025. URL [https://openreview.net](https://openreview.net) /forum?id=RDbuSCWhad.

Vollmer, H.*Introduction to Circuit Complexity*. Springer,

1999.
Waleffe, R., Byeon, W., Riach, D., Norick, B., Kor- thikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh,

S., Narayanan, D., Kulshreshtha, G., Singh, V., Casper,
J., Kautz, J., Shoeybi, M., and Catanzaro, B. An empirical study of mamba-based language models, 2024. URL [https://arxiv.org/abs/2406.07887](https://arxiv.org/abs/2406.07887).
Weiss, G., Goldberg, Y., and Yahav, E. On the practical computational power of finite precision RNNs for

language recognition. In Gurevych, I. and Miyao, Y. (eds.),*Proceedings of the 56th Annual Meeting of the* *Association for Computational Linguistics (Volume 2:* *Short Papers)*, pp. 740–745, Melbourne, Australia, July

2018. Association for Computational Linguistics. doi:
10.18653/v1/P18-2117. URL [https://aclantho](https://aclantho) logy.org/P18-2117/.
Yang, A., Watson, C., Xue, A., Bhattamishra, S., Llarena,

J., Merrill, W., Ferreira, E. D. S., Svete, A., and Chiang,
D. The transformer cookbook.*TMLR*, Jan 2026. URL [https://openreview.net/forum?id=sPsh](https://openreview.net/forum?id=sPsh) CSvDrX.
Yang, S. and Zhang, Y. FLA: A triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL [https://github](https://github) .com/fla-org/flash-linear-attention.

Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. Parallelizing linear transformers with the delta rule over sequence length. In*NeurIPS*, 2024. URL https: //openreview.net/forum?id=y8Rm4VNRPH.

Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving mamba2 with delta rule. In*The Thir-* *teenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/f](https://openreview.net/f) orum?id=r8H7xhYPwz.

## A. Nonlinear RNNs

**Proposition 1**(Upper Bound)**.***Let M be an RNN over* Q *with s*=*O*(poly(*n*)) *precision. Then the language recognized by* *Mis also recognizable by a Turing machine that usesO*(max*{s,*log*n}*)*space and*poly(*n*)*time.*

*Proof.* Given a log-precision RNN of depth *d*, we will simulate it left-to-right using a Turing machine *T*. Let *s* *′* = max*{s,*log*n}*. At each step *t*, for each layer, *T* stores the hidden state *ht*at that layer using *s* bits. Given *ht−*1and input *xt*, by Lemma1, *ht*can be computed in FO-uniform TC⁰, which is contained in L. Thus, *T* can compute *ht*using *O*(max*{s,*log*n}*) space⁶ and poly(*n*) time. This space can be reused for computing *ht*+1, *ht*+2, etc. Overall, the Turing machine uses*d·O*(max*{s,*log*n}*)bits of storage space and runs inpoly(*n*)time, as desired.

### A.1. Polynomial Precision

**Theorem 1**(Turing-Completeness)**.***For any multi-stack machine M, there exists a one-layer MLP RNN that recognizes the* *same language asM.*

*Proof.* The idea is the same as the proof of Lemma3, except that we will simulate an automaton with stacks of binary values rather than counters. The stacks allow update operations push₀*,*push₁*,*pop*,* and noop. Additionally, it has a head, which is the top element. We view each stack *s* as a positive number *s∈*[0*,*2] that starts off as *s*= 1 and is modified using the following update operations:

head(*s*) =✶[*s≥*1] <u>1</u> push*v*(*s*) =*v*+ *s* ( 2 2(*s−*head(*s*))if*s̸*= 1 pop(*s*) = noop(*s*)otherwise

noop(*s*) =*s.*

Moreover, we will let *st∈*[0*,*2] *k* denote the vector of stacks at time *t*. The update operation *ut∈{*head*,*push*,*pop*,*noop*}* is determined via a finite lookup table *u* as a function of *qt*, *σt*, and head(*st*). The state transition *δ* depends on the same inputs.

The implementation of these operations with a ReLU network follows the same idea as the proof of Lemma3. We first construct a ReLU network to compute head(*s*) from *s* as a threshold with *ϵ*= 1*/*3 (Yang et al.,2026, Section 4.7). Next, we compute *qt*=*δ*(*qt−*1*,σt,*head(*st*)) and *ut*=*u*(*qt−*1*,σt,*head(*st−*1)) as finite lookup tables (Yang et al.,2026, Section

4.8). At this point, we have the correct next state, and it just remains to be shown that we can update the stacks. Let pop₁(*s*) = 2*s−*2 and pop₀(*s*) = 2*s*. We compute the possible updated stacks push*v*(*st−*1)*,*pop*v*(*st−*1)*,* and noop(*st−*1) with parallel ReLU networks, using the fact that each is a linear function. For each stack [*st*]*i*, we then compute its new value from these *u*(*st−*1)’s using a selector gadget based on [*ut*]*i*, head(*st−*1)*i*, and whether *s*= 1 (Yang et al.,2026, Section
4.9). Thus, we can correctly compute*qt*and*st*at each step. **Corollary 2**(P-Completeness)**.***There is a one-layer MLP RNN whose language is*P*-complete under*FO*reductions.* *Proof.* We leverage the same idea asBarcelo et al.(2026, Section 4.8). Let *L* be any P-complete language under FO reductions. By construction L can be recognized by a poly-time Turing machine and thus also by a multi-stack machine
*c ′ |w|* *c* *′* running in time *O*(*n*) for some *c*. Define a new padded language *L* =*{w*□ *|w∈L}*. Then *L* is P-complete and can be solved by a multi-stack machine, and thus also by an MLP RNN by Theorem1.

### A.2. Logarithmic Precision

**Proposition 2.***Sorted deterministic graph connectivity is*L*-complete under*FO*-reductions.*

*Proof.* We describe a simple FO reduction from deterministic (but not necessarily sorted) graph connectivity. The latter is well-known to be L-complete (cf.Jones,1975;Cook & McKenzie,1987;Sipser,1997), and is in fact also complete under

Herelog*n*space is for intermediate computation of theLmachine, and*s*space is for its output.

FO-reductions just like the general graph connectivity problem (Immerman,1998). We are given a deterministic graph *G* with source*s*= 1, target*t*=*n*, and edges (*i₁,j₁*)*,...,*(*im,jm*)

with *ik,jk∈{*1*,...,n}*, for each *k*= 1*,...,m*, and *i₁* =*s* and *jm*=*t*=*n*. We create a new deterministic graph *G*
*′* with edges (*i₁* +*hn,j₁* + (*h*+ 1)*n*)*,...,*(*im*+*hn,jm*+ (*h*+ 1)*n*)

for each *h*= 0*,...,m−*1. We also add an additional node *vF*=*nm*+ 1 and edges (*n*+*hn,vF*). Note that *n* has no outgoing edge in *G*, so *G* *′*
remains deterministic. That is, *v₀,...,vs*is a path in *G* iff *v₀,v₁* +*n,v₂* + 2*n,...,vs*+*sn* is a
path in*G* *′*. Thus,*t*is reachable from*s*in*G*iff*vF*is reachable from*s*in*G* *′*.

The above reduction can easily be seen to be implementable by a logspace-uniform AC⁰ reduction. The values *hn* (for each *h*= 1*,...,m*) can be precomputed by a logspace machine since this depends only on the input size (more precisely, one can enumerate enough of*hn*for up to*m*=*O*(*n*)). TheAC⁰ circuit simply needs to perform simple additions.

**Theorem 2**(L-Complete)**.***There exists a one-layer log-precision MLP RNN that solves sorted deterministic graph* *connectivity, an*L*-complete problem.*

Our proof of this theorem will leverage the computational model of counter machines (CMs;Fischer et al.,1968;Weiss et al., 2018;Merrill,2021). Also called counter automata, CMs are a generalization of finite automata augmented with a fixed number of integer-valued registers. CMs been heavily used to analyze the expressive power of RNNs (Weiss et al.,2018; Merrill,2019). Lemma2shows a counter machine can solve sorted deterministic graph connectivity. Moreover, Lemma3 shows a log-precision RNN with a ReLU MLP gating function can simulate a counter machine. Thus, we conclude that a log-precision RNN with ReLU MLP gating can also solve sorted deterministic graph connectivity.

**Lemma 2.***Sorted deterministic graph connectivity can be solved by counter machine.*

*Proof.*Fix a unary encoding of an instance

(*s,*(*i₁,j₁*)*,...,*(*in,jn*)*,t*)

with separators between *s*, each *im*, each *jm*, and *t*. We assume that vertices are numbered in a topological order and encoded in unary by their numbers, so for every edge (*im,jm*) we have *jm≥im*. The machine uses three counters *S,I,T* and*O*(1)states to maintain the invariant that, after*k*edges,*S*encodes the furthest state reachable from*s*.

The first step is to process the initial block 0 *s*. The machine starts in a state *qs*and increments *S* on each symbol (leaving *I,T*at0), so after this block*S*=*s*.

We next enter a state *q*edgeand process each edge (*im,jm*) in turn, maintaining the invariant that after *m* edges we have *S*=*sm*(the node reached from *s* after the first *m* edges) and *I*= 0. For edge (*im,jm*), in a state *qi*the machine reads the unary block0 *i* *m*and on each symbol performs

### S←S−1, I←I+ 1.

At the following separator it tests whether *S*= 0 using the zero-mask. If *S*= 0 (i.e. *sm−*1=*im*), it moves to a state *qj* =

and on each symbol of the *jm*-block simply does *S←S*+ 1 (leaving *I* unchanged); since *S* starts at 0, after 0 *j* *m*we have *S*=*jm*, and at the next separator it resets *I←*0 and returns to *q*edge, so *sm*=*jm*. If instead *S̸*= 0, the machine moves to a state*qj̸* =. On each symbol of the*jm*-block it now performs

*S←S*+ 1*, I←I−*1if*I̸*= 0*,*and does nothing if*I*= 0*,*

where the choice again uses only the zero-mask bit of *I*. Initially *I*=*im*, so after min*{im,jm}* symbols we have added that many units back to*S*and subtracted them from*I*. Because*jm≥im*, by the end of the*jm*-block we have*I*= 0and

*S*= (*sm−*1*−im*) +*im*=*sm−*1*,*

and at the next separator we return to *q*edgewith *S*=*sm*=*sm−*1and *I*= 0. Thus the invariant is preserved across every edge.

After the last edge the machine enters a state*qt*with*S*=*sn*and*I*= 0. On the target block0 *t* it updates

### S←S−1, T←T+ 1

on each symbol, so at the end *S*=*sn−t*. On the end-of-input marker it moves to an accepting state iff the zero-mask bit of *S*is0, i.e. iff*S*= 0, which is equivalent to*sn*=*t*.

**Lemma 3.***Let M be a real-time counter machine. There exists a one-layer MLP RNN with log precision that recognizes* *the same language asM.*

*Proof.* We divide partition the hidden state *ht*into three parts: *st*, which encodes a finite state as a one-hot vector, *c* + *t*, which encodes the value of positive counters, and *c* *−* *t*, which encodes the absolute value of negative counters. Define *qt*as the state encoded by *st*and *ct*=*c* + *t−c* *−*

*t*. We will show how to construct an MLP *f* such that, applying *ht*=*f*(*ht−*1*,xt*), we will
have*st,c* + *t,c* *−* *t*such that

*q* *t*=*δ*(*qt−*1*,σt,z*(*ct−*1)) *c* *t*=*u*(*qt−*1*,σt,z*(*ct−*1))(*ct−*1)*.*

We first construct a ReLU network to compute *z*(*ct−*1) as a function of *c* + *t−*1and *c* *−* *t−*1using the standard construction for equality checks with a tolerance of *ϵ*= 1*/*3 (Yang et al.,2026, Section 4.7). We then construct ReLU networks to compute *δ* and *u* (with output encoded as one-hot vectors) given *qt−*1*,σt,z*(*ct−*1), which is possible since both are finite lookup tables (Yang et al.,2026, Section 4.8). We compute *z* with *δ* to get a one-hot encoding of *qt*=*δ*(*qt−*1*,σt,z*(*ct−*1)), i.e., our updated state vector *st*. Similarly, we compose *z* with *u* to get a one-hot encoding of the update function *ut*=*u*(*qt−*1*,σt,z*(*ct−*1)) to apply to the counters. The new values of *c* + *t*and *c* *−* *t*can be obtained by a ReLU network that computes several continuous piecewise-linear functions of *c* + *t−*1*,c* *−* *t−*1in parallel for different updates *u∈{×*0*,*+0*,*+1*,−*1*}* (Yang et al.,2026, Section

4.1) and then applies a conditional selector for*u*=*ut*(Yang et al.,2026, Section 4.9). We can then accept or reject a string of length *n* based on *sn,c*
+ *n,c* *−* *n*using an MLP. The construction uses log precision because*qt*is in a finite set and*c* + *t,c* *−* *t*=*O*(*n*).

## B. Lower Bounds for DPLR LRNNS

This section contains four simulation results, organized by architecture and target computation:

### •RWKV-7 simulates WFAs overQ(SubsectionB.2).

### •RWKV-7 computes iterated3×3matrix multiplication overQ(SubsectionB.3).

### •DeltaNet simulates WFAs overQ(SubsectionB.4).

### •DeltaNet computes iterated3×3matrix multiplication overQ(SubsectionB.5).

### B.1. Preliminaries

**Definition 13**(WFA Path Form)**.**Let *⟨*K*,⊕,⊗⟩* be a semiring. A weighted finite automaton (WFA) is a tuple *A*= (*Q,*Σ*,{Mσ}σ∈*Σ*,α,ω*) where *Q*=*{*1*,...,n}* is a finite state set, each *Mσ∈*K *n×n* is a transition matrix, and *α,ω∈*K *n*

are initial/final weight vectors, respectively. The WFA assigns a score to a string*w*=*w₁ ···wn∈*Σ *∗* via ! M O *f* *A*(*w*) = *απ*1*⊗* [*Mwi*]*πk,πk*+1*⊗ωπn,* *π*=*{*(*sk,tk*)*}nk*=1*i*=1

where*π*iterates over all paths of states of length*n*.

Equivalently, defining matrix multiplication over*⟨*K*,⊕,⊗⟩*, the computation of a WFA can be defined according to

*f* *A*(*w*) =*α* *⊤* *⊗Mw*1*⊗...⊗Mwn⊗ω∈*K*.*

**Definition 14**(Streaming iterated 3*×*3 matrix multiplication over Z)**.**The input is a token stream of length 9*N* over alphabet Z, partitioned into *N* consecutive blocks of length 9. The *t*-th block encodes a matrix *A*

(*t*) *∈*Z³ *×*3 in row-major
order:

(*t*) (*t*) (*t*) (*t*) (*t*) (*t*) (*t*) (*t*) (*t*)
*A₁,*1*,A₁,*2*,A₁,*3*, A₂,*1*,A₂,*2*,A₂,*3*, A₃,*1*,A₃,*2*,A₃,*3*.*

The task is to output the product

*PN* :=*A*

(1) *A*
(2) *···A*
(*N*) *∈*Z³ *×*3
(equivalently, its9entries).

### B.2. Simulating WFAs with RWKV-7

Lete*i∈*K *d* denote the*i*th standard basis*column*vector.

**Definition 15**(RWKV-style transition matrix (Peng et al.,2025))**.**Fix a dimension *d*. Given *w,a,κ∈*K *d* and a scalar *λ∈*K, define *A*(*w,a,κ*;*λ*) := diag(*w*)*−λκ*(*a⊙κ*) *⊤* *,*

### where⊙is elementwise product.

To handle WFA transition matrices, we use a primitive that overwrites one coordinate by a dot product of the current state against a prescribed coefficient vector.

**Definition 16**(Dot-product overwrite matrix)**.**Fix a dimension *d*, an index dst*∈{*1*,...,d}*, and a coefficient vector *c∈*K *d* with*c*dst= 0. Define *U*(dst;*c*) :=*I−*edste *⊤* dst+*c*e *⊤* dst*∈*K *d×d* *.*

Equivalently,*U*(dst;*c*)is the identity matrix with itsdst-th column replaced by*c*.

**Lemma 4**(Effect on row vectors)**.***For any row vectorr∈*K¹ *×d* *,*

X*d* (*rU*(dst;*c*))dst= *rici,*(*rU*(dst;*c*))*j*=*rj*(*j̸*= dst)*.* *i*=1

**Lemma 5**(Dot-product overwrite as an RWKV transition)**.***Fixd,*dst*, andc∈*K *d* *withc*dst= 0*. Set*

### w=1, a=edst, κ=edst−c, λ= 1.

### ThenA(w,a,κ; 1) =U(dst;c).

*Proof.*Since*a*=edst, we have*a⊙κ*=*κ*dstedst. Because*c*dst= 0,*κ*dst= 1, so*a⊙κ*=edst. Thus

*A*=*I−κ*e *⊤* dst=*I−*(edst*−c*)e *⊤* dst=*I−*edste *⊤* dst+*c*e *⊤* dst=*U*(dst;*c*)*.*

We now show that any *n×n* matrix can be applied to a row vector using Θ(*n*) overwrite steps, provided we have *n* additional scratch coordinates.

**Lemma 6**(Apply an arbitrary *n×n* matrix using 2*n* overwrites)**.***Let P∈*K *n×n* *. Consider row vectors in dimension* 2*n* *written as* [*x|s*] *with x,s∈*K¹ *×n* *. There exist m*:= 2*n overwrite matrices G₁*(*P*)*,...,Gm*(*P*)*∈*K² *n×*2*n* *such that for* *allx,s,* [*x|s*]*G₁*(*P*)*···Gm*(*P*) = [*xP|xP*]*.*

(*j*) 2*n* (*j*)
*Proof. Phase 1 (compute xP into scratch).*For each *j∈{*1*,...,n}*, define *c ∈*K by *ci*=*Pi,j*for 1*≤i≤n* and

(*j*) (*j*) *c* P *i*= 0 for *n*+ 1*≤i≤*2*n*. Let *Gj*(*P*) :=*U*(*n*+*j*;*c*). By Lemma4, this overwrites scratch coordinate *n*+*j* with *n* *i*=1*xiPi,j*= (*xP*)*j*, leaving all other coordinates unchanged. After*n*steps,[*x|s*]*G₁*(*P*)*···Gn*(*P*) = [*x|xP*]. *Phase 2 (copy scratch back to main).*For each *j∈{*1*,...,n}* let *d*
(*j*) :=e*n*+*j∈*K²
*n* and set *Gn*+*j*(*P*) :=*U*(*j*;*d*

(*j*) ).
This overwrites main coordinate *j* with the dot product against e*n*+*j*, i.e. the current scratch coordinate *n*+*j*= (*xP*)*j*, leaving scratch unchanged. Thus after*n*more steps,[*x|xP*]*Gn*+1(*P*)*···G₂n*(*P*) = [*xP|xP*].

B.2.1. SIMULATION THEOREM **Theorem 9**(RWKV-7 simulates weighted finite automata over Q)**.***Fix* K=Q*. Let A*= (*Q,*Σ*,{Mσ},α,ω*) *be an n-state* *WFA over*K*(Definition13), and letw*=*w₁ ···wTbe an input word. Then there exists a*4*-layer RWKV-7 network whose* *output at positionTequalsfA*(*w*) =*αMw*1*···MwTω∈*Q*.* *Proof.* We use the “3-layer router + 1-layer multiplicative simulator” template ofPeng et al.(2025) (Appendix D), instantiated with a block length*m*:= 2*n*and arithmetic dimension2*n*. **Block parameters.**Let *m*:= 2*n*. Write each position as *t*=*ℓm*+*τ* with *ℓ≥*0 and 1*≤τ≤m*. Introduce a padding token*⊥*and set*wt*:=*⊥*for all*t≤*0, with*M⊥*:=*I*. Let the*previous block*string be
˜*w*

(*ℓ*) :=*w*(*ℓ−*1)*m*+1*···wℓm∈*(Σ*∪{⊥}*)
*m* *,*

which is well-defined for all*ℓ*due to padding.

**Offline factorization of block products.**For each length-*m*block string˜*w*= ˜*w₁ ···*˜*wm*, define the block product

*P*( ˜*w*) :=*M*˜*w*1*···M*˜*wm∈*K *n×n* *.*

Apply Lemma6to obtain overwrite matrices*G*˜*w,*1*,...,G*˜*w,m∈*K² *n×*2*n* such that for all*x,s∈*K¹ *×n*,

Y *m* [*x|s*] *G*˜*w,i*= [*xP*( ˜*w*)*|xP*( ˜*w*)]*.* *i*=1

### Fix one such factorization for each˜w.

**Router (layers 1–3).**Define a lookup tableΞwhose key is

*t*mod 2*m, wt,wt−*1*,...,wt−*(2*m−*1)*,*

### and whose outputs are:

1. the next factor *G*˜*w*(*ℓ*)*,τ*to apply at time *t* (encoded so that the arithmetic layer can set (*w,a,κ*) accordingly, using Lemma5);
2.a “completion” readout vectorˆ*ωt*(defined below).
Both depend only on the key: ˜*w*

(*ℓ*) is contained in the last 2*m* tokens whenever *ℓ≥*1 (and is all padding when *ℓ*= 0), and
*τ* is determined by *t*mod 2*m*. By the finite-window lookup construction ofPeng et al.(2025) (Appendix D; instantiated with block length*m*), a 3-layer RWKV-7 can computeΞat every position and write its outputs into the residual stream.

**Arithmetic layer (layer 4).**Use a single wkv head of dimension 2*n*. Initialize the first row of the wkv matrix state using a single additive update on the first token: set *v₁* =e₁ and *k*˜1= [*α|α*], so the state becomes e *⊤* 1*k* ˜ 1. For all *t≥*2, set the additive term to0(i.e.*vt*= *k*˜*t*=0), so evolution is purely multiplicative: *St*=*St−*1*At*.

At each time *t*, the router specifies the factor *G*˜*w*(*ℓ*)*,τ*. Choose RWKV transition parameters so that *At*=*G*˜*w*(*ℓ*)*,τ*; since each factor is an overwrite matrix, this is possible by Lemma5.

**Completion vector and correctness.**Let *pt*denote the first row of the arithmetic layer’s wkv state *St*. As inPeng et al. (2025) (Appendix D), one checks by induction that for*t*=*ℓm*+*τ*,

*ℓ* Y *−*2Y*m*Y*τ* *pt*= [*α|α*] *G*˜*w*(*b*)*,iG*˜*w*(*ℓ−*1)*,i,* *b*=0 *i*=1 *i*=1

with the convention that empty products are*I*.

### Define the within-block prefix product

*Rt*:=*Mwℓm*+1*···Mwt, vt*:=*Rtω∈*K *n×*1 *.*

### Now define the completion vector:

Y *m* *v* *t* !

ˆ*ωt*:= *G*˜*w*(*ℓ−*1)*,i· ∈*K² *n×*1 *.* 0 *i*=*τ*+1

This ˆ*ωt*is a fixed function of the router key (position mod 2*m* and last 2*m* tokens), hence it can be returned by the lookup tableΞ.

Finally, compute the scalar readout*pt*ˆ*ωt*: *pt*ˆ*ωt*=*αMw*1*···Mwtω*

using Lemma6on each completed block. In particular, at*t*=*T*the RWKV-7 output equals*fA*(*w*).

**B.3. Iterated**3*×*3**Matrix Multiplication with RWKV-7** **Theorem 10**(RWKV-7 computes iterated 3*×*3 matrix products over Q)**.***Fix* K=Q*. There exists a* 4*-layer RWKV-7* *network such that, given the length-*9*N stream encoding A*
(1) *,...,A*
(*N*) *, the network outputs the* 9 *entries of PN* :=
*A*

(1) *A*
(2) *···A*
(*N*) *(equivalently,*vec(*PN*)*∈*Q⁹*) at the final positiont*= 9*N.*
*Proof sketch; full proof follows later.* Layers 1–2 compute the position modulo 18. This yields (i) the*within-block offset* *τ∈{*1*,...,*9*}* and (ii) a*block-parity bit b∈{*0*,*1*}* that we use to alternate between two halves of an arithmetic state. Layer 3 stores the last 18 tokens (two consecutive length-9 blocks) and, from the key (*t*mod 18*, wt,...,wt−*17), look up the RWKV transition parameters needed at time *t*. Concretely, at each token it outputs (in the residual stream) the *destination index*and*coefficient vector*for a single dot-product overwrite update implementing one coordinate of the map *x7→xB*(*A* (*ℓ−*1) ), where *A* (*ℓ−*1) is the*previous*block’s matrix. (We treat the pre-sequence padding as “block 0” encoding *A*

(0) :=*I₃*.) Layer 4 maintains an 18-dimensional row state [*x*
(0) *|x*
(1)]*∈*Q¹ *×*18
inside the first row of a wkv matrix state, where each half has length 9. During each length-9 block, layer 4 applies*nine*dot-product overwrites that compute a new vector *x*new=*x*old*B*(*A* (*ℓ−*1) ) in the*inactive*half, using the parity bit *b* to select which half is source vs. destination. At the final position *t*= 9*N*, the destination half contains vec(*A*

(1) *···A* (*N−*1)
). A*completion readout*(computed by the router from the final finite-window key, which contains the entire last block and hence determines *A*

(*N*) ) outputs the 9 coordinates
ofvec(*A*

(1) *···A*
(*N*) )via nine dot products.
*Proof.* We follow the same “router (layers 1–3) + multiplicative arithmetic (layer 4)” template used in the RWKV-7 regular-language construction (Appendix D ofPeng et al.(2025)).

**Step 1: reduce** 3*×*3 **matrix multiplication to a** 9**-dimensional linear recurrence.**Let vec :Q³ *×*3 *→*Q¹ *×*9 be row-major vectorization: vec(*P*) := (*P₁,*1*,P₁,*2*,P₁,*3*,P₂,*1*,P₂,*2*,P₂,*3*,P₃,*1*,P₃,*2*,P₃,*3)*.*

Define the*block-diagonal embedding*   *A*0 0  9*×*9 *B*(*A*) :=  0*A*0  *∈*Q*.* 0 0*A*

Right-multiplication by*A*acts independently on the three rows of*P*, hence

vec(*PA*) = vec(*P*)*B*(*A*)for all*P,A∈*Q³ *×*3

*.*(2)
Let*Pt*:=*A*

(1) *···A*
(*t*) and*xt*:= vec(*Pt*). Then
*x₀* = vec(*I₃*)*, xt*=*xt−*1*B*(*A*

(*t*) )*.*
Thus it suffices to maintain the9-dimensional state*xt*and repeatedly apply the linear map*x7→xB*(*A*).

**Step 2: implement** *x7→xB*(*A*) **with dot-product overwrites in dimension** 18**.**We use the dot-product overwrite primitive: for any destination coordinate dst and coefficient vector *c* with *c*dst= 0, the overwrite matrix *U*(dst;*c*) replaces exactly one coordinate of a row vector with a dot product (Lemmas4and5). In particular, each overwrite matrix can be realized as a*single*RWKV multiplicative transition (Lemma5).

We maintain an arithmetic row state in dimension18written as

*p*= [*x*

(0) *|x*
(1)]*, x*
(0) *,x*
(1) *∈*Q¹ *×*9 *,*
together with a bit *b∈{*0*,*1*}* indicating the*active*half *x*

(*b*) (source for dot products). Let *π*(*i,j*) := 3(*i−*1) +*j* be the
row-major index map for*i,j∈{*1*,*2*,*3*}*.

Fix a3*×*3matrix*A*. The coordinates of*xB*(*A*)are length-3dot products:

X3 (*xB*(*A*))*π*(*i,j*)= *xπ*(*i,k*)*Ak,j.*(3) *k*=1

We compute these9values into the inactive half*x* (1*−b*) using9overwrite steps. For each pair(*i,j*)*∈{*1*,*2*,*3*}* 2, define:

### •Destination coordinatein the inactive half:

### dst(b;i,j) := 9(1−b) +π(i,j)∈{1,...,18}.

•*Coefficient vectorc* (*b*;*i,j*)

(*A*)*∈*Q¹⁸ supported only on the three active coordinates corresponding to row*i*:
*c* (*b*;*i,j*)

(*A*)9*b*+*π*(*i,k*):=*Ak,j*(*k*= 1*,*2*,*3)*, c*
(*b*;*i,j*)

(*A*)*u*:= 0otherwise*.*
Since dst(*b*;*i,j*) lies in the*inactive*half, we have *c* (*b*;*i,j*)

(*A*)dst(*b*;*i,j*)= 0, so the overwrite is well-formed. Applying
*U*(dst(*b*;*i,j*);*c* (*b*;*i,j*)

(*A*)) overwrites precisely the destination coordinate with the dot product of the current row state
against *c* (*b*;*i,j*)

(*A*) (Lemma4), which equals the right-hand side of (3). Moreover, because each coefficient vector is
supported only on the active half, these overwrites do not depend on (and hence do not interfere with) already-written coordinates in the inactive half.

If we apply these overwrites for all (*i,j*) (in any order, e.g. row-major order), then the inactive half becomes *x*

(*b*) *B*(*A*)
while the active half remains unchanged:

Y [*x*

(0) *|x*
(1)]*7−→*[*x*
(0) *|x*
(1)]*· U*(dst(*b*;*i,j*);*c*
(*b*;*i,j*)

(*A*)) = [*x*
(0) *|x*
(1)]with*x* (1*−b*)
*←x*

(*b*) *B*(*A*)*.*(4)
(*i,j*)

**Step 3: align the** 9 **overwrites with the token stream using a 3-layer router.**We now explain how layers 1–3 produce, at each time*t*, exactly the overwrite parameters needed by layer 4.

Partition the input stream into *N* blocks of length 9. For convenience, define an additional “block 0” consisting of padding symbols before the sequence; the router will interpret this as encoding *I₃* and we write *A*

(0) :=*I₃*. Let *A*
(*ℓ*) be the matrix
encoded by block *ℓ* for *ℓ*= 1*,...,N*. At any position *t* in block *ℓ* (so *ℓ∈{*1*,...,N}*), the model will apply the update
corresponding to the*previous*matrix*A* (*ℓ−*1) (one-block delay).

**Computing** (*τ,b*) **from position.**Let *τ∈ {*1*,...,*9*}* be the within-block offset and let *b∈ {*0*,*1*}* be the parity bit indicating whether the active half is the first (*b*= 0) or second (*b*= 1) half. Both are determined by *t*mod 18: specifically, *τ*= 1 + ((*t−*1) mod 9) and *b* is the indicator of whether *t*mod 18*∈{*10*,...,*18*}*. ByPeng et al.(2025, Lemma 6) instantiated with*n*= 9, a 2-layer RWKV-7 can output*t*mod 18, hence can provide*τ*and*b*to deeper layers.

**Recovering the previous block.**At time *t* in block *ℓ*, the last 18 tokens (*wt,wt−*1*,...,wt−*17) contain the entire previous block*ℓ−*1(all9tokens encoding*A* (*ℓ−*1) ), together with a prefix of the current block. Therefore, the tuple

### tmod 18, wt,wt−1,...,wt−17

uniquely determines the matrix entries of *A* (*ℓ−*1) and the current offset *τ*. ByPeng et al.(2025, Lemma 7) instantiated with *n*= 9, a 3-layer RWKV-7 can implement an arbitrary lookup table on this finite key and write its outputs into the residual stream.

**What the router outputs.**Define a lookup tableΞkeyed by(*t*mod 18*, wt,...,wt−*17)whose output at time*t*is:

1. the destination index dst(*b*;*i,j*) and coefficient vector *c*
(*b*;*i,j*) (*A* (*ℓ−*1) ) for the overwrite corresponding to *τ*=*π*(*i,j*); and

2.(optionally) the bit*b*for the final readout.
The existence of such a lookup table is immediate because the key space is finite. By the previous paragraph, layers 1–3 can computeΞat every position.

Finally, using the weight-preparation linear maps in the time-mixing block, layer 4 converts (dst*,c*) into RWKV transition parameters realizing the overwrite matrix*U*(dst;*c*)via Lemma5.

**Step 4: arithmetic in layer 4 and correctness.**Layer 4 uses a single wkv head of head dimension 18. Initialize the first row of the wkv matrix state at the first token by setting

*v₁* =e₁*, k*˜1= [vec(*I₃*)*|*0]*∈*Q¹ *×*18 *,*

so the wkv state becomes e *⊤* 1*k* ˜ 1and its first row equals *p₁* = [vec(*I₃*)*|*0]. For all subsequent positions *t≥*2, set the additive term to0(i.e.*vt*= *k*˜*t*=0), so the first row evolves purely multiplicatively as*pt*=*pt−*1*At*.

At each position *t* in block *ℓ*, the router provides the overwrite parameters corresponding to the matrix *A* (*ℓ−*1) and the overwrite index *τ*. By the discussion above, we choose *At*=*U*(dst;*c*) so that exactly one coordinate of *pt−*1is overwritten. After consuming all9tokens of block*ℓ*, equation (4) implies that the inactive half has been filled with

*x* (1*−b*) =*x*

(*b*) *B*(*A* (*ℓ−*1)
)*.*

By (2), this corresponds to right-multiplying the current3*×*3product by*A* (*ℓ−*1).

By induction over blocks, after finishing block*ℓ*the destination half contains

vec *A*

(1) *A*
(2) *···A* (*ℓ−*1)
*.*

In particular, after finishing the final block *ℓ*=*N*, the destination half contains vec(*PN−*1) where *PN−*1:=*A*

(1) *···A* (*N−*1)
. We now remove the need for an additional padding block by applying the final multiplication by *A*

(*N*) in the*readout*at the
last token.

(0) (1) 1*×*18
**Completion readout at** *t*=*T*= 9*N***.**Let *T*:= 9*N*. Let *pT*= [*xT|xT*]*∈*Q denote the arithmetic layer’s row state at time *T*, and let *bT∈{*0*,*1*}* be the block-parity bit computed from *T*mod 18. Define the destination-half index (*hT*) *hT*:= 1*−bT∈{*0*,*1*}*(equivalently,*hT*=*N*mod 2). Then*xT*= vec(*PN−*1).

Since *T* is the last token of block *N*, the router key (*T*mod 18*, wT,...,wT−*17) contains the entire block *N* and hence determines*A*

(*N*). For each(*i,j*)*∈{*1*,*2*,*3*}*
2, define the completion vector

X3 (*i,j*) (*N*) 18 ˆ*ωT*:= *A* *k,j* *e₉hT*+*π*(*i,k*)*∈*Q*,*where*π*(*i,j*) = 3(*i−*1) +*j.* *k*=1

Then X3 (*i,j*) (*hT*) (*N*) *pT*ˆ*ωT*= *x* *T,π*(*i,k*) *A* *k,j* = vec(*PN*)*π*(*i,j*)*.* *k*=1 (*i,j*) Each ˆ*ωT*is a fixed function of the router key, so it can be returned by the finite-window lookup in layer 3 at time *T*. Using (*i,j*) nine parallel readouts at the final position (one per (*i,j*)), the network outputs the nine scalars *pT*ˆ*ωT*in row-major order, which equalvec(*PN*).

### B.4. Simulating WFAs with DeltaNet

This subsection shows that a 4-layer stacked DeltaNet can simulate any weighted finite automaton over Q. Throughout this subsection we fixK=Qand interpret all arithmetic inQ.

B.4.1. DELTANET MEMORY UPDATE AND THE MULTIPLICATIVE STEP MATRIX We consider a DeltaNet layer with matrix state*St∈*K
*d×d* updated by

*St*=*St−*1*I−βtktkt⊤*+*βtvtkt⊤,*(5)

where *βt∈*K and *kt,vt∈*K *d* are computed from the layer input *xt*(via linear maps / an MLP, as in standard stacked DeltaNet blocks).

We will repeatedly use the*multiplicative step matrix*

*H*(*β,k*) :=*I−βkk* *⊤* *∈*K *d×d*

*.*(6)
### Whenvt=0, the update is purely multiplicative:

*St*=*St−*1*H*(*βt,kt*)*.*

B.4.2. ROUTER PRIMITIVES:POSITION MODULO AND TOKEN BUFFERING **Notation.**Let *ei*denote the standard basis*column*vector. We assume the input stream contains an explicit BOS token at *t*= 1. **Computing***t*mod 2*m***.**The router needs a cyclic write index *t*˜*≡t*(mod 2*m*)to address a2*m*-slot token buffer. **Lemma 7**(DeltaNet can compute position modulo 2*m*)**.***For any positive integer m, there exists a* 2*-layer stacked DeltaNet* *whose output includes a one-hot encoding of t*˜:=*t*mod 2*mat every positiont.* *Proof.* The construction mirrors a standard “two-reflections-make-a-rotation” argument. Layer 1 produces the parity bit of *t* (and can also detect *t*= 1 via the BOS token). Layer 2 maintains a 2-dimensional continuous state that advances by a fixed angle every two steps: it alternates between two Householder reflections of the form *I−*2*κκ*
*⊤*, which is exactly *H*(*β,k*) with *β*= 2 and *k*=*κ* (and *vt*=0). The state therefore cycles through 2*m* distinct values, one for each residue class modulo 2*m*. A finite MLP on top of a bank of linear readouts can map these 2*m* discrete states to a one-hot encoding of *t*˜.

**Exact token buffering by column overwrite.**We store the last 2*m* tokens by overwriting one designated column per step.

**Lemma 8**(Exact column overwrite)**.***Fixdand an index*dst*∈{*1*,...,d}. Ifβt*= 1*andkt*=*e*dst*, then*

*St*=*St−*1(*I−e*dst*e* *⊤* dst) +*vte* *⊤* dst*,*

*so all columns ofSt−*1*are unchanged except column*dst*, which becomes exactlyvt.*

*Proof.* Right-multiplying by (*I−e*dst*e* *⊤* dst) zeros column dst and leaves all other columns unchanged; adding *vte* *⊤* dstwrites *v* *t*into that column.

**Buffer layout.**Let*L*:= 2*m*. In the buffer layer, use a DeltaNet head dimension

### dbuf:= (|Σ|+ 1) +L.

### We interpret:

•the first*|*Σ*|*+ 1coordinates as a one-hot token space (including padding/BOS as needed);

### •the lastLcoordinates as buffer-slot selectors.

At time *t*, let *t*˜*∈{*1*,...,L}* be the residue class from Lemma7. Write the current token into the *t*˜-th buffer slot by setting

*βt*= 1*, kt*=*e*(*|*Σ*|*+1)+*t*˜*, vt*=*ewt∈*K *d* buf *.*

By Lemma8, column (*|*Σ*|*+ 1) + *t*˜ stores exactly the one-hot token vector *ewt*. Using *L* read heads with fixed queries *q*

(*j*) =*e*(*|*Σ*|*+1)+*j*, the layer can read out all *L* stored tokens (each as a one-hot vector in the first *|*Σ*|*+ 1 coordinates) into the residual stream. **Finite lookup.**Given the one-hot encoding of *t*˜ and the last 2*m* tokens (read from the buffer), the key space is finite. Thus, an (exponentially wide) feedforward network can implement an arbitrary lookup table on this key and output: (i) the next arithmetic-step parameters(*β,k*)for Layer 4, and (ii) a readout vector used to decode the desired scalar output.
B.4.3. ARITHMETIC PRIMITIVES REALIZABLE BY MULTIPLICATIVEDELTANET STEPS We now show that the multiplicative matrix *H*(*β,k*) =*I−βkk*
*⊤* can realize the linear operations needed to apply arbitrary matrices.

**Lemma 9**(Coordinate scaling / clearing)**.***Fixj∈{*1*,...,d}ands∈*K*. Letk*=*ejandβ*= 1*−s. Then*

*H*(*β,ej*) =*I−*(1*−s*)*eje* *⊤* *j*

*scales coordinatejbysand leaves all other coordinates unchanged. In particular,s*= 0*clears coordinatej.*

*Proof.*For any row vector*r∈*K¹ *×d*, *rH*(*β,ej*) =*r−*(1*−s*)(*rej*)*e* *⊤* *j,*

so(*rH*)*j*=*rj−*(1*−s*)*rj*=*srj*and(*rH*)*ℓ*=*rℓ*for*ℓ̸*=*j*.

**Definition 17**(Unit transvection)**.**Forsrc*̸*= dstdefine the (column) unit transvection

*T*(src*→*dst) :=*I*+*e*src*e* *⊤* dst*.*

Right-multiplication by *T*(src*→*dst) updates a row vector *r* by *r*dst*←r*dst+*r*srcand leaves all other coordinates unchanged.

**Lemma 10**(Unit transvection as three multiplicative DeltaNet steps)**.***Assume* <u>1</u> 2 *,* <u>1</u> 3 *∈*K*. Fix* src*̸*= dst *and define (in* *dimensiond)* *u*:=*e*src+*e*dst*, w*:=*e*src+ 2*e*dst*.*

*Then* *H*(2*,u*)*H* <u>1</u> 2 *,e*src*H* <u>1</u> 3 *,w* =*T*(src*→*dst) =*I*+*e*src*e* *⊤* dst*.*

*Proof.* All three factors act as the identity outside the 2D subspace spanned by *e*src*,e*dst, so it suffices to verify the 2*×*2 restriction. In the ordered basis(*e*src*,e*dst), we have:

|1|2 2|
|---|---|
|2|3 3|

*−*1*−*2<u>1</u>0<u>1</u>*−* *H*(2*,*(1*,*1)) =*, H*( 2 *,*(1*,*0)) =*, H*( 3 *,*(1*,*2)) = <u>2 1</u> *.* *−*2*−*1 0 1 *−* 3 *−* 3

Multiplying gives *−*1*−*2 1 2 03 2 *−* 2 31 1*⊤* <u>2 1</u> = =*I*+*e*src*e*dst*.* *−*2*−*1 0 1 *−* 3 *−* 3 0 1

Embedding back into the full*d*-dimensional space proves the claim.

**Lemma 11**(Scaled add via a temp register)**.***Assume* <u>1</u> 2 *,* <u>1</u> 3 *∈*K*. Fix distinct indices* src*,*dst*,*tmp *and a scalar λ∈*K*. There* *exists a length-*8 *sequence of multiplicative matrices of the form H*(*β,k*) *such that for every row vector r with r*tmp= 0*,* *the resulting row vectorr* *′* *satisfies:*

*r* dst *′* =*r*dst+*λr*src*, rj′*=*rj*(*j̸*= dst)*, r*tmp *′* = 0*.*

*Proof.*Use the following four-stage program, each stage implementable by multiplicative DeltaNet steps:

1.Copysrcintotmpby a unit transvection*T*(src*→*tmp), implemented using3steps via Lemma10.
2.Scaletmpby*λ*using one coordinate scaling: apply*H*(1*−λ,e*tmp)(Lemma9).
3.Addtmpintodstby a unit transvection*T*(tmp*→*dst), implemented using3steps via Lemma10.
4.Cleartmpby scaling it by0: apply*H*(1*,e*tmp)(Lemma9).
Starting from *r*tmp= 0, the net effect is *r*dst*←r*dst+*λr*srcand tmp returns to 0. The total number of multiplicative steps is3 + 1 + 3 + 1 = 8.

**Applying an arbitrary matrix with scratch coordinates.**We now show that any *n×n* matrix can be applied to a row vector using*O*(*n²*)multiplicative DeltaNet steps, given*n*scratch coordinates and one additional temp coordinate.

**Lemma 12**(Apply any *n×n* matrix with scratch in *O*(*n²*) multiplicative steps)**.***Assume* <u>1</u> 2 *,* <u>1</u> 3 *∈*K*. Let P∈*K *n×n* *.* *Consider row vectors in dimension*2*n*+ 1*written as*

[*x|s|t*]*, x,s∈*K¹ *×n* *, t∈*K*.*

*There exists an explicit sequence of* *m*:= 8*n²* + 5*n*+ 1

*matricesG₁*(*P*)*,...,Gm*(*P*)*∈*K (2*n*+1)*×*(2*n*+1) *, each of the formH*(*β,k*)*, such that for allx,s,t,*

### [x|s|t]G₁(P)···Gm(P) = [xP|xP|0].

### Proof.Let the coordinates be indexed as:

### main:1,...,n,scratch: n+ 1,...,2n,temp:2n+ 1.

**Phase 1 (clear scratch and temp).**For each scratch coordinate*n*+*j*apply a clear:

*H*(1*,en*+*j*) (*j*= 1*,...,n*)*,*

and clear temp by*H*(1*,e₂n*+1). This uses*n*+ 1steps and yields[*x|*0*|*0].

**Phase 2 (accumulate** *xP* **into scratch).**For each destination *j∈{*1*,...,n}* and source *i∈{*1*,...,n}*, update scratch
coordinate*n*+*j*by (scratch*j*)*←*(scratch*j*) +*Pi,j·*(main*i*)

using Lemma11with src =*i,*dst = *n*+*j,*tmp = 2*n*+ 1*, λ*=*Pi,j.*

Each scaled add costs 8 multiplicative steps and leaves temp equal to 0. After all *n²* updates, scratch equals *xP* while main remains*x*.

**Phase 3 (clear main).**Clear the main coordinates by applying *H*(1*,ei*) for *i*= 1*,...,n*. This uses *n* steps and yields [0*|xP|*0].

**Phase 4 (copy scratch back to main).**For each *j*= 1*,...,n*, add scratch coordinate *n*+*j* into main coordinate *j* using a unit transvection *T*(*n*+*j→j*), implemented by 3 multiplicative steps via Lemma10. Since main is 0, after these 3*n* steps we have main=*xP*and scratch remains*xP*.

Summing the step counts gives (*n*+ 1) + 8*n²* +*n*+ 3*n*= 8*n²* + 5*n*+ 1*.*

B.4.4. SIMULATION THEOREM:A4-LAYERDELTANET SIMULATES ANYWFA **Theorem 11**(DeltaNet simulates weighted finite automata)**.***Fix* K=Q*. Let A*= (*Q,*Σ*,{Mσ},α,ω*) *be an n-state WFA* *over* K *(Definition13), and let w*=*w₁ ···wTbe an input word. Then there exists a* 4*-layer stacked DeltaNet whose scalar* *output at every positiontequals*
*αMw*1*Mw*2*···Mwtω∈*K*.*

### In particular, att=Tthe output equalsfA(w).

*Proof sketch; full proof follows later.* We use the same finite-window*router + arithmetic*template as in Theorem9(compute *t*mod 2*m*, buffer the last 2*m* tokens, and look up the next arithmetic step), but implement these router components using DeltaNet primitives.

The arithmetic layer now uses only multiplicative DeltaNet steps *H*(*β,k*) =*I−βkk* *⊤* (Definition2.3) over K=Q, which guarantees <u>1</u> 2 *,* <u>1</u> 3 *∈*K. Lemma9shows that *H*(*β,k*) can implement coordinate scalings, and Lemma10shows that constant-length products of such steps implement unit transvections *r*dst*←r*dst+*r*src. Combining these primitives, Lemma12gives an explicit program applying an arbitrary *n×n* matrix in *m*= 8*n²* + 5*n*+ 1 multiplicative steps (using scratch space and one temporary coordinate). The router streams the corresponding (*βt,kt*) parameters to the arithmetic layer exactly as in Theorem9, and the same completion-vector decoding yields*fA*(*w*).

*Proof.*Let *m*:= 8*n²* + 5*n*+ 1

be the step budget from Lemma12. For each position *t≥*1, write *t*=*ℓm*+*τ* with *ℓ*=*⌊*(*t−*1)*/m⌋≥*0 and 1*≤τ≤m*.

**Offline factorization of block products.**Introduce a padding symbol *⊥* and set *wt*=*⊥* for *t≤*0 with *M⊥*:=*I*. For any length-*m*block string˜*w*= ˜*w₁ ···*˜*wm∈*(Σ*∪{⊥}*) *m*, define the block product

*P*( ˜*w*) :=*M*˜*w*1*M*˜*w*2*···M*˜*wm∈*K *n×n* *.*

Apply Lemma12to*P*( ˜*w*)to obtain a fixed sequence

*G*˜*w,*1*,...,G*˜*w,m∈*K (2*n*+1)*×*(2*n*+1) *, G*˜*w,τ*=*H*(*β*˜*w,τ,k*˜*w,τ*)*,*

such that for all[*x|s|t*], Y *m* [*x|s|t*] *G*˜*w,τ*= [*xP*( ˜*w*)*|xP*( ˜*w*)*|*0]*.* *τ*=1 Fix one such factorization for each possible˜*w*.

**Layer 1 (parity / first-position features).**Layer 1 produces a finite-valued encoding that distinguishes the first position (via BOS) and encodes parity. This can be done by a 1D multiplicative recurrence that toggles sign each step, together with the BOS token.

**Layer 2 (compute** *t*mod 2*m***).**By Lemma7, Layer 2 outputs a one-hot encoding of *t*˜:=*t*mod 2*m* in the residual stream.

**Layer 3 (buffer last** 2*m* **tokens and perform a finite lookup).**Set *L*:= 2*m* and build a 2*m*-slot cyclic token buffer using Lemma8, with the buffer layout described earlier. At time*t*, the buffer contains the last2*m*tokens

(*wt,wt−*1*,...,wt−*(2*m−*1))*,*

and Layer 2 provides *t*˜=*t*mod 2*m*. Define a lookup tableΞwhose key is

*t, w* ˜ *t* *,wt−*1*,...,wt−*(2*m−*1)*,*

### and whose outputs are:

(4) (4)
1.the arithmetic parameters(*βt,kt*)specifying the matrix*G*˜*w*(*ℓ−*1)*,τ*to apply in Layer 4 (defined below);
2.a completion/readout vectorˆ*ωt∈*K²
*n*+1 (also defined below).

Since the key space is finite, an MLP can implementΞexactly and write these outputs into the residual stream.

**Layer 4 (arithmetic: purely multiplicative DeltaNet simulation).**Layer 4 uses one DeltaNet head of dimension 2*n*+ 1. We maintain only the*first row*of*St*as the arithmetic row vector.

*Initialization.*Let*p₀* = [*α|*0*|*0]*∈*K¹ *×*(2*n*+1). On the first token, set*S₀* = 0and write*S₁* =*e₁p₀* by choosing

*β₁* = 1*, v₁* =*e₁, k₁* =*p* *⊤*

0*.*
Then*S₁* =*β₁v₁k₁* *⊤* =*e₁p₀*.

*Updates.*For all *t≥*2, set *vt*=0. Let ˜*w* (*ℓ−*1) :=*w*(*ℓ−*1)*m*+1*···wℓm*be the previous*completed*block string (using *⊥*

(4) (4)
padding when*ℓ*= 0), and let*τ*be the within-block offset of*t*. Layer 3 outputs(*βt,kt*)so that

(4) (4)
*H*(*βt,kt*) =*G*˜*w*(*ℓ−*1)*,τ.*

Thus, for*t≥*2we have*St*=*St−*1*G*˜*w*(*ℓ−*1)*,τ*. Let*pt*denote the first row of*St*; then*pt*=*pt−*1*G*˜*w*(*ℓ−*1)*,τ*.

**Correctness via a completion vector.**Unrolling the recurrence, if*t*=*ℓm*+*τ*then *ℓ* Y *−*2Y*m*Y*τ* *pt*= [*α|*0*|*0] *G*˜*w*(*b*)*,sG*˜*w*(*ℓ−*1)*,s,* *b*=0 *s*=1 *s*=1 with the convention that empty products are identity.

### Define the current-block prefix product

*Rt*:=*Mwℓm*+1*···Mwt∈*K *n×n* *, vt*:=*Rtω∈*K *n×*1 *.*

Define the completion vector   Y *m vt* !

ˆ*ωt*:= *G*˜*w*(*ℓ−*1)*,s*0 *∈*K² *n*+1 *.* *s*=*τ*+1 0 This depends only on the router key (position mod2*m*and the last2*m*tokens), hence can be output byΞ.

Now compute the scalar *pt*ˆ*ωt*: multiplying *pt*by the remaining factors *G* Q ˜*w* (*ℓ−*1)*,τ*+1*···G*˜*w*(*ℓ−*1)*,m*completes the application *m* of the previous block product. By Lemma12, the full block product*s*=1*G*˜*w*(*ℓ−*1)*,s*maps the first *n* coordinates *x* to *xP*( ˜*w* (*ℓ−*1) )(and duplicates it into scratch), so telescoping over completed blocks yields

*pt*ˆ*ωt*=*αMw*1*···Mwtω.*

**Readout.**At position *t*, set a DeltaNet query vector *qt*:= ˆ*ωt*(routed from Layer 3) and read *ot*=*Stqt*. Because *St*=*e₁pt*has only the first row nonzero, we have *ot*=*e₁ ·*(*pt*ˆ*ωt*). An output head returns the first coordinate, which equals*αMw*1*···Mwtω*.

**B.5. Iterated**3*×*3**Matrix Multiplication with DeltaNet** We now show that DeltaNet can solve the streaming iterated 3*×*3 matrix multiplication problem (Definition14) over a bounded-precision modulus, using the same router+arithmetic template as Theorem11. The main difference is that the arithmetic state dimension is constant (*n*= 9after vectorization), so the required per-superblock step budget is a constant. **Theorem 12**(DeltaNet computes iterated 3*×*3 matrix products over Q)**.***Fix* K=Q*. There exists a* 4*-layer stacked* *DeltaNet such that, given the stream encoding A*
(1) *,...,A*
(*N*) *∈*Q³ *×*3
*(Definition14), the network outputs the* 9 *entries of*

*PN* :=*A*

(1) *A*
(2) *···A*
(*N*) *∈*Q³ *×*3
### at the final positiont= 9N.

### Proof.Work overK=Q.

**Step 1: reduce right-multiplication to a** 9**-dimensional linear update.**Let vec :Q³ *×*3 *→*Q¹ *×*9 be row-major vectorization. Define the block-diagonal embedding   *A*0 0  9*×*9 *B*(*A*) :=  0*A*0  *∈*Q*.* 0 0*A*

Then for all*P,A∈*Q³ *×*3, vec(*PA*) = vec(*P*)*B*(*A*)*.*

Let*xt*:= vec(*A*

(1) *···A*
(*t*) ). Then
*x₀* = vec(*I₃*)*, xt*=*xt−*1*B*(*A*

(*t*) )*.*
Thus it suffices to maintain a9-dimensional row state and repeatedly apply the linear map*x7→xB*(*A*).

**Step 2: choose a superblock length and step budget.**We will apply Lemma12with *n*= 9, i.e. we maintain an arithmetic state of dimension2*n*+ 1 = 19(main9, scratch9, temp1), and can apply any9*×*9matrix in

*m*arith:= 8*n²* + 5*n*+ 1 = 8*·*81 + 45 + 1 = 694

multiplicative steps.

Each input matrix*A*

(*t*) arrives as9tokens. We group matrices into superblocks of length
### L:= 78,so each superblock has9L= 702tokens.

Since 702*≥*694, we can schedule the 694 arithmetic steps for a superblock and use the remaining 702*−*694 = 8 token times as*identity*steps (i.e. apply*H*(0*,k*) =*I*).

We handle the one-superblock delay by a*completion readout*at the final position *T*= 9*N*, analogous to the RWKV completion-vector trick: the router computes a readout vector that (i) finishes the remaining multiplicative steps for the previous superblock and (ii) applies the final superblock product in the readout.

**Step 3: define superblock transition matrices.**Partition the matrix sequence into superblocks of length *L*= 78, except that the final superblock may be shorter. Let*B*:=*⌈N/L⌉*and let

*r*:=*N−*(*B−*1)*L∈{*1*,...,L}.*

For each superblock index*b < B*define the full-superblock product

Π*b*:=*B*(*A* ((*b−*1)*L*+1) )*···B*(*A* (*bL*) )*∈*Q⁹ *×*9 *.*

For the final superblock define the (possibly partial) product

Π*B*:=*B*(*A* ((*B−*1)*L*+1) )*···B*(*A*

(*N*) )*∈*Q⁹ *×*9
*.*

Then vec(*PN*) =*x₀* Π₁Π₂ *···*Π*B, x₀* = vec(*I₃*)*.*

Partition the matrix sequence into superblocks of length *L*= 78, except that the final superblock may be shorter. Let *B*:=*⌈N/L⌉*and let *r*:=*N−*(*B−*1)*L∈{*1*,...,L}.*

For each superblock index*b < B*define the full-superblock product

Π*b*:=*B*(*A* ((*b−*1)*L*+1) )*···B*(*A* (*bL*) )*∈*Q⁹ *×*9 *.*

For the final superblock define the (possibly partial) product

Π*B*:=*B*(*A* ((*B−*1)*L*+1) )*···B*(*A*

(*N*) )*∈*Q⁹ *×*
*.*

**Step 4: offline factorization of each** Π*b***into multiplicative DeltaNet steps.**By Lemma12(with *n*= 9), for each*full* superblock product Π*b*with *b < B* there exists a length-694 sequence of 19*×*19 matrices of the form *H*(*β,k*). Extend this to a length-702 sequence by appending 8 identity steps. Thus each full superblock *b < B* is associated with a fixed sequence

*Gb,*1*,...,Gb,*702*∈*Q¹⁹ *×*19 *, Gb,τ*=*H*(*βb,τ,kb,τ*)*.*

(We also define*G₀,τ*:=*I*for all*τ*, corresponding toΠ₀ :=*I*.)

By Lemma12(with *n*= 9), for each Π*b*there exists a length-694 sequence of 19*×*19 matrices of the form *H*(*β,k*) that maps [*x|s|t*]*7→*[*x*Π*b|x*Π*b|*0]*.*

Extend this to a length-702 sequence by appending 8 identity steps. Thus each superblock *b* is associated with a fixed sequence *Gb,*1*,...,Gb,*702*∈*Q¹⁹ *×*19 *, Gb,τ*=*H*(*βb,τ,kb,τ*)

such that after applying all702steps,

Y 702 [*x|s|t*] *Gb,τ*= [*x*Π*b|x*Π*b|*0]*.* *τ*=1

Fix one such factorization for each possible superblock token string.

**Step 5: the 4-layer DeltaNet construction (router + arithmetic).***Layers 1–2 (position modulo).*Let *m*tok:= 702 be the token length per superblock. Use Layers 1–2 to compute a one-hot encoding of *t*mod (2*m*tok) =*t*mod 1404 (Lemma7 with*m*=*m*tok) and write it to the residual stream.

*Layer 3 (token buffer + lookup).*Maintain a cyclic buffer of the last 2*m*tok= 1404 tokens using the column overwrite mechanism (Lemma8) with*L*= 1404slots. At time*t*, the last1404tokens contain:

•the entire previous superblock (the last702tokens), encoding the*L*= 78matrices whose product isΠ*b−*1,

•and a prefix of the current superblock.

Define a lookup tableΞkeyed by *t*mod 1404*, wt,wt−*1*,...,wt−*1403

(4) (4)
whose output is the next arithmetic-step parameters (*βt,kt*) specifying the factor *Gb−*1*,τ*to apply at this time (one-superblock delay). Because the key space is finite (and of size at most *m¹⁴⁰⁴ ≤NO*

(1) under *m≤N*
*c* ), an MLP can implementΞexactly.

*Layer 4 (arithmetic in dimension* 19*).*Layer 4 uses one DeltaNet head of dimension 19 and interprets the first row of *St*as the arithmetic row vector[*x|s|t*], where*x,s∈*Q¹ *×*9 and*t∈*Q.

**Initialization (first token).**Write [vec(*I₃*)*|*0*|*0] into the first row of *S₁* using one additive write: set *β₁* = 1, *v₁* =*e₁*, and*k₁* = [vec(*I₃*)*|*0*|*0] *⊤*.

**Purely multiplicative evolution (all later tokens).**For all *t≥*2, set *vt*=0 so *St*=*St−*1*H*(*βt,kt*). Layer 3 provides (*βt,kt*) so that *H*(*βt,kt*) =*Gb−*1*,τ*, i.e. the *τ*-th factor for the previous superblock’s product Π*b−*1. During the first superblock, takeΠ₀ =*I*and apply identity steps.

### LetT:= 9Nbe the final position. Write

### T= (B−1)mtok+τ, mtok:= 702,1≤τ≤mtok.

(So *τ*= 9*r*, where *r*=*N−*(*B−*1)*L* is the number of matrices in the final superblock.) By construction, at time *T* the arithmetic layer has applied the first *τ* factors of the previous full superblock *GB−*1*,*1*,...,GB−*1*,τ*(or identity factors if *B*= 1), but it has*not*appliedΠ*B*. We therefore decodevec(*PN*) =*x₀*Π₁ *···*Π*B*using a completion readout at time*T*.

**Final readout via completion vectors.**Let*pT∈*Q¹ *×*19 denote the arithmetic layer’s first-row state at time*T*. For each output coordinate*j∈{*1*,...,*9*}*define the completion vector

  *m* Ytok ! Π *Bej*

(*j*)  19*×*1
ˆ*ωT*:= *GB−*1*,s*  0  *∈*Q*,* *s*=*τ*+10

(*j*)
where the product is interpreted as identity when *B*= 1 or *τ*=*m*tok. This ˆ*ωT*is a fixed function of the finite router key (*T*mod 1404*, wT,wT−*1*,...,wT−*1403)*,*so it can be returned by the layer-3 lookup.

### The network outputs the nine scalars

(*j*)
*pT*ˆ*ωT*= vec(*PN*) *j* *, j*= 1*,...,*9*,*

i.e. the9entries of*PN* in row-major order, at the final position*T*= 9*N*.
## C. Upper Bounds on PD LRNNs

**Theorem 7.***LetMbe a multi-layer PD LRNN over*Q*. Then the language recognized byMis in*FO*-uniform*NC¹*.*

*Proof.* Similar to the proof of Theorem3, specifically the convolutional form of LRNNs in Equation (1), this proof reduces to showing that the product of a sequence *P₁D₁,...,PnDn*of PD matrices can be computed in FO-uniform Q NC¹.Terzic et al. *n*˜ ˜ (2025) observe PD matrices are closed under multiplication, so this product also has PD form. Let*i*=1*PiDi*= *PnDn*. We will provide a closed form for *P*˜*i*and *D*˜*i*, and then argue that it can be easily computed using FO-uniform NC¹ circuits.

Recall that *Di*is a diagonal matrix. Each *Pi*is a column one-hot matrix in *{*0*,*1*}* *d×d* and thus represents a function
*πi*:*{*1*,*2*,...,d}→{*1*,*2*,...,d}*. When *Pi*is full rank, *πi*represents a permutation of *d* elements. In general, *πi*may map
two inputs to the same output, i.e., collapse two elements into one. We will thus refer to it as a*relaxed permutation*.

We first observe that for any *d×d* matrix *A*, both *APi*and *ADi*have a very simple form: *APi*is the matrix obtained by applying the relaxed permutation *πi*to the columns of *A*, and *ADi*is the matrix obtained by scaling the *j*-th column of *A* by the *j*-th diagonal entry of *Di*. It follows that *AP₁D₁,...,PnDn*is nothing but an alternating sequence of relaxed permutations and scaling of columns of *A*. Intuitively, as long as we adjust the scaling terms appropriately, we can switch the order, i.e., first apply all relaxed permutations to the columns of *A* and then apply appropriate scaling; this, as we will shortly see, will make the matrix product easy to compute using uniform Q Q Q NC¹ circuits. Formalizing this, we will indeed ˜*n*˜*n j*+1 show that *Pn*=*i*=1*Pi*and *Dn*=*j*=1˜*πn,j*(*Dj*), where ˜*πn,j*=*i*=*nπi*is the composition of the last *n−j* relaxed permutations (by convention,˜*πn,n*is the identity permutation) and*π*(*Dj*)denotes the relaxed permutation of the diagonal entries of*Dj*according to*π*, leading to another diagonal matrix.

Conveniently, it is well-known that the product of *n* fixed size matrices that take only finitely many values (in particular fixed size 0-1 matrices) can be computed in FO-uniform NC¹ (Immerman & Landau,1992). Thus *P*˜*n*can be computed using such circuits. The relaxed permutation ˜*πj*can also be similarly computed, and applied to *Dj*. Lastly, *D*˜*n*, being a product of *n*diagonal matrices, consists of*n*products of*n*rationals each, which again can be computed inFO-uniformNC¹. ˜ Q*n* ˜ Q*n* All that remains to show is that *Pn*=*i*=1*Pi*and *Dn*=*j*=1˜*πn,j*(*Dj*). We prove this by induction on *n*. The base Q*n−*1 ˜ ˜ ˜ Q*n−*1 case of *n*= 1 holds trivially. For *n≥*2, assume by induction that*i*=1*PiDi*= *Pn−*1*Dn−*1where *Pn−*1=*i*=1*Pi*and *D* ˜ *n−*1= Q*j*=1˜*πn−*1*,j*(*Dj*). Then Q*i*=1*PiDi*= *P* ˜ *n−*1*D* ˜ *n−*1*PnDn*. *n−*1 *n*

We will use a “swap” property of P and D matrices, as well as the column one-hot nature of the P matrices, to show that *D* ˜ *n−*1*PnDn*=*PnD* ˜ *n*, which will finish the proof.

For this last part, observe that left-multiplying *Pn*by the diagonal matrix *D*˜*n−*1amounts to scaling the *j*-th row of *Pn*by the *j*-th diagonal entry of *D*˜*n−*1. Since *Pn*is column one-hot and thus has at most one non-zero entry in each column, this operation is the same as right-multiplying *Pn*with a relax-permuted version of the diagonal matrix, namely *πn*(*D*˜*n−*1).

That is,

*D* ˜ *n−*1*Pn*=*Pnπn*(*D* ˜ *n−*1)  

||n −1||
|---|---|---|
|n n|j=1 n−1,j||
|n −1||n −1|
|n j=1|n n−1,j|n j=1 n,j|

Y =*P π*  ˜*π* (*Dj*)

Y Y =*P π* ˜*π* (*Dj*) =*P* ˜*π* (*Dj*)

Right-multiplying both sides by*Dn*, we obtain:   *n* Y *−*1Y*n* *D* ˜ *n−*1*PnDn*=*Pn*  ˜*π* *n,j*(*Dj*)*Dn*=*Pn*˜*πn,j*(*Dj*) =*PnD* ˜ *n* *j*=1 *j*=1

### as desired, finishing the proof.

## D. Single-Layer LRNNs

### D.1. Inexpressibility of Deterministic Graph Connectivity

Our main results (Section3) show that, assuming PNC¹ *̸*=L, multilayer linear RNNs cannot solve the sorted deterministic graph connectivity problem. The same result holds*unconditionally*in the single layer case, since a single-layer linear RNN is a WFA and no WFA can represent the sorted deterministic graph connectivity problem (cf. Section3): **Theorem 13.***No weighted finite automaton can recognize sorted deterministic graph connectivity.*

*Proof.* Assume by way of contradiction that we can represent sorted deterministic graph connectivity. Then the Hankel matrix for this language has finite rank (Carlyle & Paz,1971). But we can easily construct an unbounded-rank subblock of the Hankel matrix where the rows are *s* and columns are *t*, and there are no edges in the graph (this yields the identity). Thus, we have reached a contradiction and this problem must not be representable.

A single-layer linear RNN can be simulated by a weighted finite automaton (Merrill et al.,2020, Theorem 4). Therefore, no linear RNN with just one layer can solve sorted deterministic graph connectivity, in the following sense: **Corollary 8.***The hidden state of a single linear RNN layer cannot solve sorted deterministic graph connectivity.*

The connection between LRNNs and WFAs shown here motivates considering a connection with an older line of work on rational recurrences (Peng et al.,2018). The central idea here, before the emergence of transformer, was that constraining RNNs so that their state was computable by a WFA was desirable because it made them more computationally efficient. Our results here reveal why: one-layer LRNNs are a kind of rational recurrence, and all rational recurrences can be computed in PNC¹, i.e., in a highly parallel fashion. While stacking multiple LRNN layers may not remain a WFA (Merrill et al.,2020; Mohri,2026), it will remain inPNC¹ by Theorem3.

### D.2. Single-Layer PD LRNNs and Deterministic WFAs

We say that a WFA (Definition13) is*deterministic*if the initial state vector *α* and each column over every transition matrix *Aσ*have at most one non-zero entry. **Theorem 8.***Any language recognized by a deterministic WFA with a zero threshold over* Q *can also be recognized by a* *one-layer PD LRNN over*Q*.*

*Proof.* We can represent an arbitrary deterministic WFA transition matrix in PD form as follows. Let *d* be the number of states. We set *bt*to be a one-hot encoding of the initial state at $ and the zero vector elsewhere. For each *σ* and state *q*, we set *Pσ*to be 1 at the unique *q* *′* to which *q* transitions. We set the entry in *Dσ*corresponding to *q* to have the weight of this transition. Thus, a single-layer PD LRNN is equivalent to a deterministic WFA.

## E. Arithmetic Circuits overQ

We define arithmetic circuits over the rational field *⟨*Q; +*,−,×,*0*,*1*⟩*, which we often write simply as Q. We represent a rational number *a/b* as a pair (*a,b*) of integers. An arithmetic circuit *C* over Q with *n* input nodes gives rise to a function *f*:*{*0*,*1*}* *n* *→*Z*×*Z. That is, the circuit will output a representation *a/b* of a rational number, but it might not be normalized. We show below that no problems arise in this way.

**Proposition 4.***Any arithmetic circuit family C*=*{Cn}n≥*1*over* Q *with logarithmic depth gives rise to a* GapNC¹ *function.* *Therefore, it can be expressed as a boolean bounded-fan-in circuit family withO*(log*n*log *∗*

*n*)*depth.*
*Proof.*To prove this, simply note that overQ: *a c ad*+*bc* + = *b d bc* and *a c ac* *×* =*.* *b d bd* Similarly, *−* <u>a</u> *b* = <u>−</u> *b* <u>a</u>. Therefore, each Q-operation can be simulated with at most four Z-operations. Therefore, if *C* has log depth, then so are theZ-circuits simulating*C*.

So, as a corollary,*C*can be expressed as a boolean bounded-fan-in circuit family with*O*(log*n*log *∗*

*n*)depth.
## F. Additional Details for Experiments

**Benchmark.**We evaluate models on a suite of synthetic algorithmic tasks designed to test length generalization.

**Deterministic graph reachability.**We construct a synthetic reachability dataset over directed graphs with *n* vertices. Vertices are assigned to two buckets under label-dependent constraints: for positive samples, vertices 0 and *n* lie in the same bucket, while for negative samples, they lie in different buckets. All remaining vertices are assigned independently with probability*p*. Directed edges are then added deterministically by connecting consecutive vertices within each bucket, yielding two disjoint directed paths. Each instance encodes a reachability query from vertex 0 to vertex *n* as a binary-serialized sequence.

**Iterated matrix multiplication with modulus.**This task requires computing prefix products of a sequence of 3*×*3 matrices over the finite ring Z*m*, where *m* is a fixed prime. Each example has the form SOS*T|m|qk|M₁ |···|MT* EOS, where *T* denotes the sequence length, *qk*is a fixed query index, and each matrix *Mt∈{−*1*,*0*,*1*}* 3*×*3 is sampled independently and rejected until invertible modulo *m*. Let *Pt*=*M₁ ···Mt*mod*m* denote the prefix product at step *t*. The target sequence is *v₁ |···|vT*, where *vt*equals the *qk*-th entry of *Pt*modulo *m*. Models are trained with a stepwise classification loss to predict*vt*at each step. The minimum sequence length for this task is1.

**Iterated matrix multiplication without modulus.**In this task, models multiply a sequence of 3*×*3 integer matrices over Z and predict a single binary label. Each example has the form SOS*T|M₁ |···|MT* EOS, where *T* is the number of matrices and each *Mt∈{−*1*,*0*,*1*}* 3*×*3 is sampled i.i.d. with probabilities (0*.*45*,*0*.*10*,*0*.*45) for entries (*−*1*,*0*,*1). Let *PT* =*M₁M₂ ···MT* denote the product over Z; the target label is defined as *y*= 1 iff (*PT*)0*,*0= 0, and *y*= 0 otherwise. We optionally construct balanced splits via rejection sampling to mitigate length-dependent artifacts. Since integer products may overflow for large *T*, labeling can be performed with an optional clipping cap that truncates intermediate values during multiplication. The minimum sequence length of this task is1.

Across both deterministic graph connectivity and iterated matrix multiplication, we use a largely standardized training setup to enable fair comparison across architectures. As shown in Table1, all models are trained with AdamW, a fixed batch size of 128, and gradient clipping at 1.0. Learning rates and weight decay are kept consistent within each task, with minor adjustments for architectures that require stronger regularization, such as DeltaNet. Training is capped at 60k steps for deterministic graph connectivity and 60k steps for iterated matrix multiplication, ensuring comparable optimization budgets while isolating architectural inductive biases as the primary source of performance differences.

|Task|Model|Configuration||
|---|---|---|---|
||RNNd|=256,L=2,d|= 256,3|
||Transformerd|=256,L=2,h= 4||
|DGC|Mambad|=256,L=2,d|=128|
||RWKV-7d|=256,L=2,h= 4||
||DeltaNetd|=256,L=2,h= 4||
||RNNd|=256,L=2,d|= 256,3|
||Transformerd|=256,L=2,h= 4||
|IMM|Mambad|=256,L=2,d|=64|
||RWKV-7d|=256,L=2,h= 4||
||DeltaNetd|=256,L=2,h= 4||

|LR|Steps|||
|---|---|---|---|
|×10|60k|||
|3×10|60k|||
|3×10|60k|||
|3×10|60k|||
|3×10|60k|||
|×10|60k|||
|3×10|60k|||
|3×10|60k|||
|3×10|60k|||
|3×10|60k|d, L, d|, h,|

*−*4 *m h* *−*4 *m* *−*4 *m* state *−*4 *m* *−*4 *m* *−*4 *m h* *−*4 *m* *−*4 *m* state *−*4 *m* *−*4 *m*

*Table 1.*Training hyperparameters for deterministic graph connectivity (DGC) and iterated matrix multiplication (IMM). *m h*

*d*state, and *LR* denote the model dimension, number of layers, hidden size, number of attention heads, SSM state expansion factor, and learning rate, respectively. All models are trained with AdamW, batch size 128, and gradient clipping at 1.0.
