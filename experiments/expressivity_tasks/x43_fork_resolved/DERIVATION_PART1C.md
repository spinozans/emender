# Direct suffix-array derivation (part 1c)

Let T[0:n] be the unchanged, finite byte corpus. Only positions 0 <= i < n
have a next byte. For 0 <= l <= i, C(i,l)=T[i-l:i], and
O(C)={j: l <= j < n, T[j-l:j]=C}. Overlaps count; no cyclic wrap,
end-of-text target, separator, or padding is introduced. Define
L_i=max{l: |O(C(i,l))| >= 2}; the empty context is admitted.
Set B_i={T[j]:j in O(C(i,L_i))}. FORCED means |B_i|=1;
FORK means |B_i|>1. The target occurrence itself is included: equivalently
all *other* matching positions must have the same next byte as T[i].
A singleton corpus has no repeated context and requires a separate convention;
this implementation rejects it.

A string C is right-maximal (in this next-byte convention) exactly when
it has at least two distinct byte right extensions. Therefore i is FORK iff
its longest repeated left suffix C(i,L_i) is right-maximal. This is a
per-occurrence statistic, not the number of distinct right-maximal strings.
A suffixient set covers the right extensions of right-maximal strings by
selected occurrence endpoints. Formally, for every right-maximal C and every
byte a for which some j in O(C) has T[j]=a, a suffixient endpoint set S must contain an
endpoint of some occurrence of Ca (equivalently Ca is a suffix of the corpus
prefix ending at that endpoint). With our target indexing those endpoints
are {j in O(C): T[j]=a}. Such coverage motivates keeping fork extensions,
but does not imply that this particular per-position fork count is the minimum
suffixient-set cardinality chi. We report chi_position/n explicitly; equality
to an xsa chi or minimal suffixient cardinality is NOT asserted. Boundary
sentinels in other definitions can change right-maximality.

For each target i associate reverse context R[n-i:] where R=reverse(T).
Use exactly these n suffixes (including the empty suffix, excluding R[0:]
which corresponds to the nonexistent target n). Their lexicographic SA and
adjacent LCP give L_i=max(LCP with predecessor, LCP with successor): any
nonadjacent shared prefix also occurs throughout its intervening SA interval.
Choose an adjacent edge attaining this maximum. Its maximal contiguous LCP
interval with all edges >= L_i is exactly O(C(i,L_i)). Thus the labels on
that interval, T[i], determine B_i and the classification above. For L_i=0
the interval is the whole SA. This proves both maximality and completeness.

Radix doubling sorts integer rank pairs with two stable numpy counting-order
passes (stable argsort on bounded integer ranks); LCP uses Kasai's amortized
linear algorithm, and nearest strictly smaller LCP edges give intervals.
No language model or held-out likelihood enters classification. Corpus-wide
labels use held-out text deliberately and are descriptive, transductive oracle
labels, not a causal predictor. Training-count baselines use only j<split,
and only contexts within the same training prefix; zero observed target counts
have infinite unsmoothed NLL. Coverage and zero counts are reported explicitly.
Model contexts reset every 128 bytes, whereas these contexts can be longer.
