# axis parameterization

two axes score a crossing motif from cell-type means. both are signed coherence scores: the same euclidean-distance margin between the true parent-child coupling and its nearest competitor, using the same norm and the same aggregation. positive means the true coupling is the tightest (the method recovers it) and negative means a competitor is tighter (the method fails).

## setup

- parents (progenitors) at $t_\text{src}$: $p_a, p_b$
- mid $a$'s children at $t_\text{tgt}$: anchor $a_1$, inner $a_2$
- mid $b$'s children at $t_\text{tgt}$: inner $b_1$, anchor $b_2$
- imposed lineage ($\alpha = 0$): $\{a_1, a_2\} \to p_a$, $\{b_1, b_2\} \to p_b$
- every symbol is the mean embedding over that (type, time) cell pool

shared matched-distance cost of a coupling (mean euclidean distance from each object to the parent it is matched to):

$$ D(\{u_i\}, \{v_i\}) = \frac{1}{n} \sum_{i=1}^{n} \lVert u_i - v_i \rVert $$

## ot coherence

objects: the four children individually. competitors: the other balanced couplings sending two children to each parent.

$$ D_\text{imposed} = D\big(\{a_1, a_2, b_1, b_2\},\ \{p_a, p_a, p_b, p_b\}\big) $$

let $\mathcal{B}$ be the balanced 2-to-2 couplings (two children per parent) *excluding* the imposed one. then

$$ s_\text{ot} = \min_{\pi \in \mathcal{B}} D_\pi \;-\; D_\text{imposed} $$

- $s_\text{ot} > 0$: the imposed coupling is cheaper than every competitor $\Rightarrow$ ot recovers it.
- $s_\text{ot} < 0$: a regrouping is cheaper $\Rightarrow$ ot mismatches (anti-ot).

## felsenstein coherence

objects: each mid's children pooled to their centroid. competitor: the swap that matches the two centroids to the wrong parents.

$$ r_a = \tfrac{1}{2}(a_1 + a_2), \qquad r_b = \tfrac{1}{2}(b_1 + b_2) $$

$$ s_\text{fels} = D\big(\{r_a, r_b\}, \{p_b, p_a\}\big) \;-\; D\big(\{r_a, r_b\}, \{p_a, p_b\}\big) $$

- $s_\text{fels} > 0$: each centroid is nearest its own parent $\Rightarrow$ pooling recovers the ancestry.
- $s_\text{fels} < 0$: a centroid is nearer the wrong parent $\Rightarrow$ pooling mismatches (anti-felsenstein).

## shared choices

- norm: euclidean $\lVert \cdot \rVert$ (not squared)
- aggregation: mean over matched pairs
- sign: (nearest competitor cost) - (true coupling cost); positive = coherent
- the two axes differ *only* in what the objects are, individual children ($s_\text{ot}$) vs pooled sibling centroids ($s_\text{fels}$), and hence which competitors exist. this is the ot vs felsenstein distinction.

## crossing strength $\alpha$

fixed at $0$ (an ordered inner-pair permutation already realises both couplings). general $\alpha$ crosses the inner pair with that probability:

$$ r_a = \tfrac{1}{2}\big(a_1 + (1-\alpha)\,a_2 + \alpha\,b_1\big), \qquad r_b = \tfrac{1}{2}\big(b_2 + (1-\alpha)\,b_1 + \alpha\,a_2\big) $$

and $D_\text{imposed}$ replaces each inner child's term by its $\alpha$-expected distance, e.g. $(1-\alpha)\lVert a_2 - p_a \rVert + \alpha \lVert a_2 - p_b \rVert$.
