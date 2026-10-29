# manual trajectory classes

each row is one $(\text{gene}, \text{source})$ pair: gene $g$'s log expression smoothed against EC fate bias $b\in[0,1]$ among nodes whose progenitor was that source, on a grid $b_1..b_{60}$. three features of the raw trajectory $x$ drive a rule-based label (no clustering).

features:
- slope $\ \beta = \dfrac{\sum_i (b_i-\bar b)\,x_i}{\sum_i (b_i-\bar b)^2}$, the expression change per unit fate bias.
- level $\ \ell = \operatorname{mean}_i x_i$.
- range $\ \rho = \max_i x_i - \min_i x_i$; peak at $b^\ast = \arg\max_i x_i$.

thresholds (relative to the mean absolute slope $\bar\beta = \operatorname{mean}_g|\beta_g|$): mild $= 0.5\,\bar\beta$, strong $= 1.5\,\bar\beta$; stable-high level $\ell_0 = 0.5$; transient window $(0.5, 0.9)$, endpoint clearance $0.1\rho$, movement floor $\rho \ge 0.5$.

precedence: transient is tested first (a shape call), then the slope bands, then the remaining flat genes split high/low.

## classes

- **transient**: interior bump: $b^\ast \in (0.5, 0.9)$, $\rho \ge 0.5$, and the peak clears *both* endpoints by $\ge 0.1\rho$. rises then falls along one path (e.g. *Etv2*).
- **rising strongly**: $\beta \ge 1.5\,\bar\beta$. turns on hard toward commitment (shared EC program: *Lmo2, Fli1*).
- **rising mildly**: $0.5\,\bar\beta \le \beta < 1.5\,\bar\beta$.
- **falling strongly**: $\beta \le -1.5\,\bar\beta$. turned off toward commitment (source memory erased: *Pax3, Meox1*).
- **falling mildly**: $-1.5\,\bar\beta < \beta \le -0.5\,\bar\beta$.
- **high, stable**: flat ($|\beta| < 0.5\,\bar\beta$) and $\ell \ge 0.5$.
- **low, stable**: flat and $\ell < 0.5$ (the near-silent majority).

## similarity

for a source pair, the fraction of genes assigned the **same non-flat class** (any class except the two stable ones) in *both* sources: of the genes actively moving in both tissues, how often they move the same way.

## row-5 groups

- **shared EC identity**: rising strongly in *every* covered source, TFs first.
- **path-specific memory erasure**: falling strongly in *one* source, stable (high/low) in the rest.
- **path-specific dynamics**: transient in some sources, rising in others.

rule + feature construction: `fatebias_manual_trajectory_classes.py` (`classify`, `load`). labels/palette/groups: `style.py`. See [concept](../concept.md).
