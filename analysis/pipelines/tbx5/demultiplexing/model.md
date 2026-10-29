# the generative model

cells `i = 1..N`, bins `b = 1..B`, embryos `e = 1..K`. `a_ib` is the G1 reads a cell has in
a bin, `g_ib` the G2 reads, `n_ib = a_ib + g_ib` the coverage. coverage is conditioned on,
never modelled.

each embryo has a genotype `θ_e ∈ [0,1]^B`, its expected G1 fraction per bin.

a droplet is one of `K + K(K-1)/2` components:

- singlet from embryo `e`, genotype `p_e = θ_e`
- doublet of embryos `e ≠ e'`, genotype `p_ee' = (θ_e + θ_e') / 2`

there is no ambient component: a doublet is a 50/50 mix of two specific embryos, and no
component is a mix of the whole pool.

component `c` has weight `π_c`, and given `c` the reads in a bin are beta-binomial with a
per-bin concentration `s_b`:

    a_ib | c ~ BetaBinom(n_ib, p_cb · s_b, (1 - p_cb) · s_b)

so `E[a_ib / n_ib] = p_cb` and the variance is the binomial one inflated by
`(n_ib + s_b) / (1 + s_b)`. reads are not deduplicated, so `s_b` absorbs the pcr
duplication in bin `b`; `s_b → ∞` is the binomial.

bins are independent given the component, and the likelihood of a cell is

    P(a_i) = Σ_c π_c Π_b BetaBinom(a_ib | n_ib, p_cb, s_b)

## fitting

EM over `π` and `θ`, with `s` held fixed within a pass.

- E step: `r_ic ∝ π_c Π_b BetaBinom(a_ib | n_ib, p_cb, s_b)`.
- M step: `π_c = mean_i r_ic`, and `θ_eb = (Σ_i r_ie a_ib + 1) / (Σ_i r_ie n_ib + 2)`, a
  Beta(1,1) posterior mean. only the singlet responsibilities enter `θ`; the doublet
  genotypes follow from it.

three passes, each warm starting the next from its hard embryo assignment:

1. `coarse`: bins summed back to one column per chromosome, k-means on the z-scored G1
   fractions to initialise. at full resolution the log-likelihood gaps between components
   run to hundreds of nats, the responsibilities saturate, EM behaves like hard k-means and
   a component that starts badly collapses rather than migrating.
2. `binomial`: full resolution, `s = ∞`.
3. `betabinomial`: full resolution with `s_b` estimated by moments from the cells the
   binomial pass is sure are singlets. this is the fit the calls come from.

`log C(n_ib, a_ib)` is dropped from every reported log-likelihood. it does not depend on the
component so the E step never needs it, and it is the same for every model on the same
data.

## the call

a cell's embryo is `argmax_e r_ie`, and its score is the posterior of not being a singlet:

    score_i = Σ_{c a doublet} r_ic

cells with `score ≤ 0.05` keep their embryo; everything else is thrown away.
