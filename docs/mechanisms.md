# Why changing CART's split selection can improve accuracy

Changing CART's split rule can improve population prediction error under specific
conditions. The arguments below identify those conditions and distinguish them
from guarantees for the implemented algorithms. The [study](../README.md) tests
their empirical implications against tuned CART and close prior methods.

## Gini and accuracy can prefer different splits

At a binary-classification node, let a split's children have masses `w_j` and
positive-class probabilities `p_j`. With population majority labels, its stump
error and weighted child Gini are

$$
R(s)=\sum_j w_j\min(p_j,1-p_j),\qquad
G(s)=2\sum_j w_jp_j(1-p_j).
$$

CART minimizes `G`; accuracy maximization minimizes `R`. Their orderings can
differ even without sampling error. Consider this joint population, where the
entries are probability masses expressed as percentages:

| A | B | Y = 1 | Y = 0 |
|---|---|---:|---:|
| 0 | 0 | 14 | 0 |
| 0 | 1 | 6 | 0 |
| 1 | 0 | 22 | 14 |
| 1 | 1 | 8 | 36 |

Splitting on A gives child masses `(0.2, 0.8)` and class probabilities
`(1, 0.375)`: `G(A) = 0.375` and `R(A) = 0.30`. Splitting on B gives
`(0.5, 0.5)` and `(0.72, 0.28)`: `G(B) = 0.4032` and `R(B) = 0.28`.
Population Gini prefers A; population stump accuracy prefers B by two percentage
points. This is an exact counterexample to equivalence of the objectives.

It does not show that Gini is generally a bad criterion. If both features can
subsequently be split, the two roots can lead to the same final partition.
Nor does it establish that our proposal generator will offer B, or explain the
digits result. It identifies a possible source of advantage under a constrained
tree budget. The calculation uses exact probability masses, so no simulation is needed
to establish the ordering.

## Why many noise features matter more for a weak signal

In the noise experiment, `X1` is a balanced bit and the label is flipped with
probability `eta`. Its population Gini reduction is

$$
\Delta G_{\mathrm{signal}}=\tfrac12(1-2\eta)^2.
$$

This is 0.32 when `eta = 0.1` and 0.02 when `eta = 0.4`. A noise feature is
independent of the label, so its population gain is zero. Its empirical gain
need not be zero.

For a fixed noise-feature threshold with nonempty children, let `w = n_L/n`.
The empirical Gini reduction is exactly

$$
\widehat{\Delta G}=2w(1-w)(\widehat p_L-\widehat p_R)^2.
$$

Conditional on the noise-feature observations, the labels are independent
balanced Bernoulli variables. The weighted Hoeffding bound for the difference
of the two means gives

$$
\Pr(|\widehat p_L-\widehat p_R|\geq u\mid X_{\mathrm{noise}})
\leq2\exp\{-2nw(1-w)u^2\}.
$$

Substituting the expression for the Gini gain and taking a union bound over
`M` noise splits yields

$$
\Pr\left(\max_s\widehat{\Delta G}_s\geq g\right)
\leq 2M\exp(-ng).
$$

The splits may share observations and labels; the union bound does not require
independence between them. For `p - 1` numeric noise features, there are at most
`M = (p - 1)(n - 1)` observed midpoint splits. Thus a sufficient threshold for
controlling their maximum gain at level `delta` is `log(2M/delta)/n`.
This gives the sample-size and search-complexity scales used in the experiment.
It does not prove that CART chooses the signal: its empirical gain also varies.
Nor does it prove that holdout or frequency scoring removes that variation.
The bound concerns the root of this specified population; conditioning on
adaptively chosen descendant nodes requires a separate argument.

## When honest validation selects a better stump

Fix a node independently of its evaluation observations. Conditional on training
data, suppose `M` candidate classifiers, including their leaf labels, are fixed.
Evaluate them on the same `m` fresh iid observations with zero-one loss. Write
`R_s` for population error, `Rhat_s` for evaluation error, and choose
`s_hat = argmin Rhat_s`. A union bound and Hoeffding's inequality give

$$
\Pr\left\{\max_s|\widehat R_s-R_s|>\epsilon\right\}
\leq 2M e^{-2m\epsilon^2}.
$$

On the complementary event,

$$
R_{\hat s}\leq\widehat R_{\hat s}+\epsilon
\leq\widehat R_{s^*}+\epsilon
\leq R_{s^*}+2\epsilon,
\quad
\epsilon=\sqrt{\frac{\log(2M/\delta)}{2m}},
$$

where `s*` is the lowest-risk candidate. Consequently, if a CART candidate is
worse than `s*` by more than `2 epsilon`, validation selects a better classifier
with probability at least `1 - delta`. The guarantee depends on evaluation size,
candidate count, and the risk gap. It can be uninformative in small nodes.
The concentration inequality is standard; the displayed selection bound follows
by the three inequalities above. [Hoeffding (1963)](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)

This theorem covers a fixed-candidate, fresh-holdout procedure. Our current
implementations require additional analysis:

- `holdout_selected` scores a split only when proposed, with different training
  sets, leaf labels, and evaluation sets. It also skips resamples whose held-out
  labels contain only one class. Its scores are not estimates from one common
  independent holdout; repeated use of observations does not create fresh data.
- `holdout_common` separates proposal labels from evaluation labels, but estimates
  leaf labels in overlapping folds and filters splits using all node covariates.
  The displayed fixed-classifier bound does not directly apply to that procedure.
- Below the root, the ancestors were themselves learned from these observations.
  Independence inside an externally fixed node cannot simply be assumed there.
- A lower stump error does not guarantee a lower error after growing subtrees.
  The relevant quantity for a depth budget `d` is the best achievable descendant
  error after a proposed root split, not only immediate majority-leaf error.

The objective experiment holds two proposals and their fitted leaf labels fixed,
then compares training Gini, training error, and error on an independent half of
the data. It varies sample size and the population mixture. The full-tree study
also includes a direct in-sample zero-one-loss splitter to separate the change
of objective from validation.

## What frequency voting can and cannot guarantee

Condition on a fixed training dataset `D` and a finite universe of `M` possible
splits. Let `q_s(D)` be the probability that an independently randomized pool
tree contains split `s` at least once. Its observed support is

$$
\widehat q_s=B^{-1}\sum_{b=1}^{B} I\{s\in T_b\}.
$$

For maximum-support selection without a cutoff on a fixed admissible candidate
set, suppose one split has conditional support
at least `gamma > 0` greater than every competitor. For each competitor, the
per-tree difference of the two indicators lies in `[-1,1]`. Hoeffding's inequality
and a union bound therefore imply

$$
\Pr\{\text{support voting fails to select that split}\mid D\}
\leq (M-1)e^{-B\gamma^2/2}.
$$

This is a guarantee about estimating the pool's preferred split. Predictive
improvement additionally requires that the preferred split is useful for the
target node and remaining depth. Increasing `B` reduces Monte Carlo error
conditional on `D`; it does not supply additional independent training data or
correct a systematically poor preference. The fixed-node argument is not a
whole-tree guarantee for adaptively selected descendants. A positive support
cutoff additionally requires the preferred split to survive that cutoff.

The current builder uses **global** support: a split earns a vote wherever it
appears in a pool tree. A good split deep in one context can be poor in another.
Exact numeric thresholds can also divide votes among almost equivalent splits.
This makes node-wise voting and median thresholds useful comparators.

The noise experiment deliberately contrasts a binary signal with continuous
irrelevant features. The signal has one possible midpoint; votes for a noise
variable can disperse across many midpoints. Exact-split support can therefore
favor the signal even when the pool often selects some noise variable. This is
a property of candidate multiplicity as well as predictive usefulness. A gain
in this experiment cannot be generalized to continuous signals with fragmented
threshold support. ALOOF directly addresses the related variable-selection
bias and is an essential comparator here.

Relative normalization has a precise, limited role. If
`q_max = max_s qhat_s > 0`, then

$$
\widehat q_s/q_{\max}\geq\tau
\quad\Longleftrightarrow\quad
\widehat q_s\geq\tau q_{\max}.
$$

It preserves every ranking and turns the absolute cutoff into a data-dependent
one. Given the same pool and `tau = 0`, the absolute and relative frequency
builders are identical. At other cutoffs, their differences arise from candidate
eligibility and consequent stopping. Thus the relative-frequency win could
reflect a better range of effective cutoffs and resulting tree sizes. Compare
matched effective cutoffs and leaf budgets before attributing an advantage to a
different ranking rule. The two cutoffs admit exactly the same candidates at every node.

The impurity-weighted score also has a simple interpretation. For positive
support and `eta > 0`, maximizing `qhat_s exp(-eta L_s)` is equivalent to minimizing

$$
L_s-\eta^{-1}\log\widehat q_s.
$$

It trades local child impurity against a penalty for low empirical support.
This algebra does not make the score a posterior probability or give it an
online multiplicative-weights regret guarantee.

## XOR distinguishes immediate scoring from useful descendants

Let `X1` and `X2` be independent balanced bits and `Y = X1 XOR X2`. Then

$$
\Pr(Y=1\mid X_1)=\Pr(Y=1\mid X_2)=1/2.
$$

Every root split on either feature has error `1/2` and zero Gini improvement.
Independent noise features have the same population properties. Yet splitting
on X1 and then X2 produces a depth-two tree with zero error. The same argument
holds for the independent Gaussian sign variables used in our recorded XOR
scenario: any threshold on a single feature leaves the label balanced.

Immediate population holdout accuracy cannot favor the interacting features.
Global pool frequencies might expose informative splits learned below other
splits, but that explanation requires evidence. The continuous generator
puts informative features first, and the deterministic custom splitter resolves equal
scores by feature index. The study compares the original column order with
three random permutations and includes random tie resolution. The binary study
also uses three permutations per dataset. The balanced-bit calculation and depth-two representation hold exactly in
the specified population.

There is also an exact tie counterexample for the current builder. Give both
signal features and one independent balanced noise bit equal split support. With
a depth-two cap, lexicographic selection gives 100% accuracy when the signal
features occupy columns 0 and 1, but 50% when the noise feature occupies column 0.
This comparison holds the population and supplied support fixed under relabeling. It
shows that order can matter; it does not establish how often such consequential
ties occur in the recorded pools.

For this mechanism, a two-level lookahead splitter is a direct benchmark.
Sliding two-level search is established work; our exhaustive binary reference
uses the same next-root objective as error-based lookahead, rather than the
optimized counting machinery of [Kiossou et al. (2024)](https://pschaus.github.io/assets/publi/ida2024_efficient_lookehead_decision_trees.pdf).
Globally optimized small trees provide a further comparison when computationally
feasible. Methods such as MurTree optimize a full tree rather than a sequence of
greedy local choices; optimization of training error alone still does not promise
better test error. [Demirović et al. (2022)](https://www.jmlr.org/papers/v23/20-520.html)

### An exact model where global support exposes the interaction

Consider `p >= 3` independent balanced bits, with `Y = X1 XOR X2`. Grow population
Gini trees to depth two, allowing zero-gain splits. At a tie, choose uniformly
among nonconstant features, independently at different nodes and across trees.
Every feature has one admissible threshold, 0.5. This idealized pool has no
sampling noise; its only randomness is tie resolution.

If the root is a signal feature, both child splits select the other signal
feature because it perfectly predicts the label there. If the root is noise,
each child still has pure XOR, so it chooses uniformly among the remaining
`p - 1` features. Write

$$
a_p=1-\left(1-\frac{1}{p-1}\right)^2.
$$

This is the chance that a specified remaining feature appears in at least one
of the two child splits. Counting a split at most once per tree gives

$$
q_{\mathrm{signal}}=\frac{2}{p}+\frac{p-2}{p}a_p,\qquad
q_{\mathrm{noise}}=\frac{1}{p}+\frac{p-3}{p}a_p.
$$

For the first expression, either signal at the root guarantees the specified
signal appears somewhere. For the second, the specified noise feature appears
at its own root, cannot appear below a signal root, and can appear below one of
the other `p - 3` noise roots. Thus

$$
\gamma=q_{\mathrm{signal}}-q_{\mathrm{noise}}
=\frac{1+a_p}{p}>0.
$$

Both signals outrank every noise feature in population support. A frequency
builder with a depth-two cap and a cutoff admitting the signals therefore
chooses a signal at the root and the other signal in each child. It predicts
perfectly. Uniform root ties alone select a signal only with probability `2/p`.

### Expected prediction error with a finite pool

A population CART tree selects a signal at the root with probability `2/p`.
That root permits both children to split on the other signal, giving zero error.
After a noise root, each depth-two leaf has conditioned on at most one signal
bit, so its label remains balanced. Hence

$$
\mathbb E[R(T_{\rm CART})]=\frac12\left(1-\frac2p\right).
$$

Let `T_B` be a depth-two tree reconstructed by maximum global support from `B`
independent population proposal trees, with no support cutoff and population
majority leaf labels. If both signal supports strictly exceed all noise supports,
`T_B` predicts perfectly. For each signal-noise pair, the per-tree difference
of inclusion indicators is in `[-1,1]` and has mean `gamma`. Hoeffding's
inequality gives a probability at most `exp(-B gamma^2 / 2)` that the empirical
difference is nonpositive. Taking a union bound over the `2(p - 2)` pairs gives

$$
\Pr\{\text{a noise support reaches a signal support}\}
\leq \min\{1,2(p-2)e^{-B\gamma^2/2}\}.
$$

Population majority labels have error at most one half on any partition.
The tree has zero error outside the event above, so

$$
\mathbb E[R(T_B)]\leq
\min\left\{\frac12,(p-2)e^{-B\gamma^2/2}\right\}.
$$

The upper bound is smaller than CART's exact expected error whenever

$$
B>\frac{2\log(2p)}{\gamma^2}.
$$

This proves an expected prediction-error advantage for a finite pool in the
stated population model. Since `gamma` is of order `1/p`, the sufficient pool
size is of order `p^2 log(p)`. The bound is conservative: it requires both
signals to outrank all noise variables, although some imperfect rankings still
produce useful trees.

The model removes training-sample error to isolate information contributed by
tree structure. It requires depth-two population Gini proposals, zero-gain
splitting, independent uniform ties, and population leaf labels. It does not
cover empirical bootstrap trees, which share a finite training sample and have
unequal empirical gains. Continuous predictors also distribute support across
multiple thresholds. A theorem for the implementation would need to control
those departures, including the probability that its conditional support favors
useful splits. Increasing the number of proposal trees alone cannot repair a
poor ordering caused by the training sample.

### Simulation of the population model

`cart_study/population.py` samples proposal structures directly from the model.
It does not generate training or test observations. If the root is signal, both
children select the other signal; otherwise the children independently choose
among the remaining features. Each feature receives at most one vote per tree.

Given the resulting support counts, let `u_j` be the probability of selecting
signal `j` at the reconstructed root under uniform ties. Let `v_k` be the
probability of selecting the other signal `k` in a child, excluding the root
variable. A signal root followed by the other signal classifies that child's
half of the population perfectly; other leaves have error one half. Thus the
exact accuracy, averaged over reconstruction ties, is

$$
\frac12+\frac12(u_1v_2+u_2v_1).
$$

The experiment averages this quantity over 2,000 independent proposal pools for
each feature count and pool size. Monte Carlo standard errors describe variation
across pools. Pool sizes share nested draws within a feature count, so points
along a curve are correlated. Greedy CART's population accuracy is `1/2 + 1/p`;
two-level lookahead has accuracy one.

At `p = 20`, accuracy is 54.0% with 20 proposal trees, 62.7% with 100, and 90.4%
with 500; CART's value is 55.0%. Small pools can perform worse than CART even
under the favorable population assumptions. The analytic result concerns a
sufficiently large pool, not monotonic improvement at every pool size.

## The quantity a whole tree must improve

For a fixed fitted tree, omit zero-mass nodes and define
`r(t) = 1 - max_c P(Y=c | X in t)`, the
population error of the best constant label in node `t`. For an internal node,
let `Delta(t)` be its reduction in this error after splitting, using population
majority labels in its two children. If `p(t)` is the population mass of the
node, then

$$
R(T)=r(\mathrm{root})-\sum_{t\text{ internal}}p(t)\Delta(t)
  +\sum_{l\text{ leaf}}p(l)
   \left[\max_c\Pr(Y=c\mid l)-\Pr(Y=\widehat c_l\mid l)\right].
$$

The last term is the cost of assigning leaf labels from a finite sample rather
than knowing each leaf's population majority. To prove the identity, expand
`p(t) Delta(t)` as the parent's weighted majority error minus its children's.
Every non-root internal-node term cancels with its occurrence as a child. The
remaining terms are the root error minus the population-majority leaf errors.
Adding each leaf's label-estimation cost gives the fitted tree's actual risk.

This identity holds for the realized partition; it does not assert independence
of adaptively selected nodes. It gives an exact condition for one tree to beat
another: its additional total population split gain must exceed its additional
leaf-label cost. Stump scoring addresses only the next gain. In pure XOR the
first gain is zero and the second-level gains total one half. Pool support could
help identify such useful descendants, but support itself appears nowhere in
the risk identity. Its connection to those gains must be established.

A change in leaf budget can improve the sum without improving split selection.
That is why the study reports accuracy against attained leaf count and includes
a CART comparator with a wider tuning grid.

## Benchmarks chosen to answer specific questions

| Comparator | What it distinguishes |
|---|---|
| CART with validation-selected depth, leaf size, and pruning, on both common and wider grids | A split-selection improvement versus a favorable fixed size cap or stopping rule |
| Zero-one-loss greedy splitter and fixed-proposal honest holdout scoring | Accuracy versus Gini objectives, separately from evaluation overfitting |
| ALOOF | Cross-validated variable selection versus our scoring of exact variable-threshold pairs |
| Dannegger node-wise resampling | Local variable votes and median cutpoints versus global exact-split support |
| Exact two-level lookahead on binary predictors | Whether the advantage requires seeing an interaction beyond the next split |

These comparators are implemented. The ALOOF implementation is a binary, numeric
reference based on direct leave-one-out squared-probability loss, checked against
exhaustive refits. It does not reproduce the paper's specialized fast algorithm.
Node voting uses 100 bootstrap stumps, variable votes, and median cutpoints,
with the study's shared leaf constraints. Lookahead searches all binary root
splits and the best immediate child splits, preferring lower immediate error
when two-step errors tie. It is fitted separately at each depth cap.

Consolidated Tree Construction and globally optimal trees remain related work,
rather than additional implemented baselines. The size curves show attained
leaf counts; they are not exact matches on leaf budget.

ALOOF reports predictive improvements using leave-one-out variable selection
followed by an ordinary CART threshold search. It is close prior work, not an
implementation equivalent. [Painsky and Rosset (2015/2017)](https://arxiv.org/abs/1512.03444)
Node-wise resampling has also produced accuracy gains while preserving a single
tree, with failures explicitly acknowledged. [Dannegger (1997 working paper; published 2000), §5.2](https://epub.ub.uni-muenchen.de/1466/1/paper_72.pdf)
Consolidated Tree Construction coordinates split choices across subsamples and
was originally compared against C4.5; adapting it to CART should be identified as
an adaptation. [Pérez et al. (2004)](https://www.scitepress.org/PublishedPapers/2004/26022/)

## How the simulation design follows from the arguments

Each comparison varies a quantity that appears in the argument: the risk gap,
the number of competing splits, the support gap, or the remaining tree depth.
The grid includes settings where the predicted advantage should disappear.

| Mechanism | Controlled variation | Prediction or possible falsification |
|---|---|---|
| Gini and accuracy disagree | Child masses and class probabilities, including cases where both objectives agree | Advantage from the objective change requires a risk-order reversal and access to the better candidate |
| Honest scoring avoids selection error | Evaluation size `m`, candidate count `M`, risk gap; fixed proposals | Reliable choice requires sufficient data relative to candidate complexity and the gap; splitting data can hurt when proposals or leaf labels become poorly estimated |
| Frequencies recover useful splits | Pool size `B` on a fixed dataset; then independent datasets | Increasing `B` stabilizes support estimates; any accuracy benefit needs useful population preferences and need not grow with `B` |
| Relative cutoffs change stopping | Matched effective thresholds and leaf budgets | Matching effective thresholds must give identical fitted frequency trees from the same pool |
| Global support exposes interactions | Balanced XOR with and without marginal signal; column permutations; lookahead comparator | Immediate population scoring cannot distinguish pure-XOR root features; a robust interaction advantage must survive changes in informative-feature positions |
| Exact thresholds fragment support | Continuous versus discrete predictors; local median-threshold voting | Failure of exact support should track fragmentation; coarse or median thresholds can also introduce bias |

The counterexample and bounds establish possible mechanisms under stated
conditions. The simulation grid tests them through objective disagreement,
weak signal among irrelevant variables, and interactions without marginal
signal. Independent simulation replicates and parameter perturbations measure sampling
and tuning sensitivity within these populations. The XOR theorem establishes a
whole-tree advantage under its population assumptions; the bootstrap results
measure behavior when split scores and leaf labels must be estimated.

## Isolating threshold identity from predictive information

Exact split frequency depends on how a partition is written. Two thresholds can
send every possible observation to the same child yet receive separate votes.
The controlled experiment in `cart_study/ablation.py` intervenes on that identity
while holding the proposal trees' predictions fixed.

Let $Z_j$ be independent balanced bits and

$$
X_j^{(\delta)}=Z_j+\delta U_j,\qquad
U_j\sim\operatorname{Uniform}[-1,1],\qquad 0\leq\delta<\tfrac12.
$$

The $U_j$ are independent of the bits, nuisance covariates, and labels. The label
is $Y=f(Z)\mathbin{\mathrm{XOR}}E$, with independent
$E\sim\operatorname{Bernoulli}(\eta)$ and $\eta\leq1/2$. Recovering each bit by
$\mathbf1\{X_j^{(\delta)}>1/2\}$ recovers $f(Z)$ exactly. Both the binary and
jittered representations therefore have Bayes error $\eta$. Jitter changes
CART's available thresholds without changing the information needed to predict
$Y$. It is applied to every constructed bit, including irrelevant bits.

**Partition invariance.** For every threshold
$t\in[\delta,1-\delta)$,

$$
\mathbf1\{X_j^{(\delta)}\leq t\}=1-Z_j.
$$

Every threshold in that interval induces the same population partition. Moving
any eligible split within the interval preserves a proposal tree's routing and
predictions, by induction down its nodes. The claim concerns a known gap in the
population support. An empty interval in a finite sample would not justify it.

Let $G$ be one such equivalence class of feature-threshold pairs. With $B$
proposal trees, compare

$$
q_G^{\mathrm{exact}}=
\max_{s\in G}\frac1B\sum_{b=1}^B\mathbf1\{s\in T_b\},\qquad
q_G^{\mathrm{grouped}}=
\frac1B\sum_{b=1}^B\mathbf1\{T_b\cap G\ne\varnothing\}.
$$

A tree contributes at most once to either count. Exact support cannot exceed
grouped support. Replacing all thresholds in $G$ with one common threshold makes
the counts equal. Giving each tree its own threshold within the same gap instead
makes exact support $1/B$ whenever $G$ appears. Neither operation changes a
proposal's prediction. Splits outside the known gaps retain their original
thresholds and scores. The intervention treats signal and nuisance bits alike.

Reconstruction uses the same candidate partitions, numerical representatives,
leaf constraints, training labels, and tie priorities in every score arm.
Candidates are deduplicated into these groups in both arms; this controlled
comparator differs from the original algorithm's separate treatment of every
threshold. Grouped support is invariant to threshold reassignment, so its fitted
tree is identical under all three encodings. Exact support can select a different
tree. That difference identifies the effect of vote fragmentation in the frozen
pool. There is no theorem that pooling equivalent votes improves accuracy:
irrelevant partitions receive their pooled votes too.

### Paired interventions and uncertainty

Each experimental unit is an independent training sample, evaluation sample,
and set of proposal draws. All treatment cells for that unit share the actual
latent bits, label flips, nuisance rows, bootstrap indices, and tree seeds.
Semantic partition keys retain their tie priorities across representations.
The runner evaluates every arm on the same 5,000 underlying evaluation
observations in their corresponding representations.
There is no treatment assignment to infer from observational data: each unit's
algorithmic counterfactuals are computed directly. Independent repetitions
estimate their average difference under the specified data-generating process.

The primary contrasts are the accuracy benefit of pooling artificially
fragmented votes in each representation, the benefit of grouping the natural
votes in each representation, and the interaction

$$
\bigl(A_{\mathrm{grouped}}-A_{\mathrm{exact}}\bigr)_{\delta=1/4}
-
\bigl(A_{\mathrm{grouped}}-A_{\mathrm{exact}}\bigr)_{\delta=0}.
$$

The interaction includes representation-induced changes in candidate availability
and proposal quality. Only the intervention on the frozen pool isolates threshold
identity. Realized leaf count is a consequence of the scoring rule; it is reported,
not used as a post-treatment matching variable. CART receives the same depth and
minimum-leaf constraints, but that does not impose equal attained tree sizes.
These CART comparisons provide context for the mechanism experiment. The broader
study supplies the tuned and alternative-algorithm comparisons.

For a fitted binary predictor $h$, evaluation integrates out the independent
label flip:

$$
R_\eta(h)=\eta+(1-2\eta)\Pr\{h(X)\ne f(Z)\}.
$$

The evaluation sample estimates the remaining covariate expectation. Paired
accuracy differences therefore measure ordinary predictive accuracy with less
Monte Carlo noise than newly sampled test labels. For a change from $h_0$ to
$h_1$, the gain is $(1-2\eta)$ times the fraction of latent errors repaired minus
the fraction introduced. Both fractions are saved. Observed test-label accuracy
is saved separately. At $\eta=1/2$, integrated accuracy is exactly one half for
every predictor; that identity checks the risk calculation, not absence of
training-label overfitting.

The grid contains nine contexts: a single signal bit with Gaussian, binary,
mixed, or correlated Gaussian noise; XOR or majority-of-three signal bits with
Gaussian noise; and a single signal bit with breast-cancer, wine, or digits
covariate vectors sampled independently as nuisance variables. The first six
contexts have 20 predictors. The last three retain all original covariates plus
the injected bit, and never use the original outcomes. Inference for those three
contexts conditions on their empirical covariate distributions. They are not
real-outcome prediction benchmarks.

Each context uses $(n,\eta)=(200,.4),(200,.1),(800,.4)$, with 200 replicates per
cell. Three additional cells use $\eta=.5$. Each pool contains 20 bootstrap CART
trees; depth-two and depth-five comparisons use the corresponding portions of
the same depth-five pool. Final trees have minimum leaf size ten, no support
cutoff, and no impurity weighting. These choices separate score ranking from
stopping and weighting. Settings and repetition counts are written to the run
configuration before fitting. Earlier weak-signal and XOR results informed the
design; it is not a preregistration.

Uncertainty uses replicate-level paired differences, not individual evaluation
rows or trees as independent observations. The output supplies Monte Carlo
standard errors, pointwise 95% paired t intervals, and Bonferroni intervals over
all 300 primary contrasts. The simultaneous intervals are approximate because
they use the t approximation. No overall average across contexts substitutes for
the individual effects. These intervals quantify training, evaluation, and
algorithm randomness within each specified population, not uncertainty about
transfer to other populations. Failed fits or checks stop the run; units are never
silently dropped. This use of explicit simulation estimands and paired random
inputs follows established [simulation-study design](https://doi.org/10.1002/sim.8086)
and [common-random-number methods](https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf).

### What must stay invariant

The runner checks proposal predictions after both threshold interventions. It
checks reconstruction structure and predictions under changes to candidate
iteration order, feature-column order with identities carried along, and scaling
features and thresholds by two. It also checks grouped support and reconstruction
under threshold reassignment, equality of score arms when no votes fragment,
and predictions after changing every unused feature of a fixed fitted tree.
The run records eligible and ineligible unused-feature checks separately.

Adding an irrelevant training feature is not an exact negative control: it gives
a finite-sample search another chance to fit noise. Refitting CART after a column
permutation can also change tied choices, because its random search uses column
positions. The experiment measures that refit sensitivity separately; it does
not demand equality. The exact column-order check concerns reconstruction from
a frozen pool with preserved feature identities and tie priorities.

The equivalence classes in this experiment use known population support. They
are an oracle diagnostic for the mechanism, not a method for finding safe groups
in arbitrary observed data. Learning those groups without merging genuinely
different partitions would require a separate argument and evaluation.

### What the paired experiment finds

Changing only threshold identity can materially change the accuracy of the final
tree. The table reports the cost of artificial fragmentation in binary proposal
pools, alongside the gain from grouping the natural votes in jittered pools.
All entries use 200 training observations, 40% label flips, depth two, and 200
independent replicates. Units are percentage points; intervals adjust for all
300 primary contrasts. Positive values favor grouped support.

| Context | Removing artificial fragmentation, binary | Grouping natural votes, jittered |
|---|---:|---:|
| Gaussian noise | 4.31 [3.20, 5.42] | 0.15 [-0.10, 0.39] |
| Binary noise | 4.68 [3.50, 5.87] | 0.18 [-0.14, 0.50] |
| Mixed noise | 5.53 [4.43, 6.63] | 0.07 [-0.18, 0.31] |
| Correlated Gaussian noise | 5.20 [4.05, 6.34] | 0.12 [-0.13, 0.37] |
| XOR + Gaussian noise | 0.00 [-0.02, 0.01] | 0.00 [0.00, 0.00] |
| Majority + Gaussian noise | 1.58 [1.02, 2.14] | 0.04 [-0.13, 0.21] |
| Breast-cancer covariates | 4.85 [3.75, 5.95] | 0.25 [-0.11, 0.61] |
| Wine covariates | 5.60 [4.53, 6.67] | 0.06 [-0.14, 0.26] |
| Digits covariates | 4.35 [3.23, 5.47] | 0.24 [-0.14, 0.61] |

The two columns answer different questions. The first isolates threshold identity
in a fixed pool. The second measures the effect of grouping votes produced by
CART after jitter has changed its search. In the Gaussian-noise context, the
useful partition appears in 84% of binary depth-two pools but only 23.5% of
jittered pools. Pooling votes cannot invent a missing candidate. These frequencies
describe a change in availability; they are not a decomposition of the total
representation effect.

Natural grouping has a positive simultaneous lower bound in 11 of the 54
non-null context/regime/depth cells. The remaining cells include small estimated
losses: mixed noise at $n=800$, $\eta=.4$, and depth five gives $-0.16$ points
[-0.46, 0.14]. Those inconclusive intervals do not establish benefit or equivalence.
The largest estimated natural-grouping gain occurs for jittered XOR with 10%
label flips and depth five: 12.51 points [8.31, 16.71], raising accuracy from
55.00% to 67.51% against 90% Bayes accuracy. Majority-of-three in the
same regime gains 3.41 points [2.39, 4.44]. The full grid, including small and
uncertain effects, is in [the contrast table](../results/ablation/contrasts.csv).
These are effects of grouping with known support gaps, not gains from a learned
grouping algorithm.

All 6,000 paired replicates completed. Checks included 480,000 proposal prediction
comparisons, 288,000 reconstruction structure comparisons, and 72,000 unused-feature
perturbations. Grouping and exact support agree in every binary cell, as the
absence of natural fragmentation requires. The 50%-flip cells have integrated
accuracy exactly 50%; their ordinary test-label accuracy means range from 49.87%
to 50.06%. Refitting CART after a column permutation changes a mean of up to
3.46% of predictions within a cell, while the largest absolute mean accuracy
change is 0.073 percentage points. These refit differences are sensitivity to
search and ties, distinct from the exact fixed-pool invariances.

## Learning groups and improving proposals

A training sample can define useful partition groups without revealing the
population support. On each feature, group thresholds that place every training
observation on the same side. Represent a group by the midpoint between the
adjacent observed values that bound it. Use each proposal tree's membership in
the group as one vote. This removes arbitrary threshold aliases from the score,
but an empty training interval need not be empty in the population.

### What empirical equivalence guarantees

Let $X_1,\ldots,X_n$ be independent, identically distributed observations with
$p$ features. For every
feature $j$, consider all intervals $I=(a,b]$ containing no training observations.
For $0<\epsilon<1$,

$$
\Pr\left\{\exists j,I:\ N_j(I)=0,
\ \Pr(X_j\in I)>\epsilon\right\}
\leq p(n+1)(1-\epsilon)^n
\leq p(n+1)e^{-n\epsilon}.
$$

To see this, represent one feature as $X_{ij}=F_j^{-1}(U_{ij})$ with independent
uniform draws across observations. The event $a<X_{ij}\leq b$ is equivalent,
almost surely, to $F_j(a)<U_{ij}\leq F_j(b)$. An empty observed interval therefore
maps into an empty interval in uniform probability space. The $n$ uniform points
form $n+1$ spacings, each with probability $(1-\epsilon)^n$ of exceeding
$\epsilon$. A union bound over spacings and features proves the result. Dependence
between features is allowed, as are discrete and mixed marginal distributions.
The argument uses the standard [uniform-spacing distribution](https://www.stat.purdue.edu/~dasgupta/orderstats.pdf).

Consequently, with probability at least $1-\alpha$, all thresholds grouped by
training-partition identity disagree on at most

$$
\epsilon_\alpha=
1-\left[\frac{\alpha}{p(n+1)}\right]^{1/n}
$$

of the population for their feature. This is simultaneous over the groups, so
choosing thresholds after examining labels does not invalidate it. For $n=200$,
$p=20$, and $\alpha=.05$, the bound is about 5.5% per split. Substituting thresholds
at $L$ nodes of a fixed tree changes its predictions on at most
$\min(1,L\epsilon_\alpha)$ population mass on this event. The same quantity bounds
the absolute change in classification risk. It can be a loose whole-tree bound.
Pooling scores can change which partitions the tree selects; the bound supplies
no accuracy guarantee for that additional change.

### A proposal rule with a stated geometric condition

Coarsening predictors before tree fitting is established practice in
[discretization](https://ai.stanford.edu/~ronnyk/disc.pdf). The experiment uses a
conservative rule that needs neither labels nor a fitted separation threshold:

1. Sort a feature and find its largest adjacent gap among cuts leaving at least
   ten training observations on each side.
2. Flag the feature only if that gap is wider than the observed range on either
   side of the cut.
3. Replace a flagged feature by an indicator at the gap midpoint when fitting
   coarsened proposals. Leave other features unchanged.

Suppose a feature's support is contained in two intervals of widths $w_0,w_1$,
separated by a gap $g>\max(w_0,w_1)$. If both components supply at least ten
observations, their separating observed gap exceeds every within-component gap
and both observed within-component ranges. The rule therefore detects that
separation. Its midpoint also lies in the population gap: even the worst sampled
endpoints cannot move it outside when $g$ exceeds both widths. With component
probability $\pi$, the probability of sufficient observations is
$\Pr\{10\leq\operatorname{Binomial}(n,\pi)\leq n-10\}$.

For the continuous uniform-jitter generator at $\delta=.25$, the population gap
and component widths are equal. Sampled endpoints lie strictly inside the
intervals almost surely, so sufficient component counts still give a larger
observed gap than either within-component range and a midpoint inside the true
gap.

There is a simple negative-control calculation. For a continuous uniform
feature, a flagged gap must exceed one third of the observed total range.
Conditional on the sample extrema, the $n-1$ normalized spacings have marginal
upper tail $(1-x)^{n-2}$. Thus

$$
\Pr(\text{flag a uniform feature})
\leq(n-1)(2/3)^{n-2}.
$$

This bound concerns a uniform distribution, not every distribution without a
support gap. The detector also depends on numerical distances: positive affine
recoding preserves its decision, while an arbitrary monotone transformation
need not. Outliers can prevent detection by expanding a within-side range.

A support gap alone says nothing about whether labels are constant inside its
clusters. The final learner therefore receives a mixed pool: ten ordinary CART
proposals and ten coarsened proposals, rather than twenty coarsened proposals.
The raw half preserves opportunities to split within clusters. This policy does
not guarantee that reconstruction will select those opportunities.

### Separating the two changes

The runner `cart_study/learned.py` crosses ordinary versus mixed proposals with
peak exact versus grouped support. All four arms use empirical partition groups
and their common midpoint representatives. Peak support retains the largest
exact-threshold count within a group; grouped support counts the union of trees
containing it. Each score comparison holds candidates and tie priorities fixed.
The proposal comparison replaces the latter ten proposals using the corresponding
bootstrap rows and tree seeds. The interaction measures whether this proposal
change increases or decreases the grouping benefit.

The gap map and equivalence relation use all training covariates and are shared
by the bootstrap proposals. They require neither labels, test covariates, nor
population metadata. Proposal thresholds and votes are learned using training
labels. The original literal-threshold recurrence rule is included separately;
its comparison with empirical grouping also changes representatives, candidate
multiplicity, and tie identities, so it is not a pure aggregation contrast.

Baselines are ordinary CART, CART with exactly the same learned coarsening,
three-fold-tuned CART, ALOOF, and node voting with twenty bootstrap stumps per
node. CART tuning searches minimum leaf sizes 5, 10, and 20 and pruning penalties
0, .01, and .03, with depths 2 and 5 subject to the comparison's depth cap. It
uses only training folds and refits the chosen configuration on all training
rows. All final models are single trees. Their attained leaf counts are outcomes,
not post-treatment matching variables. Gap-CART coarsens the final learner's
inputs completely; comparing it with a mixed proposal pool assesses two complete
policies, not just a difference in aggregation.

There are 34 settings with 200 independent replicates each. The nine preceding
contexts use $n=200$ and $n=800$ with 40% label flips. Gaussian-noise, XOR, and
majority contexts also use $n=200$ with 10% flips. The Gaussian context varies
jitter width through 0, .1, .25, .4, and .6. Additional controls use a continuous
threshold signal, a label determined by position *within* the separated clusters,
a 20%-prevalence signal bit, and 1% covariate contamination that shifts the signal
feature by ten. Those four controls use both 10% and 40% label flips. A final
Gaussian setting uses 50% flips.

The overlap case, $\delta=.6$, changes predictive information: the latent bit
cannot be decoded exactly. Its latent-bit Bayes error is $1/12$, giving Bayes
accuracy $7/12$ with 40% label flips for the underlying continuous generator.
The integrated-risk formula remains valid because it averages errors against
the latent truth. Widths and contexts use different condition-specific samples;
paired causal contrasts concern algorithm changes *within* each setting.

The design was fixed before the full run, informed by the earlier oracle
experiment. Evaluation uses 5,000 independent covariate draws per replicate and
integrates out label flips. The results report paired Monte Carlo intervals,
with simultaneous intervals over 680 primary contrasts. Invariance checks on
gap detection and reconstruction from frozen proposals cover training-row order,
feature-column order, doubling features and thresholds, threshold aliases, and
unused features. Canonicalization must preserve proposal
predictions on training rows; its test disagreement is measured rather than
required to vanish. If no feature is flagged, replacing proposals must have no
effect. Fits and checks must all complete; failures cannot remove units from the
comparison.

### What learning the groups changes

All 6,800 replicates completed, giving 200 paired observations for each contrast
in each setting and depth. The following effects are in percentage points; all
bracketed intervals are approximate 95% simultaneous Monte Carlo intervals over
the 680 primary contrasts. The first two columns change only vote aggregation
within their respective proposal pools. The third replaces half the proposals
while retaining grouped scoring.

| Setting | Grouping, raw proposals | Grouping, mixed proposals | Mixed minus raw, grouped scoring |
|---|---:|---:|---:|
| XOR, $n=200$, 10% flips, depth 5 | 7.99 [4.14, 11.85] | 5.01 [1.13, 8.89] | -3.03 [-6.79, 0.73] |
| Majority, $n=200$, 10% flips, depth 5 | 3.83 [2.35, 5.32] | -0.01 [-0.17, 0.15] | 1.87 [1.02, 2.72] |
| Binary noise, $n=200$, 40% flips, depth 2 | 0.06 [-0.74, 0.86] | 0.07 [-0.23, 0.37] | 1.95 [0.83, 3.07] |
| Binary noise, $n=800$, 40% flips, depth 5 | -0.37 [-0.95, 0.21] | -0.89 [-1.64, -0.14] | -1.52 [-2.53, -0.51] |
| Within-cluster signal, $n=200$, 40% flips, depth 5 | -0.04 [-0.33, 0.25] | -0.10 [-0.33, 0.13] | -0.53 [-0.84, -0.22] |

All rows use jitter width $.25$. Binary-noise rows jitter the 19 nuisance bits
as well as the signal. These examples illustrate distinct responses; the
[full contrast table](../results/learned/contrasts.csv) contains every cell.

For XOR, raw peak support gives 54.01% accuracy and raw grouped support gives
62.00%. This isolates a grouping benefit with groups learned from observations.
The mixed grouped procedure gives 58.97%, compared with 54.15% for tuned CART,
53.88% for ALOOF, and 50.97% for gap-CART. Its gain over tuned CART is 4.83 points
[0.52, 9.14]. The proposal intervention's estimated effect is negative, with an
interval crossing zero.

For majority-of-three, mixed grouped support reaches 89.95% accuracy against
90% Bayes accuracy, versus 84.90% for tuned CART, 84.98% for ALOOF, and 88.31%
for gap-CART. The gains over tuned and gap-CART are 5.05 [3.84, 6.26] and 1.64
[0.52, 2.77]. The grouping-by-proposals interaction is -3.85 [-5.34, -2.36]:
grouping helps raw proposals, but adds little once half the proposals are
coarsened. The changes are substitutes in this setting.

The weak binary-noise example shows a proposal benefit at depth two and $n=200$:
the combined procedure reaches 55.57% versus 53.53% for tuned CART, a gain of
2.04 [0.96, 3.12]. At $n=800$ and depth five, it reaches 54.76% versus 58.52%
for tuned CART, a loss of 3.76 [2.78, 4.74]. Both component interventions have
negative simultaneous upper bounds there. Mixed peak and mixed grouped trees
both have 32 leaves in every replicate, so their grouping contrast is not
explained by different tree sizes. Tuned CART averages 3.63 leaves; its comparison
also includes a complexity difference. A positive effect in one regime does not
justify applying either change throughout the grid.

Across the 66 non-null setting/depth cells, raw-pool grouping has a positive
simultaneous lower bound in seven and no negative upper bounds. Mixed-pool
grouping has two positive and one negative cells; mixed proposals with grouped
scoring have thirteen positive and three negative cells. The combined procedure
has eleven positive and nine negative cells against tuned CART. Remaining
intervals include zero. These counts summarize the chosen settings; they do not
estimate a success rate over an external population of datasets. Node voting
comparisons are secondary and have pointwise intervals only.

The controls delimit the geometric argument:

- In uncontaminated settings, every constructed two-component separation at
  widths 0, .1, and .25 was detected. At width .4 the support still has a gap,
  but the rule's stronger separation condition
  fails; no features were flagged. No features were flagged at width .6 or in
  either continuous-signal setting. Proposal effects were exactly zero when
  the gap map made no change.
- With 1% contamination, the signal was flagged in 13% and 12% of replicates
  at 10% and 40% label flips. The probability of no contaminated training row is
  $.99^{200}\approx13.4\%$. Expanded ranges largely disable this detector.
- With strong within-cluster signal and depth five, gap-CART gives 50.01%
  accuracy, while mixed grouped support gives 88.26% and tuned CART 88.68%.
  Raw proposals preserve useful information in this control. With weak
  within-cluster signal, mixed proposals reduce grouped accuracy by 0.53 points
  [0.22, 0.84], and the combined procedure loses 0.56 [0.17, 0.95] to tuned CART.
  Preserving candidates is insufficient to ensure they are selected.
- Every model in the 50%-flip setting has integrated accuracy exactly 50%.
- In the weak imbalanced setting, $\Pr(Z=1)=.2$ and $\eta=.4$ imply
  $\Pr(Y=0)=.56$. The fixed constant-zero rule therefore has population accuracy
  56%, above the depth-two combined procedure's 53.49% and tuned CART's 54.66%.
  This analytic reference uses the known population; it is distinct from a
  fitted majority classifier and exposes a shortcoming of both fitted methods
  under the stated tuning grid.

All prescribed checks passed, including 272,000 comparisons of proposal
predictions before and after threshold canonicalization on training rows and
235,800 tree-structure comparisons. The largest cell mean of the per-proposal
test disagreement after canonicalization was 0.3375%. Empirical equivalence
therefore behaved as an approximation outside training, as the proof allows.
The [invariance counts](../results/learned/invariances.json),
[gap detection rates](../results/learned/gap_detection.csv), and
[model accuracies and leaf counts](../results/learned/summary.csv) are saved
alongside every replicate and its input hash. These experiments identify useful
learned groups and conditional proposal gains; they do not yet provide a
training-only rule for deciding when to coarsen proposals.

## Selecting the learner from training data

A useful proposal rule need not help every population. The next experiment asks
whether validation can choose among CART, grouped raw proposals, and grouped
mixed proposals, preserving accuracy gains while avoiding their observed losses.
Each output is still one tree. This is ordinary cross-validation applied to these
learners, not a new model-selection principle.

### The selection rule

Five shuffled folds share the same candidate grid. Every gap map, partition
group, bootstrap pool, and leaf label is learned from that fold's training rows.
The held-out rows supply only classification errors. This follows the
[fold-local preprocessing requirement](https://scikit-learn.org/1.8/modules/cross_validation.html#data-transformation-with-held-out-data).
Fold assignments are independent of covariates and labels, allowing the
conditional argument below.

CART searches depths 1, 2, and 5, minimum leaf sizes 5, 10, and 20, and pruning
penalties 0, .01, .03, and .1. Each grouped learner searches the same depths and
leaf sizes, using 20 proposals. Mixed pools contain ten raw and ten coarsened
proposals. The gap detector's minimum component count equals the candidate's
minimum leaf size. A learned majority-class constant is also eligible for every
family. There are 55 distinct configurations, with comparisons restricting depth
to at most two or five.

The selector minimizes total held-out mistakes. Ties favor fewer leaves across
fold fits, then the constant, CART, raw, and mixed families, then shallower depth,
larger minimum leaves, and stronger pruning. A deterministic configuration index
breaks remaining ties. The selected configuration is refit on all training rows.
Family-specific baselines use the same folds, scores, and tie order; the selector
reuses the same full-data tree as its winning family. This makes their difference
an effect of expanding the selection menu, with no extra random refit.

### What validation can and cannot guarantee

For two fixed binary classifiers fitted independently of a validation sample,
let $D$ be one classifier's zero-one loss minus the other's, let
$\Delta=\mathbb E[D]$, and let $d$ be their probability of disagreeing. Then
$D^2$ is exactly the disagreement indicator, so

$$
\operatorname{Var}(D)=d-\Delta^2,\qquad
\operatorname{SE}(\overline D)=\sqrt{(d-\Delta^2)/m}
$$

for $m$ independent validation observations. Pairing removes variation on rows
where both classifiers agree. Small gains can still be difficult to distinguish:
a two-point accuracy difference with $d=.3$ has signal-to-noise ratio one only
around $m=749$. This calculation concerns fixed classifiers and ordinary noisy
labels. It is not a standard-error formula for a selected model, dependent
out-of-fold predictions, or integrated-noise evaluation.

A separate uniform bound describes cross-validation. Let $h_{kf}$ be candidate
$k$ fitted without fold $f$, let $R(h)$ denote population classification risk,
and let $\widehat R_{kf}$ be its validation error on that fold. For $K$ candidates
and $F$ equal folds of size $m$, conditional Hoeffding bounds followed by a union
bound give, with probability at least $1-\alpha$,

$$
\max_{k,f}|\widehat R_{kf}-R(h_{kf})|\leq
\epsilon=\sqrt{\frac{\log(2KF/\alpha)}{2m}}.
$$

Each validation fold is independent of its own fitted models; independence
between folds is unnecessary for the union bound. The fitting randomness must
also be independent of the held-out observations. Averaging over folds and
minimizing validation error gives

$$
\overline R_{\hat k}\leq\min_k\overline R_k+2\epsilon,
\qquad \overline R_k=F^{-1}\sum_f R(h_{kf}).
$$

This bounds the average risk of fold-trained trees, not the risk of the final
refit. If additionally
$\max_k|R(h_{k,\mathrm{full}})-\overline R_k|\leq\tau$, then

$$
R(h_{\hat k,\mathrm{full}})
\leq\min_k R(h_{k,\mathrm{full}})+2\epsilon+2\tau.
$$

The experiment assumes no such stability guarantee for CART. Even the first
bound is loose: using $K=55$, $F=5$, and $\alpha=.05$ gives $2\epsilon\approx.682$
at $n=200$ and $.341$ at $n=800$. The mathematical argument identifies selection
noise and refitting instability as obstacles; it does not predict that the
selector must improve accuracy. Untouched evaluation data test the final tree.

### The paired comparison

The design retains the preceding 34 populations and uses a new master seed,
20261011. The previous results informed the candidate grid, particularly its
constant and pruning options. Each setting has 200 independent training/test
replicates; evaluation uses 5,000 independent draws and integrates out label
flips. Data for evaluation are generated only after selection and refitting.
The smoke run checks implementation, without outcome-driven tuning of this grid.

At each depth cap, the five primary contrasts compare the selector with
family-tuned CART, raw grouping, mixed grouping, and the two previous fixed
policies (raw or mixed grouping at the depth cap with minimum leaf size ten).
There are 340 primary contrasts. Paired t Monte Carlo intervals use replicate
units and Bonferroni adjustment across this family. Folds and test rows are not
treated as independent simulation replicates. Failed fits or checks abort the
run; they cannot remove observations from the comparisons.

Before the full run, the criterion for preserving gains while avoiding material
harm was set to: at least one positive simultaneous selector-minus-CART contrast,
and a simultaneous lower bound above minus one percentage point in every non-null
setting/depth cell. The one-point margin is an explicit practical tolerance for
this grid, not a guarantee over other distributions. An interval spanning that
margin is unresolved, not evidence of safety. With 200 replicates, large
interaction gains may be distinguishable while small gains or losses remain
uncertain; precision is reported rather than assumed.

Diagnostics record selected families, attained leaf counts, gains and losses
from switching away from CART, and changes between mean fold-tree risk and
full-refit risk on the evaluation sample. The best observed test accuracy among
the three family winners is an infeasible, optimistic finite-test reference. It
is computed after selection and never guides it. These diagnostics explain
where performance is lost; they do not authorize choosing a learner using the
population name or evaluation labels.

Harmful-switch frequencies, positive and negative gain components, and rank
reversals are observed on a finite evaluation sample. Their nonlinear summaries
need not be unbiased for the corresponding population events. The primary mean
paired accuracy contrasts do not use these nonlinear transformations.

An adversarial check puts outliers only in one validation fold, removing a gap
from the full sample while leaving a gap in that fold's training subset. The
fold must learn its own gap. Changing that validation fold's covariates and
labels must leave all 55 fitted candidate trees for that fold unchanged. Other
checks cover disjoint and exhaustive folds, one validation prediction per row,
alignment between saved predictions and loss totals, candidate-order invariance,
shared family winners, constant predictions, and exact raw/mixed identity when
no gaps are detected. Replays with different evaluation sample sizes check that
selection and fitted trees do not depend on evaluation data.

### What the selector recovers

Training-only selection retains the two main interaction gains. With $n=200$,
10% label flips, jitter width $.25$, and depth capped at five, XOR accuracy is
64.47% for the selector and 55.22% for tuned CART. The paired gain is 9.25
percentage points [4.84, 13.67]. For majority-of-three, the corresponding
accuracies are 89.87% and 85.06%, a gain of 4.81 [3.69, 5.92]. These are
approximate 95% simultaneous Monte Carlo intervals over all 340 primary contrasts.

The choices differ across populations without using population metadata. For
strong XOR at depth five, the selector chooses raw grouping in 117 of 200
replicates, mixed grouping in 49, CART in 32, and the constant in two. For strong
majority it chooses mixed grouping in 188, raw grouping in ten, and CART in two.
The grouped learners are useful alternatives that validation often recognizes.
This result does not make cross-validation itself a new method.

The earlier binary-noise failure is also avoided. With $n=800$, 40% label flips,
and a depth cap of five, fixed mixed grouping gives 55.14%, tuned CART 58.84%,
and selection 59.46%. The gain over tuned CART is 0.62 [0.15, 1.08]. With
$n=200$ and a cap of two, selection gives 54.77% versus 53.43%, gaining 1.34
[0.39, 2.29]. The fixed rules and family-tuned baselines distinguish adding
learner choice from retaining a fixed tree size.

A wider menu still costs accuracy in some comparisons. At depth two in the
strong majority population, selection gives 69.66%, while choosing only within
the mixed family gives 70.02%: a difference of -0.37 [-0.60, -0.14]. At depth
five, those methods are close, 89.87% and 89.88%. Expanding the menu cannot
increase its minimum validation error, but it can select a worse final tree.

The strong XOR result also illustrates the limit of the harm criterion. At depth
two the selector-minus-CART difference is -0.55 [-2.02, 0.92]. This does not
establish a loss greater than one point, but it cannot rule one out. The declared
criterion therefore fails even though the depth-five gain is clear. In the
$n=200$ digits-nuisance setting, the depth-two difference is -0.32 [-1.04, 0.39],
another interval crossing the one-point harm margin. These are statements about
precision and the chosen tolerance, not proof of a universally harmful or safe
selector.

Weak class imbalance supplies the other two unresolved cells. At $n=200$ and
40% label flips, the selector-minus-CART differences are -0.45 [-1.04, 0.14]
at depth two and -0.42 [-1.05, 0.22] at depth five. The learned constant achieves
55.70%, while selection gives 55.29% and 54.94%. Including a useful fallback
does not ensure that validation chooses it.

The mean interaction gains also differ in their consistency across training
samples. Strong majority at depth five improves on CART in 193 of 200 runs,
ties in three, and loses in four; its median gain is 3.55 points. Strong XOR
improves in 110, ties in 35, and loses in 55; its median gain is 0.44 points and
its worst observed loss is 29.76 points. These are descriptive comparisons on
finite evaluation samples, not simultaneous guarantees about population win rates.

That worst XOR run illustrates refitting instability. In replicate 111,
validation favors the selected mixed configuration over the CART family winner
by nine points. Independent evaluation of the fold-trained models also favors
it, by 8.66 points on average. Yet the full-data refits give 53.71% for the
selected tree and 83.47% for CART. Refitting, including renewed algorithm
randomness, reverses a real fold-model advantage. This post hoc example shows
why the stability term in the bound matters; it does not isolate which aspect
of refitting caused the reversal.

Across all 66 non-null setting/depth cells, the simultaneous intervals have the
following signs. The remaining intervals include zero.

| Comparator | Positive selector difference | Negative selector difference |
|---|---:|---:|
| Tuned CART | 12 | 0 |
| Tuned raw grouping | 19 | 0 |
| Tuned mixed grouping | 3 | 8 |
| Fixed raw grouping | 27 | 1 |
| Fixed mixed grouping | 15 | 3 |

These counts describe the designed grid, not a distribution of prediction tasks.
The four unresolved one-point margins concern average accuracy in their cells;
even resolving them would not guarantee protection for each training sample.
All 6,800 replicates completed. Fold isolation, selection, shared-tree, and
invariance checks passed, as did serial replays and evaluation-size changes.
Under the null, integrating label noise gives exactly 50% accuracy for every
tree. The [full contrasts](../results/selection/contrasts.csv),
[selection frequencies](../results/selection/choices.csv), and
[diagnostics](../results/selection/diagnostics.csv) are generated from the saved
replicates.

### Why a zero-gain split can help a small tree

The within-cluster control admits a population calculation that helps interpret
its result. This explanation was derived after observing the control, rather
than used to predict its outcome. Let $X=Z+\delta U$, where $Z$ is a balanced bit,
$U$ is uniform on $[-1,1]$, $\delta=.25$, and the latent label is
$T=\mathbf 1\{U>0\}$. Observed labels flip independently with probability
$\eta<.5$; other predictors are independent nuisance variables.

In order along $X$, the four equal-mass regions have labels $0,1,0,1$.
A root split in the gap between clusters has zero population Gini gain, because
both children remain balanced. Splitting each child at its cluster center then
recovers $T$ exactly. This depth-two tree achieves accuracy $1-\eta$.

Population-greedy CART instead splits at a cluster center. To see why, write
$r=F_X(t)$ and $H(r)=\Pr(T=1,X\leq t)$. Across the four regions, $H(r)$ is
$0$, $r-1/4$, $1/4$, and $r-1/2$, respectively. The root Gini gain is

$$
G(r)=\frac{2(1-2\eta)^2[H(r)-r/2]^2}{r(1-r)}.
$$

It is maximized at $r=1/4$ or $3/4$, with gain $(1-2\eta)^2/6$.
After the next greedy split, half the population lies in pure latent-label
leaves and half in a balanced leaf. Its accuracy is therefore
$3/4-\eta/2$: 70% at $\eta=.1$, compared with the attainable 90%.
At the same noise level and depth cap, the experiment gives 69.39% for tuned
CART and 74.77% for selection, a gain of 5.38 points [2.95, 7.81]. A globally
reused split can create useful children despite having no immediate impurity
gain. The calculation establishes that opportunity; it does not prove which
splits caused the empirical gain.

At depth five, the same strong-signal comparison is close: 88.50% versus 88.57%.
With 40% flips, both depth caps give about 50.7%, and the selector-minus-CART
intervals include zero. The population opportunity alone does not ensure that
finite noisy samples reveal the useful tree.
