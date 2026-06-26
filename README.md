# The Regime

A weekly leveraged-Nasdaq regime tracker. One strategy, flipping between
**Aggressive** (100% LQQ, 2x net Nasdaq), **Moderate** (75% EQQQ + 25% money
market) and **Cash**, gated by price trend and financial conditions.

Live: https://soylee22.github.io/the-regime/

## Rule (checked weekly)
- **CASH** if the Chicago Fed NFCI is in the top 20% of its 5-year range (held 3 months once triggered)
- else **MODERATE** if the Nasdaq-100 is below a rising 200-day average
- else **AGGRESSIVE**

`build_site.py` fetches fresh data (Yahoo + FRED), computes the signal and metrics,
and renders `index.html` from `template.html`. A GitHub Action rebuilds it each weekday.

Personal research tool. Not investment advice.
