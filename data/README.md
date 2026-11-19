# Katalóg veľkých svetových akcií

`global_stocks_largecap.csv` is a local snapshot created on 2026-11-18 from the
[FinanceDatabase](https://github.com/JerBouma/FinanceDatabase) equities dataset.
The project is MIT-licensed and describes its equity data as a weekly updated
directory sourced from exchange and market listings.

The snapshot keeps active primary listings in the `Mega Cap` and `Large Cap`
categories. It contains 2,339 search candidates. FinanceDatabase supplies
capitalization categories rather than an exact numeric global ranking, so this
is a broad catalog of large companies, not a claim that the rows are the exact
top 1,000 in current market-cap order. The local CSV does not update itself.

The app uses Yahoo Finance symbols and still downloads prices live when a
simulation runs. Some exchange listings may not have a complete Yahoo Finance
history; the app will report that when the symbol is simulated.
