# Monte Carlo analýza akcií

Python skript na scenárovú analýzu akcií pomocou korelovaného geometrického Brownovho pohybu (GBM). Stiahne historické ceny z Yahoo Finance cez `yfinance`, odhadne drift, volatilitu a koreláciu, spustí simulácie a uloží grafy a CSV reporty.

## Požiadavky a spustenie

- Python 3.10 alebo novší
- Internetové pripojenie pri sťahovaní dát

```bash
python -m pip install -r requirements.txt
python monte_carlo_akcie.py
```

Predvolené nastavenie používa `AAPL`, `MSFT`, `NVDA`, `TSLA` a `SPY`, päť rokov denných dát, horizont 252 obchodných dní a 10 000 simulácií. Parametre vrátane tickera benchmarku, bezrizikovej sadzby, limitu poklesu a váhového limitu portfólia sú v hornej časti `monte_carlo_akcie.py`.

## Metóda a výstupy

Pre každé aktívum sa historický priemer denných jednoduchých výnosov anualizuje ako drift `mu = 252 × priemer` a smerodajná odchýlka ako volatilita `sigma = sqrt(252) × std`. Korelované normálne šoky tvoria spoločnú maticu výnosov pre všetky tickery. Denná GBM aktualizácia je:

```text
S[t+1] = S[t] * exp((mu - 0.5 * sigma²) / 252 + sigma * z / sqrt(252))
```

Výsledky sa zapisujú do `mc_vysledky/`:

- `fan_chart.png` – pásma 5.–95. a 25.–75. percentilu a medián cien
- `rozdelenie_konecnych_cien.png` – rozdelenie simulovaných konečných cien
- `metriky_akcii.csv` – výnosy, VaR/CVaR, pravdepodobnosti, Sharpe, drawdown a orientačný filter
- `portfolio.csv` – váhy a metriky najlepšieho z náhodne vygenerovaných prípustných long-only portfólií podľa historického Sharpeho pomeru

VaR a CVaR sú uložené ako znamienková strata: kladná hodnota znamená stratu, záporná hodnota zisk aj v dolnom chvoste. Pravdepodobnosť prekonať benchmark sa počíta párovým porovnaním výnosov z rovnakých korelovaných simulácií. Benchmark sa neporovnáva sám so sebou a jeho `screen_pass` je označený ako `BENCHMARK`.

## Obmedzenia

GBM predpokladá konštantnú volatilitu a normálne denné šoky. Nezachytáva zmeny režimov, zhlukovanie volatility, cenové skoky ani neistotu odhadu driftu. Portfóliové váhy sú aproximáciou z konečného počtu náhodných kandidátov, nie globálne garantovaným optimom. Prahy vo filtri sú konfigurovateľné heuristiky, nie investičné odporúčania. Výsledky simulácie nie sú zárukou budúcich výnosov.

## Testy

Po nainštalovaní závislostí spustite:

```bash
python -m unittest discover -s tests -v
```

Testy používajú syntetické dáta a nenavštevujú Yahoo Finance.
