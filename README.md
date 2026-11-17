# Monte Carlo analýza akcií

Lokálna Streamlit aplikácia na scenárovú analýzu cien akcií a portfólií. Sťahuje upravené denné ceny z Yahoo Finance, pri zahraničných tituloch ich historicky prevádza do meny účtu a porovnáva päť modelov vývoja výnosov. Výstupom sú interaktívne grafy, rizikové metriky, portfóliové stratégie, stresové testy, walk-forward backtest a export do CSV/PDF.

## Spustenie vo Windows PowerShell

V koreňovom priečinku projektu:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Otvoriť `http://localhost:8501`. Zastaviť server pomocou `Ctrl+C` v termináli. Pri ďalšom spustení stačí posledný príkaz.

Ak sa Streamlit už inštaloval, vynechať vytvorenie prostredia aj `pip install`. Internet je potrebný na stiahnutie cien, mien a FX kurzov. Prvé spustenie môže trvať dlhšie, pretože aplikácia vyhodnocuje aj spätný test piatich modelov. Dáta sa ukladajú v Streamlit cache na 30 minút.

## Ovládanie aplikácie

- Pridaj jeden až desať tickerov a vyber benchmark, horizont, menu účtu a počiatočnú investíciu.
- Vyber 1 000, 5 000 alebo 10 000 simulácií a jeden z piatich modelov.
- Nastav bezrizikovú sadzbu, zmenšenie historického drifta, neistotu odhadu drifta, rizikový limit, obmedzenia váh a odhad nákladov.
- Stlač **Spustiť simuláciu**. Po zmene nastavení klikni znovu, aby sa prepočítali výsledky.
- Záložky ukazujú akcie a fan chart, štyri portfóliové stratégie, hypotetický a historický stres, porovnanie modelov spätným testom a export.

## Modely a odhady

Modely používajú spoločné výnosy aktív a regularizovanú korelačnú maticu:

1. **GBM** – konštantná volatilita a korelované normálne šoky.
2. **Blokový bootstrap** – náhodne vzorkuje spoločné 20-dňové bloky historických logaritmických výnosov.
3. **Student-t GBM** – GBM-i podobný model s 8 stupňami voľnosti pre ťažšie chvosty.
4. **GARCH(1,1) s Student-t šokmi** – podmienená volatilita reaguje na nedávne šoky; parametre sa vyberajú z malej obmedzenej mriežky.
5. **Prepínanie režimov** – pokojný/stresový režim sa odhaduje z 20-dňovej volatility a historickej prechodovej matice.

Denné historické logaritmické výnosy sa pri odhade priemeru orežú na 1. a 99. percentile. Odhad drifta sa zmenšuje k bezrizikovej sadzbe. Prepínač neistoty pridá každej simulovanej ceste vlastný náhodný posun drifta podľa odhadovanej štandardnej chyby; je to približná citlivostná metóda, nie úplný Bayesov model. Volatilita sa odhaduje z historických denných výnosov. Zahraničné ceny sa prevádzajú podľa denných Yahoo FX kurzov, budúce FX pohyby sú však neisté.

Fan chart ukazuje až 1 000 sivých ukážkových ciest, 10., 50. a 90. percentil a počiatočnú cenu. Percentily sa počítajú zo všetkých simulácií. Pásmo je scenárové, nie garantovaná cenová hranica.

## Metriky, portfólio a backtest

- Priemerný a mediánový výnos po zadaných transakčných a priebežných nákladoch.
- VaR 95 %: strata na 5. percentile výnosov; CVaR 95 %: priemerná strata v najhorších 5 % scenárov.
- Pravdepodobnosť zisku a párová pravdepodobnosť prekonať benchmark v spoločných simuláciách.
- Sharpe Ratio a pravdepodobnosť, že maximálny pokles počas celej cesty prekročí používateľský limit.
- Rovnomerné portfólio, približné maximum Sharpe, minimum volatility a minimum historického CVaR. Váhy sa hľadajú medzi 50 000 náhodnými prípustnými long-only kandidátmi; výsledok nie je garantované globálne optimum.
- Walk-forward porovnanie každého modelu: 3 roky tréningových dát, 63 obchodných dní prognózy a 21-dňový posun. Zobrazuje pokrytie 10.–90. percentilu, presnosť smeru, chybu mediánu a šírku pásma.
- Hypotetický jednorazový pokles, aj historický najhorší 63-dňový spoločný blok; v historickom teste sa po bloku simuluje zvyšok zvoleného horizontu.
- CSV súbory pre akcie, váhy, portfóliá a spätný test; súhrnný PDF report.

## Obmedzenia

Yahoo Finance je dátovým zdrojom aplikácie, nie garantovaným produkčným feedom. Dostupnosť a kvalita údajov sa môže meniť. Päť rokov dát neposkytuje dostatok pozorovaní pre každý trhový režim. Modely nepoznajú budúce správy, zmeny fundamentov, likviditu, dane, spread, sklz ani všetky poplatky brokera. Pri 63-dňovom horizonte historického stresového testu tvorí pozorovaný stresový blok celý horizont, a preto v tomto teste nie je náhodná pokračovacia časť. Backtestové okná sa prekrývajú. Heuristický verdikt ani simulácie nie sú investičným odporúčaním a negarantujú budúce výnosy.

## Testy

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Testy existujúcich dátových a portfóliových pomocných funkcií používajú syntetické dáta a nesťahujú Yahoo Finance. Nové simulačné modely boli navyše overené syntetickým smoke testom bez sieťového sťahovania dát.

## Spustenie pôvodného skriptu

Pôvodný dávkový nástroj ostáva zachovaný:

```powershell
.\.venv\Scripts\python.exe monte_carlo_akcie.py
```

Výstupy skriptu ukladá do `mc_vysledky/`.
