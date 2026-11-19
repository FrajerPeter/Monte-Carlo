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

- Skontroluj alebo uprav sledované tickery, potom nastav parametre a spusti simuláciu v bočnom paneli.
- Vyhľadaj akciu podľa názvu alebo tickeru v katalógu 2 339 veľkých a mega-cap titulov, prípadne zadaj ticker ručne. Formáty `Visa (V)`, `Visa` aj `V` sa rozpoznajú; do jednej simulácie môžeš pridať najviac desať tickerov.
- Katalóg je lokálna snímka z 18. 11. 2026. Obsahuje aktívne primárne záznamy klasifikované ako Mega Cap alebo Large Cap v [FinanceDatabase](https://github.com/JerBouma/FinanceDatabase); nejde o presne zoradený živý rebríček prvých 1 000 podľa kapitalizácie. Podrobnosti sú v `data/README.md`.
- Vyber benchmark, horizont, menu účtu a počiatočnú investíciu.
- Vyber 1 000, 5 000 alebo 10 000 simulácií a jeden z piatich modelov.
- Nastav bezrizikovú sadzbu, zmenšenie historického drifta, neistotu odhadu drifta, rizikový limit, obmedzenia váh a odhad nákladov. Náklady môžu zahŕňať províziu, používateľský odhad spreadu/sklzu, priebežný poplatok a ročný obrat ako nákladový proxy pre rebalansovanie. Voliteľná zjednodušená daň sa odpočíta len z kladného konečného zisku.
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
- Walk-forward porovnanie každého modelu: 3 roky tréningových dát, 63 obchodných dní prognózy a 21-dňový posun. Zobrazuje pokrytie 10.–90. percentilu, presnosť smeru, chybu mediánu a šírku pásma. Tréning končí pred testovacím oknom; model sa porovnáva aj s nulovou prognózou.
- Citlivosť konečných výnosov na drift, volatilitu a dodatočné náklady, aj portfóliovej volatility pri rôznych korelačných predpokladoch. Tieto tabuľky sú aproximácie, nie nové plné Monte Carlo behy.
- Hypotetický jednorazový pokles, aj historický najhorší 63-dňový spoločný blok; v historickom teste sa po bloku simuluje zvyšok zvoleného horizontu.
- Porovnanie rovnomerne váženého portfólia pred a po pridaní vybranej akcie na tých istých simulovaných cestách.
- Prehľad zdroja, čerstvosti a pokrytia jednotlivých tickerov; pri starých dátach alebo nízkom pokrytí sa zobrazí upozornenie.
- CSV súbory pre akcie, váhy, portfóliá a spätný test, súhrnný PDF report a JSON nastavení so seed-om a SHA-256 odtlačkom dát.

## Obmedzenia

Yahoo Finance je dátovým zdrojom aplikácie, nie garantovaným produkčným feedom. Dostupnosť a kvalita údajov sa môže meniť. Päť rokov dát neposkytuje dostatok pozorovaní pre každý trhový režim. Ročný obrat je iba nákladový proxy; aplikácia neprehráva skutočné rebalansovacie obchody po jednotlivých cestách. Daň je plochá používateľská aproximácia kladného konečného zisku, nie výpočet daňovej povinnosti. Spread a sklz sú používateľský odhad a likvidita sa neoveruje živými objemami. Pri 63-dňovom horizonte historického stresového testu tvorí pozorovaný stresový blok celý horizont. Backtestové okná sa prekrývajú a katalóg dnešných aktívnych firiem neodstraňuje survivorship bias. Heuristický verdikt ani simulácie nie sú investičným odporúčaním a negarantujú budúce výnosy.

## Testy

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Testy pomocných funkcií používajú syntetické dáta alebo lokálny katalóg a nesťahujú Yahoo Finance. Zahŕňajú vyhľadanie Visa podľa názvu/tickeru aj ručné zadanie `Visa (V)`.

## Codex agenti

V `.codex/agents/` sú projektové read-only profily pre kvantitatívnu analýzu, dátovú vedu, UX kontrolu, code review a stručné odovzdanie kontextu. Sú odvodené z [VoltAgent/awesome-codex-subagents](https://github.com/VoltAgent/awesome-codex-subagents) a použiteľné pri explicitnom delegovaní v Codexe. Profily nepúšťajú automaticky ďalších agentov. Licencia MIT a pôvod sú zaznamenané v `.codex/agents/README.md`.

## Spustenie pôvodného skriptu

Pôvodný dávkový nástroj ostáva zachovaný:

```powershell
.\.venv\Scripts\python.exe monte_carlo_akcie.py
```

Výstupy skriptu ukladá do `mc_vysledky/`.
