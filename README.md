# Monte Carlo analýza akcií

[English version](README.en.md)

Interaktívna Streamlit webová aplikácia na scenárovú analýzu cien akcií a portfólií. Sťahuje upravené denné ceny z Yahoo Finance, pri zahraničných tituloch ich historicky prevádza do meny účtu a porovnáva päť modelov vývoja výnosov. Výstupom sú interaktívne grafy, rizikové metriky, portfóliové stratégie, stresové testy, walk-forward backtest a export do CSV/PDF.

**Live demo:** [Otvoriť aplikáciu](https://omgcotopeterspravil.streamlit.app/)

<!-- Screenshot placeholders: add the exported images at these paths. -->

![Fan chart simulovaných cenových scenárov](assets/fan_chart.png)

![Porovnanie výsledkov portfólia](assets/portfolio_comparison.png)

## Prehľad

- **Technológie:** Python, Streamlit, NumPy, pandas, yfinance, Plotly a ReportLab.
- **Simulácia:** päť modelov výnosov s koreláciou aktív, historickými dátami a menovým prepočtom.
- **Riziko a portfólio:** VaR/CVaR, Sharpe Ratio, drawdown, stresové scenáre a štyri váhovacie stratégie.
- **Validácia a export:** walk-forward porovnanie modelov, CSV výstupy a súhrnný PDF report.

## Spustenie vo Windows PowerShell

V koreňovom priečinku projektu:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Spustenie v Linuxe alebo macOS (bash/zsh)

V koreňovom priečinku projektu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

V oboch prípadoch aplikáciu otvor na `http://localhost:8501`. Server zastavíš pomocou `Ctrl+C` v termináli; pri ďalšom spustení stačí posledný príkaz. Ak je virtuálne prostredie a závislosti už pripravené, preskoč ich inštaláciu. Internet je potrebný na stiahnutie cien a FX kurzov. Prvé spustenie môže trvať dlhšie, pretože aplikácia vyhodnocuje aj spätný test piatich modelov. Ceny a FX kurzy sa ukladajú v Streamlit cache na 30 minút.

## Ovládanie aplikácie

- Zoznam sledovaných tickerov je po otvorení prázdny. Vyhľadaj a vyber akcie z katalógu alebo pridaj vlastný ticker, potom spusti simuláciu v bočnom paneli.
- Katalóg obsahuje **2 340** titulov. Vyhľadávanie podporuje názov alebo ticker a filtre podľa burzy/regiónu; možno zadať aj vlastný symbol. Formáty `Visa (V)`, `Visa` aj `V` sa rozpoznajú. V jednej simulácii možno sledovať najviac 10 tickerov.
- Katalóg je lokálna snímka z 18. 11. 2026 založená na [FinanceDatabase](https://github.com/JerBouma/FinanceDatabase), ku ktorej bol 19. 11. 2026 pridaný titul Anthropic (`ANT`) po jeho IPO na Nasdaqu. Základná snímka obsahuje aktívne primárne záznamy klasifikované ako Mega Cap alebo Large Cap. Kategórie zdroja nie sú presným číselným rebríčkom kapitalizácie, preto katalóg nie je tvrdením, že ide o presných 1 000 najväčších akcií podľa aktuálnej kapitalizácie. Podrobnosti sú v [`data/README.md`](data/README.md); katalóg sa sám neaktualizuje.
- Aplikácia používa symboly Yahoo Finance a pri simulácii sťahuje ceny aktuálne. Dostupnosť histórie sa medzi titulmi líši. Nový titul ANT má zatiaľ krátku históriu, preto môžu byť dlhšie horizonty nedostupné alebo obzvlášť neisté.
- Vyber benchmark (vrátane S&P 500, Nasdaq-100, NYSE Composite `^NYA` a ďalších indexov/ETF), horizont, menu účtu a počiatočnú investíciu. Minimálny vklad je 10 € alebo 10 $; vlastný benchmark sa nezadáva ručne.
- Vyber 1 000, 5 000 alebo 10 000 simulácií a jeden z piatich modelov.
- V pokročilých nastaveniach uprav bezrizikovú sadzbu, zmenšenie historického drifta, neistotu odhadu drifta, rizikový limit, obmedzenia váh a náklady. Náklady môžu zahŕňať províziu, používateľský odhad spreadu/sklzu, priebežný poplatok a ročný obrat ako nákladový proxy pre rebalansovanie. Voliteľná zjednodušená daň sa odpočíta iba z kladného konečného zisku.
- Stlač **Spustiť simuláciu**. Po zmene nastavení spusti simuláciu znova, aby sa výsledky prepočítali.
- Záložky zobrazujú vývoj akcií, štyri portfóliové stratégie, hypotetický a historický stres, porovnanie modelov spätným testom a export.

## Modely a odhady

Modely simulujú spoločné výnosy aktív a používajú regularizovanú korelačnú maticu:

1. **GBM (geometrický Brownov pohyb)** – konštantná volatilita a korelované normálne šoky.
2. **Blokový bootstrap** – náhodne vzorkuje spoločné 20-dňové bloky historických logaritmických výnosov, čím zachováva ich krátkodobú časovú závislosť.
3. **Student-t GBM** – GBM model s 8 stupňami voľnosti pre ťažšie chvosty.
4. **GARCH(1,1) s Student-t šokmi** – podmienená volatilita reaguje na nedávne šoky; parametre sa vyberajú z malej obmedzenej mriežky.
5. **Prepínanie režimov** – pokojný a stresový režim sa odhaduje z 20-dňovej volatility a historickej prechodovej matice.

Denné logaritmické výnosy sa pri odhade priemeru (driftu) orežú na 1. a 99. percentile, aby jednotlivé extrémy neovládli odhad očakávaného výnosu. Odhad drifta sa zmenšuje k bezrizikovej sadzbe. Volatilita a korelácia sa počítajú z pôvodných, neorezaných výnosov, aby sa zachovali historické chvosty; Student-t a GARCH-t modely takisto pracujú s neorezanými výnosmi. Prepínač neistoty pridá každej simulovanej ceste vlastný náhodný posun drifta podľa odhadovanej štandardnej chyby. Je to približná citlivostná metóda, nie úplný Bayesov model. Zahraničné ceny sa prevádzajú dennými historickými Yahoo FX kurzami; budúce FX pohyby zostávajú neisté.

Fan chart zobrazuje až 1 000 sivých ukážkových ciest, 10., 50. a 90. percentil a počiatočnú cenu. Percentily sa počítajú zo všetkých simulácií. Pásmo predstavuje scenáre, nie garantované cenové hranice.

## Metriky, portfólio a backtest

- Priemerný a mediánový výnos po zadaných transakčných a priebežných nákladoch.
- **VaR 95 %:** strata na 5. percentile konečných výnosov. **CVaR 95 %:** priemerná strata v najhorších 5 % scenárov.
- Pravdepodobnosť zisku a párová pravdepodobnosť prekonať benchmark v spoločných simuláciách.
- Sharpe Ratio a pravdepodobnosť, že maximálny pokles počas celej simulovanej cesty prekročí používateľský limit.
- Štyri portfóliové stratégie: rovnomerné váhy, približné maximum historického Sharpe, minimum historickej volatility a minimum historického CVaR. Váhy sa hľadajú medzi 50 000 náhodnými prípustnými long-only kandidátmi; nejde o garantované globálne optimum.
- Walk-forward porovnanie každého modelu používa tri roky tréningových dát (756 obchodných dní), 63 obchodných dní prognózy a 21-dňový posun. Pri každom odhade tréningové okno končí pred testovacím obdobím. Spätný test používa 2 000 simulácií na odhad a zobrazuje pokrytie 10.–90. percentilu, presnosť smeru, chybu mediánu a šírku pásma. Okná sa prekrývajú a porovnanie zahŕňa aj nulovú prognózu.
- Citlivosť konečných výnosov na drift, volatilitu a dodatočné náklady, ako aj citlivosť portfóliovej volatility na korelačné predpoklady. Tieto tabuľky sú aproximácie, nie nové plné Monte Carlo behy.
- Hypotetický jednorazový pokles a historický najhorší spoločný 63-dňový blok. Po historickom bloku model simuluje zvyšok zvoleného horizontu.
- Porovnanie rovnomerne váženého portfólia pred a po pridaní vybranej akcie na rovnakých simulovaných cestách.
- Prehľad zdroja, čerstvosti a pokrytia dát pre každý ticker; pri starých dátach alebo nízkom pokrytí sa zobrazí upozornenie.
- Export CSV pre akcie, váhy, portfóliá a spätný test, súhrnný PDF report a JSON nastavení so seed-om a SHA-256 odtlačkom dát.

## Náklady a metodické obmedzenia

Model nákladov zahŕňa jednorazovú províziu pri nákupe aj predaji, používateľský odhad spreadu a sklzu za obchod, ročný priebežný poplatok (TER), nákladový proxy pre zadaný ročný obrat pri rebalansovaní a zjednodušenú daň z kladného konečného kapitálového zisku. Ročný obrat je proxy: aplikácia neprehráva skutočné rebalansovacie obchody po jednotlivých cestách. Daň je plochá používateľská aproximácia, nie výpočet individuálnej daňovej povinnosti. Spread a sklz sú používateľský odhad a likvidita sa neoveruje živými objemami.

Yahoo Finance je zdroj dát aplikácie, nie garantovaný produkčný feed; dostupnosť a kvalita údajov sa môžu meniť. Päť rokov dát nemusí zachytiť všetky trhové režimy. Pri 63-dňovom horizonte historického stresového testu pozorovaný stresový blok tvorí celý horizont. Walk-forward testovacie okná sa prekrývajú, preto nie sú nezávislé. Katalóg dnešných aktívnych spoločností neodstraňuje survivorship bias. Heuristický verdikt a simulácie nie sú investičným odporúčaním ani zárukou budúcich výnosov.

## Testy

Spustenie testov vo Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Spustenie testov v Linuxe alebo macOS po aktivácii virtuálneho prostredia:

```bash
python -m unittest discover -s tests -v
```

Testy pomocných funkcií používajú syntetické dáta alebo lokálny katalóg a nesťahujú Yahoo Finance. Zahŕňajú vyhľadanie Visa podľa názvu/tickeru aj ručné zadanie `Visa (V)`.

## Codex agenti

V `.codex/agents/` sú projektové read-only profily pre kvantitatívnu analýzu, dátovú vedu, UX kontrolu, code review a stručné odovzdanie kontextu. Sú odvodené z [VoltAgent/awesome-codex-subagents](https://github.com/VoltAgent/awesome-codex-subagents) a použiteľné pri explicitnom delegovaní v Codexe. Profily nespúšťajú automaticky ďalších agentov. Licencia MIT a pôvod sú zaznamenané v [`.codex/agents/README.md`](.codex/agents/README.md).

## Pôvodný dávkový skript

Pôvodný dávkový nástroj ostáva zachovaný:

```powershell
.\.venv\Scripts\python.exe monte_carlo_akcie.py
```

V Linuxe/macOS po aktivácii virtuálneho prostredia:

```bash
python monte_carlo_akcie.py
```

Výstupy skriptu ukladá do `mc_vysledky/`.
