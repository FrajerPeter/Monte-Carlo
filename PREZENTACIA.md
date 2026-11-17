# Prezentačný scenár aplikácie

## Krátky úvod (30 sekúnd)

„Táto aplikácia porovnáva možné scenáre vývoja akcií a portfólií. Historické ceny sťahuje z Yahoo Finance, upraví ich na dividendy a splity, podľa potreby ich prepočíta do meny účtu a z historických výnosov vytvorí tisíce simulovaných ciest. Výsledky čítame ako rozsah neistoty a rizika, nie ako presnú predpoveď ceny.“

## Ako ju ukázať

1. Spusti aplikáciu príkazom `.\.venv\Scripts\python.exe -m streamlit run app.py` a otvor `http://localhost:8501`.
2. V bočnom paneli zadaj tickery, napríklad `AAPL, MSFT, NVDA`, benchmark `SPY`, horizont a počet simulácií. Ukáž menu účtu a nákladové predpoklady.
3. Vyber model a stlač **Spustiť simuláciu**. Zmena parametrov výsledky sama neprepočíta; treba simuláciu spustiť znova.
4. Na grafe vysvetli sivé čiary ako ukážkové individuálne cesty, prerušované čiary ako 10. a 90. percentil a modrú čiaru ako medián všetkých simulácií. Ide o ceny pred nákladmi; KPI a koncové hodnoty investície náklady zohľadňujú.
5. V kartách ukáž priemerný a mediánový výsledok, VaR/CVaR, Sharpe, pravdepodobnosť zisku a pravdepodobnosť prekonať benchmark.
6. V záložke Portfólio porovnaj rovnomerné váhy s približným maximom Sharpe, minimálnou volatilitou a minimálnym historickým CVaR.
7. V Stresovom scenári ukáž rozdiel medzi nastaviteľným okamžitým poklesom a historickým najhorším spoločným 63-dňovým blokom. Po historickom bloku sa modeluje zvyšok horizontu.
8. V Backtest ukáž porovnanie piatich modelov na rovnakých dátumoch. Vyber model, aby sa zobrazil jeho časový priebeh predikovaných pásiem a skutočných výsledkov.
9. V Exporte stiahni tabuľky do CSV a stručný súhrn do PDF.

## Vysvetlenie modelov (približne 1 minúta)

- **GBM:** zachováva historickú volatilitu a koreláciu, predpokladá normálne denné šoky a stálu volatilitu.
- **Blokový bootstrap:** vyberá 20-dňové bloky historických spoločných výnosov. Zachováva pozorované extrémy a krátkodobé zhluky výnosov, ale môže opakovať iba to, čo sa v minulosti skutočne stalo.
- **Student-t GBM:** podobný GBM, ale pripúšťa častejšie veľké pohyby v chvostoch rozdelenia.
- **GARCH(1,1) + Student-t:** mení budúcu volatilitu podľa predchádzajúcich šokov; parametre sa vyberajú z malej obmedzenej mriežky.
- **Prepínanie režimov:** modeluje pokojný a stresový stav. Stav sa odhaduje pomocou 20-dňovej volatility a pravdepodobností prechodu pozorovaných v histórii.

Historické denné výnosy sa pri odhade priemeru orežú na 1. a 99. percentile a priemer sa zmenší k bezrizikovej sadzbe. Tým sa obmedzí vplyv extrémnych denných pozorovaní na drift. Parameter neistoty potom pridáva každému scenáru osobitnú odchýlku drifta podľa štandardnej chyby; zobrazuje citlivosť na neistotu priemeru, nie všetku neistotu trhu.

## Metriky jednoducho

- **Priemer:** priemerný výsledok zo všetkých simulovaných koncových hodnôt. Môžu ho výrazne ovplyvniť extrémne scenáre.
- **Medián:** stredný scenár; polovica výsledkov je pod ním a polovica nad ním.
- **VaR 95 %:** strata na 5. percentile. Približne 5 % simulácií dopadlo horšie.
- **CVaR 95 %:** priemerná strata v najhorších 5 % simulácií.
- **Sharpe:** anualizovaný pomer nadvýnosu oproti bezrizikovej sadzbe k volatilite. Je založený na historickom odhade a nezohľadňuje všetky riziká.
- **Pravdepodobnosť zisku:** podiel simulácií s kladným výnosom po zadaných nákladoch.
- **Pravdepodobnosť prekonať benchmark:** párové porovnanie v tých istých korelovaných scenároch.
- **Maximálny pokles:** najväčší pokles od predchádzajúceho maxima počas celej simulovanej cesty; neznamená iba stratu na konci horizontu.

## Backtest

„Backtest si kladie otázku, ako by si model počínal v minulosti, keby sme ho vtedy spustili bez prístupu k budúcim dátam. Pri každom dátume používa predchádzajúce tri roky a simuluje ďalších 63 obchodných dní. Dátum posunie o 21 dní a postup opakuje. Porovnávame pokrytie predikčného pásma, správnosť smeru mediánu a chybu mediánu. Okna sa prekrývajú, takže výsledky nie sú nezávislé a spätná úspešnosť nezaručuje budúcu presnosť.“

## Záver a poctivé obmedzenia

„Výsledky sú citlivé na výber obdobia, modelu, driftu, FX a nákladov. Model nepozná budúce správy, zmeny fundamentov ani krízy, ktoré v tréningových dátach neboli. Optimalizované portfólio je najlepšie z preskúšaných náhodných váh, nie matematicky garantované globálne optimum. Aplikáciu preto používam na porovnávanie scenárov a odhaľovanie rizík, nie ako automatický pokyn na nákup.“

## Typické otázky

**Prečo graf vyzerá ako náhodná prechádzka?**  
Pretože každá ďalšia simulovaná cena závisí od náhodného výnosového šoku. Rôzne modely menia predpoklady o chvostoch, zhlukovaní volatility, korelácii a režimoch, ale stále nejde o deterministickú predpoveď.

**Odkiaľ sú ceny?**  
Z Yahoo Finance cez knižnicu `yfinance`, päť rokov denných cien s automatickou úpravou o korporátne udalosti. Aplikácia zobrazuje dátum poslednej dostupnej ceny. FX kurzy tiež pochádzajú z Yahoo Finance.

**Čo znamená historický stres?**  
Nájde najhorší súvislý 63-dňový úsek rovnomerne váženého portfólia aktuálne sledovaných tickerov v dostupných dátach. Prehrá rovnaké denné výnosy všetkých titulov a na zvyšok zvoleného horizontu aplikuje zvolený model. Ak horizont sám trvá 63 dní, celý scenár je historický.

**Je výsledok odporúčaním na nákup?**  
Nie. Semafor je heuristický skríning podľa zobrazených prahov; nenahrádza vlastné posúdenie, fundamentálnu analýzu ani finančné poradenstvo.

## Lokálne overenie po úpravách

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```
