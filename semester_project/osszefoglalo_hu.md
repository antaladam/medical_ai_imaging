# Kiinduló MI-alkalmazások és egy saját, bizonyítékra épülő kép + szöveg rendszer mellkasröntgen-döntéstámogatáshoz

**Félévi projekt – összefoglaló**

---

## 1. A kiinduló kérdés

A kiírt szakdolgozati téma (4. projekt) címe: *A Multimodal, Explainable AI Decision-Support Prototype for Medical Image and Clinical-Text Analysis*. A javasolt kutatási kérdés:

> Javíthatja-e a bizonyítékokra épülő multimodális visszakeresés az LLM által írt magyarázatok hasznosságát és ellenőrizhetőségét az orvosi képeken alapuló döntéstámogatásban?

A kiírás feltételei:

- a rendszer **kizárólag döntéstámogató** lehet, önálló diagnózist nem adhat;
- csak **nyilvános vagy anonimizált adat** használható;
- a munka három megadott, nyílt forráskódú kiinduló alkalmazásra épül;
- mérni kell a képértelmezést, a bizonyítékra való támaszkodást, a magyarázatok minőségét és a hibákat, és össze kell vetni a csak képes és a kép + szöveges megközelítést.

A projekt ezt egy ellenőrizhető kérdéssé alakította: **jobban ismeri-e fel a mellkasröntgen-elváltozásokat egy olyan rendszer, amely a képhez klinikai szöveget és hasonló korábbi eseteket is kap, mint egy önálló, erre a feladatra tanított képmodell?** Az eredmény bármelyik irányban értékes: ha nincs különbség, az is válasz a kérdésre.

A konkrét feladat: a nyilvános IU X-Ray gyűjtemény frontális mellkasröntgen-felvételein **hat elváltozás** felismerése:

| Angol név | Magyar név |
|---|---|
| Lung Opacity | tüdőárnyék |
| Cardiomegaly | kardiomegália (megnagyobbodott szív) |
| Atelectasis | atelektázia (tüdőösszeesés) |
| Lung Lesion | tüdőlézió (góc, csomó, terime) |
| Pleural Effusion | pleurális folyadék |
| Fracture | csonttörés |

---

## 2. A részkérdések

A félévi munka három kérdésre bontotta a feladatot:

- **Q1 – Alkalmasság:** Hogyan működik a három kiinduló alkalmazás, és mennyire ismerik fel a mellkasröntgen-elváltozásokat a lehető legkisebb átalakítással?
- **Q2 – Erősségek:** Mely részeiket érdemes átvenni egy képet és szöveget együtt kezelő, bizonyítékra hivatkozó rendszerbe, és milyen hiányokat kell pótolni?
- **Q3 – Összevetés:** Mennyiben térnek el a saját rendszer (G) eredményei a kiinduló alkalmazásokétól, egy specialista képmodelltől (S0) és a puszta visszakereséstől (R0)?

Összesen nyolc rendszer futott ugyanazon a 400 tesztvizsgálaton:

| Csoport | Rendszer | Röviden |
|---|---|---|
| Kiinduló alkalmazások | P1, P2, P3 | a három megadott alkalmazás, minimális átalakítással |
| Saját rendszer | G | a beteg képe + klinikai indikáció + 10 hasonló eset lelete → Gemini |
| | G−ind | G klinikai indikáció nélkül |
| | G-img | G modellje és promptja, de csak a képpel (sem eset, sem indikáció) |
| Referencia (LLM nélkül) | S0 | specialista képmodell (előre tanított mellkasröntgen-osztályozó) |
| | R0 | csak visszakeresés: a leghasonlóbb tanítóesetek címkéinek szavazása |

---

## 3. Az adathalmaz

### 3.1 Az IU X-Ray gyűjtemény

Az Indiana University mellkasröntgen-gyűjteménye, amelyet az amerikai National Library of Medicine tesz közzé az Open-i szolgáltatáson keresztül (Demner-Fushman és mtsai., *JAMIA* 2016). Két indianai kórházi rendszer **járóbeteg-vizsgálatai**, ezért túlnyomórészt normális vagy enyhe eltéréseket mutató felvételek. Licenc: CC BY-NC-ND 4.0.

- **3 955 lelet** és **7 470 kép**.
- Minden vizsgálat egy XML-lelet négy szakasszal: COMPARISON (összehasonlítás korábbi felvétellel), INDICATION (a kérés oka, vagyis a klinikai kérdés), FINDINGS (leírás), IMPRESSION (vélemény).
- Általában egy frontális és egy oldalirányú kép tartozik hozzá, PNG formátumban.
- A gyűjtemény szerzői minden lelethez kézzel MeSH-kódokat (Medical Subject Headings) rendeltek.
- Az anonimizálás során eltávolított szövegrészek helyén „XXXX” áll.

### 3.2 Tisztítás és ellenőrzés

| Lépés | Eredmény |
|---|---|
| Kép–lelet megfeleltetés | Mind a 7 470 PNG pontosan egy lelethez tartozik, és mind megnyitható. A képek RGB formátumúak, de tartalmuk szürkeárnyalatos. |
| Betegazonosító | A kiadásban **nincs betegazonosító**, csak vizsgálat- és képazonosító. A képnevek „IM-####” része sorszám, nem betegkapcsolat (egy-egy csoportjában egyértelműen különböző betegek vannak). A gyűjteményt leíró cikk szerint viszont betegenként **legfeljebb egy vizsgálat** került be, így a vizsgálatonkénti felosztás egyben betegenkénti is: ugyanaz a beteg nem kerülhet a tanító- és a teszthalmazba is. |
| Nézetcímke | A kiadás nem jelöli, melyik kép frontális és melyik oldalirányú. A nézetet BiomedCLIP zero-shot osztályozás határozta meg (három „frontális” és három „oldalirányú” szöveges leírással). Szabály: vizsgálatonként a legfrontálisabb kép kerül kiválasztásra, ha a frontális valószínűsége legalább 0,05. Szemrevételezéses ellenőrzés: 60 véletlen kép közül 60 helyes, a 11 alacsony pontszámú kiválasztott kép mind frontális, 40 kizárt vizsgálat legjobb képe mind oldalirányú. |
| Használhatatlan képek | 4 kép kiszűrve (3 szinte üres, 1 tiszta zaj); egyetlen vizsgálat sem veszítette el emiatt az egyetlen használható képét. |
| Duplikátumok | BiomedCLIP-hasonlóság alapján nincs 0,99 feletti képpár; a 30 leghasonlóbb pár szemrevételezve mind különböző beteg. Pixelszinten nincs azonos fájl és azonos kicsinyített kép sem. |

A tisztítás menete:

| | Vizsgálat |
|---|---|
| Leletek a kiadásban | 3 955 |
| − kép nélküli leletek | −104 |
| − csak oldalirányú felvétellel rendelkezők | −161 |
| = frontális képpel rendelkező vizsgálatok | 3 690 |
| − kézi kódolás nélküliek („No Indexing”, nincs referenciacímke) | −86 |
| **= a vizsgálatban használt esetek** | **3 604** |

A kizárt, csak oldalirányú felvétellel rendelkező 161 eset valamivel betegebb (több kardiomegália, több eszköz, kevesebb normális lelet). A képes esetek 4%-áról van szó, de szelekciós hatásként számolni kell vele.

### 3.3 Referenciacímkék

**Az eredeti terv** a leletekből a CheXbert nyelvi modellel automatikusan kinyert címkék használata volt. Ezeket a szerzők kézi MeSH-kódjaival összevetve **szisztematikus hiba** derült ki: egy sablonos normális lelet („… Both lungs are clear and expanded. Heart and mediastinum normal.”) 62 előfordulásából 55-öt kardiomegáliának címkézett, ami a CheXbert összes kardiomegáliás címkéjének 13%-a. Az anonimizálási jelölések eltávolítása nem segített.

**Döntés:** a referenciacímkék a **kézi MeSH-kódokból** származnak, egy minden rendszer futtatása előtt rögzített megfeleltetéssel. A hat elváltozás kiválasztása az esetszám és az S0 lefedettsége alapján történt. A „Support Devices” (eszközök) kategóriának elég esete lett volna, de az S0 nem tudja pontozni, ezért kimaradt.

| Elváltozás | Pozitív, ha a kódok között szerepel … | Pozitív esetek |
|---|---|---|
| Tüdőárnyék | árnyék/denzitás a tüdőben, légtéri betegség, infiltrátum, konszolidáció | 519 (14,4%) |
| Kardiomegália | kardiomegália vagy megnagyobbodott szívárnyék („határérték” = negatív) | 295 (8,2%) |
| Atelektázia | pulmonális atelektázia | 296 (8,2%) |
| Tüdőlézió | csomó, terime vagy üreg a tüdőben (meszes granuloma nem számít) | 104 (2,9%) |
| Pleurális folyadék | pleurális folyadék | 138 (3,8%) |
| Csonttörés | csonttörés (gyógyult is) | 83 (2,3%) |
| A hatból egyik sem | | 2 623 (72,8%) |

### 3.4 Adatfelosztás

| Halmaz | Esetek | Tüdőárnyék | Kardiomegália | Atelektázia | Tüdőlézió | Pleurális folyadék | Csonttörés | Egyik sem | Felhasználás |
|---|---|---|---|---|---|---|---|---|---|
| Tanító | 2 804 | 396 | 206 | 221 | 49 | 83 | 43 | 2 112 | a visszakeresés egyetlen forrása; S0 és R0 küszöbei |
| Validációs | 400 | 39 | 23 | 26 | 11 | 12 | 2 | 319 | G küszöbe; ellenőrzések |
| – ebből előteszt | 50 | 5 | 2 | 1 | 1 | 1 | 0 | 42 | promptfejlesztés, viselkedés |
| Teszt | 400 | 84 | 66 | 49 | 44 | 43 | 38 | 192 | az egyetlen összevetés |

- A felosztás rögzített véletlenmaggal (2026) készült; a halmazok diszjunktak, és együtt lefedik mind a 3 604 esetet.
- A **teszthalmaz pozitív esetekkel dúsított**: a ritkább elváltozásokkal kezdve addig kerültek bele pozitív esetek, amíg mindegyik elváltozásból legalább 35 lett, a maradékot véletlen esetek töltik ki. Ez nem érinti a szenzitivitást, a specificitást és az AUROC-t, az F1 és a pozitív prediktív érték viszont nem vethető össze a természetes gyakorisággal.
- A validációs halmaz természetes gyakoriságú, ezért a ritka elváltozásokból kevés van benne (csonttörés 2, tüdőlézió 11). **Döntés:** az S0 és az R0 küszöbei a tanítóhalmazon készülnek, G-é egyetlen közös küszöbként a validációs halmazon.

---

## 4. A kiinduló rendszerek (Q1)

### 4.1 Közös keretek

- A felhasználói felületek helyett mindhárom alkalmazás kötegelt futtatóból fut, amely a saját kódjukat hívja.
- **Modell:** a P1 (`gemini-2.5-pro`) és a P2 (`gemini-2.5-flash`) eredeti modelljét új felhasználók már nem érhetik el. Mindhárom alkalmazás a P3 modelljét, a **`gemini-3-flash-preview`**-t használja, így a köztük lévő különbség a felépítésből fakad, nem a modellből.
- Mindhárom ugyanazt a rögzített kérdést kapja:
  *„Which of these findings are present: Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion, Pleural Effusion, Fracture? Begin your answer with exactly six lines, one per finding, in the form "<finding>: present", "<finding>: absent" or "<finding>: uncertain". Then continue as instructed.”*

**Miért kell az ellenőrzőlista a válasz elejére?** Eredetileg a szabad szöveges válaszokat a CheXbert címkézte volna. Az előteszten azonban rendszeresen félreolvasta őket:

| Mondat a válaszból | CheXbert | Valódi jelentés |
|---|---|---|
| „Cardiomegaly: Not present.” | pozitív | negatív |
| „Cardiomegaly refers to an enlarged heart …” (definíció) | pozitív | nem a betegről szól |
| „Pleural Effusion was specifically ruled out …” | pozitív | negatív |

**Döntés:** az alkalmazások saját promptja változatlan marad, csak a kérdés kapott formai utasítást. Minden rendszer pontozása a saját ellenőrzőlistája alapján történik: „present” = igen; „absent”, „uncertain” és a hiányzó sor = nem. A teszten mind az 1 200 lista értelmezhető volt. 45 válasz kézi ellenőrzésén a lista 270 elváltozásból 268-ban egyezett a válasz többi részével, ellentmondás egyszer sem fordult elő.

### 4.2 P1 – `ai_medical_imaging_agent`

**Eredeti formájában:** Streamlit-alkalmazás. A felhasználó feltölt egy orvosi képet, egy agno-ágens (Gemini 2.5 Pro, DuckDuckGo webes keresőeszközzel) pedig egy rögzített, ötrészes prompt alapján leletet ír. Kérdést nem lehet feltenni, a prompt mindig ugyanaz:

1. képtípus és régió;
2. fő eltérések;
3. diagnosztikai értékelés (elsődleges diagnózis bizonyossággal, differenciáldiagnózisok);
4. betegnek szóló magyarázat;
5. kutatási háttér („IMPORTANT: Use the DuckDuckGo search tool …”: szakirodalom, protokollok, linkek, hivatkozások).

**Változtatások (P1-adapted-ns):**

- kötegelt futtatás, `gemini-3-flash-preview`;
- a webes keresés kikapcsolva, mert az élő keresés nem megismételhető;
- a kérdés a prompt végére került, mivel a P1-nek nincs kérdésmezője;
- **az 5. rész törölve:** keresőeszköz nélkül a prompt továbbra is a keresést írta elő. A modell egy nem létező eszközt próbált hívni, és az előteszt 50 esetéből **31 (62%) hibával (`MALFORMED_FUNCTION_CALL`) zárult**. Az 5. rész nélkül 50/50 válasz érkezett. Az eredeti prompttal adott válaszok emlékezetből kitalált hivatkozásokat is tartalmaztak.

### 4.3 P2 – `vision_rag`

**Eredeti formájában:** Streamlit-alkalmazás. A feltöltött képeket a Cohere embed-v4 beágyazza; a felhasználó szöveges kérdése alapján az alkalmazás megkeresi a leghasonlóbb **egyetlen** képet (top-1), és csak ezt, a kérdéssel együtt küldi a Gemini 2.5 Flash-nek. A prompt kéri, hogy a válasz „a lehető legrészletesebb” legyen.

**Változtatások (P2-adapted):**

- kötegelt futtatás, `gemini-3-flash-preview`;
- Gemini Embedding 2 a Cohere helyett (nincs Cohere-fiók); 3 072 dimenziós vektorok a 2 804 tanítóképről;
- a keresés a beteg saját képével történik szöveges kérdés helyett;
- **a beteg képe is a modellhez kerül:** eredetileg csak a visszakeresett kép jutna el a modellhez, amely így egy másik beteg felvételét írná le. A promptba egy mondat került, amely megnevezi a két képet.

A „legyen a lehető legrészletesebb” utasítás miatt az előteszt 50 válaszából 38 tankönyvi definíciókat is tartalmazott az elváltozásokról.

### 4.4 P3 – `multimodal_agentic_rag`

**Eredeti formájában:** React + FastAPI alkalmazás egy multimodális tudásbázis fölött.

- Minden forrást a Gemini Embedding 2 ágyaz be (768 dimenzió); a képek vektora 0,68 : 0,32 arányban keveredik a hozzájuk tartozó szöveges leírás vektorával.
- Egy Google ADK-ágens két eszközzel (a beágyazási tér áttekintése, visszakeresés) a 6 legjobb találatból válaszol, hivatkozásokkal (`gemini-3-flash-preview`, hőmérséklet 0,25, legfeljebb 900 kimeneti token).
- **Lényeges tulajdonság:** a válaszgenerátor csak a visszakeresett dokumentumok szövegét kapja meg, **képet soha nem lát.**

**Változtatások (P3-adapted):**

- kötegelt futtatás az alkalmazás saját kódjával;
- az index a 2 804 tanítókép, mindegyik a saját leletével (FINDINGS + IMPRESSION) mint szöveges leírással;
- a keresés a beteg képének beágyazásával történik;
- **a 900 tokenes kimeneti korlát eltávolítva:** ez a korlát ennél a modellnél a „gondolkodást” is magában foglalja, és az ellenőrzőlistás kérdésnél az 50 válaszból 45-öt még a lista közepén levágott. Minden más (ágens, prompt, top-6, hőmérséklet, modell) változatlan.

A korlát nélkül a válaszok egy része elszabadult gondolkodásba futott (kb. 63 000 token): az előteszten 50-ből 3, a teszten 400-ból 11 esetben.

### 4.5 Viselkedés az előteszten és a Q1 válasza

Az 50 esetes előteszten csak 10 pozitív elváltozás volt, ezért az eredmények a viselkedést mutatják, nem a pontosságot:

| | P1-adapted-ns | P2-adapted | P3-adapted |
|---|---|---|---|
| Kiadott „igen” válaszok | 31 | 11 | 75 |
| ebből helyes | 5 | 3 | 6 |
| Normálisnak ítélt normális esetek (42-ből) | 34 | 40 | 16 |
| Jellemző | rendszerezett lelet, a képet olvassa | óvatos, egy szomszédra támaszkodik | túl sokat jelez, a szomszédok eltéréseit jelenti a betegre |

**Q1 válasza:** egyik alkalmazás sem működött mellkasröntgenen eredeti formájában: megszűnt modellek, egy eltávolított eszközhöz kötött prompt és egy gondolkodást is magában foglaló kimeneti korlát akadályozta őket. Átalakítva mindhárom minden esetre válaszol, de ítéleteik durvák. A P1 és a P2 a beteg képét olvassa; a P3 nem, valójában a visszakeresett leletek fölötti szavazást fogalmazza meg szövegben.

---

## 5. A referenciarendszerek: R0 és S0

Mindkét rendszer folytonos pontszámot ad elváltozásonként. Az igen/nem döntéshez egy küszöb kell, amelyet mindkét rendszer ugyanazzal a szabállyal kap: minden lehetséges küszöb kipróbálásra kerül, és az marad meg, amelynél a **Youden-index** (szenzitivitás + specificitás − 1) a legnagyobb, egyenlőség esetén a magasabbat.

**Miért a Youden-index?** A küszöb mindig kompromisszum: alacsonyabb küszöbbel több beteg eset akad fenn (nő a szenzitivitás), de több a téves riasztás (csökken a specificitás). A Youden-index a kettőt egyenlő súllyal veszi figyelembe, és – mivel csak a szenzitivitásból és a specificitásból áll – nem függ attól, mennyire gyakori az elváltozás. Ez azért fontos, mert a küszöbök természetes gyakoriságú adaton készülnek, a teszthalmaz viszont dúsított. Egy olyan küszöb, amely például az F1-et maximalizálja a ritka elváltozásokkal teli tanítóhalmazon, túl óvatos lenne a sok pozitív esetet tartalmazó teszthalmazon. Ugyanezt a szabályt kapja később G is, így minden hangolt rendszer egyformán hangolt.

### 5.1 R0 – csak visszakeresés, LLM nélkül

**Működés:** a BiomedCLIP a frontális képet egy 512 elemű vektorrá alakítja. A rendszer koszinusz-hasonlóság alapján megkeresi a k leghasonlóbb tanítóvizsgálatot, és elváltozásonként a pontszám azoknak a szomszédoknak az aránya, amelyekben az adott elváltozás megvan.

- **Az index csak a tanítóhalmaz;** egy vizsgálat sosem találhatja meg önmagát (ellenőrizve).
- **k megválasztása** leave-one-out módszerrel a tanítóhalmazon: minden tanítóvizsgálat pontszáma a többi 2 803 alapján készült, és a legmagasabb makró AUROC-t adó k lett kiválasztva.

| k | 1 | 5 | 10 | 20 | 50 | 100 | **150** | 300 | 1 000 |
|---|---|---|---|---|---|---|---|---|---|
| Makró AUROC | 0,572 | 0,651 | 0,671 | 0,704 | 0,712 | 0,741 | **0,741** | 0,740 | 0,739 |

- **k = 150.** A teljesítmény kb. k = 100-tól platózik. Tanulság: az R0 jele egy elváltozás gyakorisága egy tág környezetben, nem néhány szinte azonos eset; egyetlen szomszéd alig hordoz információt (0,572).
- A nagy, az egész képet érintő elváltozásoknál jól működik (kardiomegália 0,891, pleurális folyadék 0,898), a kis, gócos elváltozásoknál viszont minden k-nál a véletlen szintjén marad (tüdőlézió 0,555, csonttörés 0,565).
- Makró AUROC: tanítóhalmaz (leave-one-out) 0,741, validációs halmaz 0,700.

### 5.2 S0 – specialista képmodell

**Működés:** frontális kép → négyzetesre kiegészítés fekete szegéllyel → átméretezés 224 pixelre → DenseNet-121 (TorchXRayVision könyvtár) → elváltozásonként egy valószínűség → elváltozásonkénti küszöb. Nincs LLM, nincs szöveg, nincs visszakeresés. A modell zero-shot módon fut: más kórházak adatain tanították, IU-adaton soha.

**Megengedett súlyok:** a súlyok nem tanulhattak az Open-i (= IU X-Ray) adatain, különben az S0 olyan képeken vizsgázna, amelyeket már látott. Ez kizárta a könyvtár kombinált DenseNet-súlyait. Az egyetlen adathalmazon tanított DenseNetek közül csak a CheXpert, a MIMIC-nb és a MIMIC-ch változat ad kimenetet mind a hat elváltozásra.

**Biztonságos betöltés:** a könyvtár a súlyfájlokat olyan módon tölti be, amely a fájlban tárolt kódot is lefuttathatja. Helyette egy korlátozott betöltő készült, amely csak a fájlokban ténylegesen előforduló osztályokat engedi; a kimenetek bitre azonosak a könyvtár saját betöltőjéével.

**Előfeldolgozás:** a könyvtár dokumentált lépése a kép közepének kivágása. Az IU-felvételek nagy része fekvő tájolású, és a kivágás levágja a mellkas széleit, ahol a pleurális folyadék és a bordatörések látszanak. Ezért a négyzetesre kiegészítés is kipróbálásra került.

**Kiválasztás a tanítóhalmazon** (előre rögzített szabály: a legmagasabb makró AUROC):

| Súlyok | Előfeldolgozás | Makró AUROC (tanító) |
|---|---|---|
| DenseNet CheXpert | kivágás / kiegészítés | 0,691 / 0,683 |
| DenseNet MIMIC-nb | kivágás / **kiegészítés** | 0,713 / **0,728 (kiválasztva)** |
| DenseNet MIMIC-ch | kivágás / kiegészítés | 0,711 / 0,712 |
| ResNet-50 (csak viszonyításként) | kiegészítés | 0,808 |

- A kiegészítés ott segített, ahol a kivágás anatómiát vágott le: a pleurális folyadéknál 0,842 → 0,868, a csonttörésnél 0,616 → 0,676.
- **A ResNet-50 egyértelműen erősebb** (+0,080), és szintén nem tanult IU-adaton. A tüdőlézió-kimenete azonban betanítatlan (egyik tanító-adathalmaza sem definiálja), ezért csak helyettesítő értékkel (a csomó és a terime közül a nagyobb) pontozhatná. **Döntés:** az S0 a terv szerinti DenseNet marad; a ResNet-50 csak viszonyítási pont, a teszthalmazon nem futott. Következmény: az S0 nem a legerősebb elérhető specialista, ezért minden „G jobb, mint S0” eredmény erre a modellre vonatkozik.
- A tüdőlézió minden DenseNetnél a véletlen közelében van (0,49–0,56). Ennek oka **nem a definíciók eltérése**. Az S0 tanítócímkéit a MIMIC-CXR leleteiből a NegBio program nyerte ki, a CheXpert-címkéző kulcsszólistájával. E szerint tüdőlézió a csomó, terime, csomószerű árnyék, üreges lézió, daganat (mass, nodule, nodular density/opacity, lump, cavitary lesion, carcinoma, neoplasm, tumor), a meszes csomó viszont nem számít. Ez lényegében megegyezik a projekt referenciacímkéjével (csomó, terime vagy üreg a tüdőben, meszes granuloma nélkül). Az eltérés a címkék előállításában van: a MIMIC-címkék kulcsszavas szabályokkal, a lelet IMPRESSION szakaszából készültek, a bizonytalan említések a tanításból kimaradtak, az IU-kódokat viszont emberek adták. Hogy a modell gyengeségét mi okozza – a zajos, automatikusan kinyert tanítócímkék, a 224 pixeles képen nehezen látható kis gócok, vagy a kórházak közötti különbség –, az nem állapítható meg. Ilyen gyenge pontszámon a Youden-küszöb szélsőségesen alacsony lesz (tüdőlézió: 0,0072), vagyis az S0 szinte mindenre igent mond.
- Makró AUROC: tanító 0,728, validációs 0,757.

---

## 6. A saját rendszer: G (Q2)

### 6.1 Mit érdemes átvenni, és mi hiányzik

| Forrás | Átvett elem | Változtatás G-ben | Indok |
|---|---|---|---|
| P1 | a beteg képének rendszerezett radiológiai olvasása | pontos elváltozás-definíciók, rövid „munkamenet”, JSON a markdown helyett | a definíciók egyezzenek a referenciacímkékkel; géppel olvasható válasz |
| P2 | beágyazás → visszakeresés → generálás, a beteg képe eljut a modellhez | BiomedCLIP általános beágyazó helyett; 10 eset 1 helyett; a leleteik, nem a képeik | egy szomszéd alig hordoz jelet; a BiomedCLIP jobb szomszédokat talál |
| P3 | visszakeresett leletek mint bizonyíték; hivatkozási objektumok | elváltozásonkénti hivatkozás esetazonosítóval, kóddal ellenőrizve; a beteg képe is a modellhez kerül | a P3 sosem látja a beteget; hivatkozásai elváltozásonként nem ellenőrizhetők |
| Új | – | klinikai indikáció; valószínűség elváltozásonként; egyetlen közvetlen hívás ágens nélkül | egyik alkalmazás sem fogad klinikai szöveget; az AUROC-hoz pontszám kell; az ágensek rejtett újrapróbálkozásokat és elszabadult válaszokat okoztak |

**A beágyazók összevetése** (R0 szavazása leave-one-out módszerrel a tanítóhalmazon, makró AUROC, k = 5): BiomedCLIP 0,651; Gemini Embedding 2 0,574 (különbség −0,077 [−0,100; −0,054]); a P3 kép + lelet keveréke 0,595. Ennyi esetnél, amennyi egy promptba belefér, a BiomedCLIP egyértelműen informatívabb szomszédokat talál. A kivágás vagy kiegészítés a visszakeresésnél nem számított.

### 6.2 Felépítés

1. A beteg frontális képe → BiomedCLIP → a **10** leghasonlóbb tanítóvizsgálat (ugyanaz az index, mint az R0-é: G 10 szomszédja mindig az R0 első 10 szomszédja).
2. A prompt tartalma: döntéstámogató szerep; a hat elváltozás definíciója a referenciacímkék szerint (pl. a csomó vagy terime tüdőlézió, nem tüdőárnyék; a határérték-szívméret nem kardiomegália; a régi, gyógyult törés is számít); munkamenet (először a beteg képét kell rendszerezetten olvasni, a hasonló esetek más betegek és csak gyenge, alátámasztó bizonyítékok, nem a válasz); a klinikai indikáció; a 10 szomszéd lelete azonosítóval és hasonlósággal; végül a kép.
3. **Egyetlen** `gemini-3-flash-preview`-hívás, eszközök nélkül, alapértelmezett hőmérséklettel és gondolkodási szinttel, kimeneti korlát nélkül. A JSON-sémát az API kényszeríti ki.
4. A válasz elváltozásonként: **státusz** (present / absent / uncertain), **valószínűség** (0–1), a **hivatkozott esetek azonosítói** és egy-két mondatos **indoklás**, végül egy rövid összegzés.
5. Ellenőrzés kóddal: pontosan a hat elváltozás szerepel, a valószínűség 0 és 1 közé esik, és minden hivatkozott azonosító a megkapott 10 eset között van.

**Promptfejlesztés** (csak az 50 esetes előteszten, legfeljebb 3 változat): az 1. változatnál 3 válasz a lista sorszámával hivatkozott, és a valószínűségek durvák voltak (300-ból 198 pontosan 0,05). A 2. változatban az esetek sorszám nélkül, azonosítóval szerepelnek, és a prompt két tizedesjegyű valószínűséget kér; ezzel 0 hibás hivatkozás maradt. A **2. változat rögzítésre került**, ezután nem változott.

**Ablációk:** a **G−ind** azonos G-vel, csak az indikáció sora és az arra utaló lépés hiányzik. A **G-img** G modellje, promptja és kimeneti formátuma, de csak a képpel (sem eset, sem indikáció).

**Küszöb:** G státusza („present”) nagyon óvatos, valószínűségéből viszont küszöbbel igen/nem is képezhető. A küszöb a validációs halmaz 400 esetén, a hat elváltozás 2 400 válaszán közösen, Youden-index szerint készült: **0,08** (szenzitivitás 0,64, specificitás 0,82); G−ind-nél is 0,08 lett. Validációs makró AUROC: G 0,770, G−ind 0,724 (S0 0,757, R0 0,700). A G-img-nek nincs küszöbe; értékelése AUROC és saját státusz alapján történik.

---

## 7. Tesztelés és eredmények (Q3)

### 7.1 Protokoll

- Mind a nyolc rendszer **egyszer** futott ugyanazon a 400 tesztvizsgálaton, miután minden választás (promptok, k, küszöbök, értékelési módszer) rögzítve lett a tanító- és a validációs halmazon. Minden szkript megtagadja a teszthalmaz használatát, ha erre nem kap kifejezett utasítást.
- Mind a 400 vizsgálatra mind a nyolc rendszer válaszolt, egyik válaszból sem hiányzott ítélet.
- **Előre rögzített összevetések** (a tesztfuttatás előtt; ezek közül csak az 1. számít fő eredménynek):

| # | Összevetés | Mérőszám | Kérdés |
|---|---|---|---|
| **1 (fő)** | **G − S0** | makró AUROC | Jobb-e a kép + esetek + szöveg a specialista képmodellnél? |
| 2 | G − R0 | makró AUROC | Hozzáad-e az LLM a puszta visszakereséshez? |
| 3 | G − G−ind | makró AUROC, F1 | Mit ad hozzá a klinikai indikáció? |
| 4 | G − G-img | makró AUROC, F1 | Mit ad hozzá az esetek és az indikáció együtt, azonos modellel és prompttal? |
| 5 | G-img − S0 | makró AUROC | Általános LLM a specialista ellen, csak képpel |
| 6–8 | G − P1 / P2 / P3 | szenzitivitás, specificitás, F1 | A saját rendszer a kiinduló alkalmazások ellen |

**Miért előre rögzített lista, és miért csak egy fő összevetés?** Nyolc rendszer, kétféle ítélet és több mérőszám mellett több tucat páros összevetés lehetséges. 95%-os szinten ezek közül néhány pusztán a véletlen miatt is „szignifikánsnak” látszana, és utólag ki lehetne választani azt, amelyik a legjobban mutat. A lista ezért a tesztfuttatás előtt rögzült, és a szakdolgozati kérdésre egyetlen előre megnevezett válasz van (G − S0). A többi összevetés G felépítését bontja szét lépésenként, ezek leíró jellegűek; többszörös összehasonlítási korrekció nem történt, helyette csak az 1. számít fő eredménynek.

### 7.2 Mérőszámok és választásuk oka

**Az értékelés egysége** egy (vizsgálat, elváltozás) pár. Minden rendszer vizsgálatonként hat döntést hoz, a teszthalmazon tehát rendszerenként 400 × 6 = 2 400-at. Egy pár **pozitív**, ha a referenciacímke szerint az elváltozás jelen van, és **negatív**, ha nincs. A rendszer döntését a referenciacímkével összevetve négy kimenet lehetséges: valódi pozitív (VP), álpozitív (ÁP, téves riasztás), álnegatív (ÁN, kihagyott elváltozás) és valódi negatív (VN).

**Miért nem a találati arány (a helyes döntések aránya)?** A 2 400 párból csak 324 pozitív. Egy rendszer, amely mindenre nemmel felel, 2 076 helyes döntést hozna (86,5%), pedig egyetlen elváltozást sem ismerne fel. A találati arány tehát az óvatosságot jutalmazza, és nem mutatja, hogy a rendszer felismeri-e az elváltozásokat.

**Szenzitivitás és specificitás.**
- *Szenzitivitás* = VP / (VP + ÁN): a pozitív esetek közül hányat jelez a rendszer pozitívnak.
- *Specificitás* = VN / (VN + ÁP): a negatív esetek közül hányat ítél helyesen negatívnak.

Ezek az alapmérőszámok, mert külön mutatják a két hibatípust, a kihagyott elváltozást és a téves riasztást, amelyeknek klinikailag eltérő a súlya. A szenzitivitás csak a pozitív, a specificitás csak a negatív esetekből számolódik, ezért egyik sem függ az elváltozás gyakoriságától, és a dúsított teszthalmazon is ugyanazt jelentik, mint természetes gyakoriság mellett.

**F1-pontszám.** A pozitív prediktív érték (PPV = VP / (VP + ÁP), vagyis a pozitív döntések közül a helyesek aránya) és a szenzitivitás harmonikus közepe: F1 = 2 · PPV · szenzitivitás / (PPV + szenzitivitás). Akkor magas, ha mindkét tényező magas, ezért egyetlen számmal bünteti a „mindenre igen” és a „szinte semmire sem igen” viselkedést is; így két eltérő stílusú (óvatos, illetve sokat jelző) rendszer egy mérőszámon vethető össze. A PPV a gyakoriságtól függ, ezért az F1 csak a teszthalmazon belül hasonlítható össze, természetes gyakoriságú adattal nem.

**AUROC (a ROC-görbe alatti terület).** A pontszámot adó rendszerek két lépésben döntenek: először elváltozásonként pontszámot adnak, majd ebből egy döntési küszöb képez igen/nem ítéletet. A pontszám az S0-nál a modell kimeneti valószínűsége, az R0-nál a szomszédok közül az elváltozást mutatók aránya, G-nél és két változatánál (G−ind, G-img) a válaszban megadott valószínűség. Az AUROC az első lépést, magát a pontszámot értékeli, a küszöbtől függetlenül.

- *Definíció:* ha a küszöb a legmagasabb pontszámtól a legalacsonyabbig végigfut, minden küszöbhöz tartozik egy szenzitivitás–specificitás pár. Ezek görbéje a ROC-görbe (a szenzitivitás az 1 − specificitás függvényében), az AUROC pedig a görbe alatti terület. Ugyanez az érték így is megfogalmazható: egy véletlenül választott pozitív és egy véletlenül választott negatív esetből álló párban mekkora valószínűséggel kap a pozitív eset magasabb pontszámot (egyenlő pontszám fél találatnak számít). Hogy melyik eset pozitív és melyik negatív, azt a referenciacímke dönti el.
- *Jelentése:* azt méri, mennyire követi a pontszámok sorrendje a valóságot, vagyis mennyi információt hordoz a pontszám az elváltozás jelenlétéről. Ez a rendszer **diszkriminációs (megkülönböztető) képessége**. 0,5 esetén a pontszám semmit nem árul el (a sorrend véletlenszerű); 1,0 esetén minden pozitív eset magasabb pontszámot kap minden negatívnál, tehát létezik hibátlan küszöb. Csak a sorrend számít, a pontszámok nagysága nem, ezért eltérő skálájú rendszerek is összevethetők.
- *Amit nem mér:* nem mondja meg, hány ítélet helyes a ténylegesen alkalmazott küszöbnél. Két eset lehet helyes sorrendben (a pozitív kapja a magasabb pontszámot) úgy is, hogy mindkettő a küszöb fölé esik, és a küszöbnél az egyik téves riasztás. A jó AUROC tehát azt jelzi, hogy *létezik* jól működő küszöb, nem azt, hogy a választott küszöb jó; ezt a szenzitivitás és a specificitás mutatja. A kalibrációt sem méri, vagyis azt, hogy egy 0,8-as valószínűség valóban 80%-os gyakoriságnak felel-e meg.

**Miért az AUROC a fő mérőszám?** A rendszerek küszöbei különböző adatokon és különböző módon készültek (S0 és R0 a tanítóhalmazon, elváltozásonként; G a validációs halmazon, egyetlen közös küszöbbel). Ha két rendszer a saját küszöbénél kerülne összevetésre, a különbség legalább annyira tükrözné a küszöbválasztást, mint magukat a modelleket. Az AUROC sem a küszöbtől, sem a gyakoriságtól nem függ, így a két rendszer pontszámában rejlő információt veti össze. A kiinduló alkalmazások nem adnak pontszámot, ezért rájuk nem számolható.

**Makróátlag.** A mérőszám elváltozásonként készül, majd a hat érték súlyozatlan átlaga adja a makróátlagot. Ha az összes pár egyben szerepelne (mikroátlag), a gyakori tüdőárnyék uralná az eredményt, a ritka csonttörés és tüdőlézió alig számítana. Ennek ára, hogy egyetlen elváltozás is jelentősen elmozdíthatja az átlagot; ezért készült a tüdőlézió nélküli érzékenységvizsgálat (7.4).

**Különbség és konfidenciaintervallum (páros bootstrap).** Két rendszer összevetésekor a különbség ugyanannak a mérőszámnak a két rendszeren mért értéke közötti eltérés, ugyanazon a 400 tesztvizsgálaton: például a fő összevetésben G makró-AUROC-ja mínusz S0 makró-AUROC-ja, 0,754 − 0,705 = +0,049. Az eredmények táblázataiban szereplő **különbség** mindig ilyen „első rendszer − második rendszer” érték; pozitív előjel azt jelenti, hogy az első rendszer ért el magasabb értéket.

Ez a különbség egyetlen, 400 esetes mintán született, ezért a mintavétel miatt ingadozik: más 400 vizsgálaton kicsit más érték adódna. A 95%-os konfidenciaintervallum azt a tartományt adja meg, amelyben a különbség valódi értéke nagy valószínűséggel van. Erre nincs zárt képlet, ezért az intervallum bootstrap-módszerrel készült: a 400 vizsgálatból 2 000 visszatevéses, 400 elemű minta (rögzített véletlenmaggal, 2026), mindegyiken mindkét rendszer mérőszámának és különbségüknek az újraszámolása, és a 2 000 különbség középső 95%-a adja az intervallumot.

- **A mintavétel egysége a vizsgálat**, nem az egyes elváltozás, mert egy vizsgálat hat címkéje ugyanabból a képből és leletből származik, így nem függetlenek egymástól.
- **Páros:** minden újramintázásban mindkét rendszer értékelése ugyanazokon a vizsgálatokon történik, és a különbség is ezen a közös mintán készül. Így kiesik az a változékonyság, amely abból ered, hogy éppen könnyebb vagy nehezebb esetek kerültek a mintába (ez mindkét rendszert egyformán érinti), és az intervallum csak a két rendszer közötti eltérés bizonytalanságát tükrözi. Ez két, rendszerenként külön számolt intervallum átfedéséből nem olvasható ki.
- **Értelmezés:** ha a különbség intervalluma nem tartalmazza a nullát, a különbség valódinak tekinthető, nem a mintavétel véletlenjének. Ha tartalmazza, a különbség „nem mutatható ki”: ez nem jelenti azt, hogy a két rendszer egyforma, csak azt, hogy 400 esetből nem dönthető el.

### 7.3 Megkülönböztető képesség (AUROC) elváltozásonként

| Rendszer | Tüdőárnyék | Kardiomegália | Atelektázia | Tüdőlézió | Pleurális folyadék | Csonttörés | **Makró [95% CI]** |
|---|---|---|---|---|---|---|---|
| **G** | 0,709 | 0,861 | 0,717 | 0,786 | 0,839 | 0,611 | **0,754 [0,722; 0,784]** |
| G−ind | 0,679 | 0,867 | 0,707 | 0,744 | 0,848 | 0,561 | 0,734 [0,703; 0,763] |
| R0 | 0,715 | 0,873 | 0,707 | 0,657 | 0,874 | 0,530 | 0,726 [0,696; 0,755] |
| G-img | 0,731 | 0,770 | 0,668 | 0,753 | 0,838 | 0,531 | 0,715 [0,683; 0,747] |
| S0 | 0,740 | 0,794 | 0,709 | 0,593 | 0,853 | 0,543 | 0,705 [0,671; 0,737] |

Minden rendszer gyengébb a teszthalmazon, mint a validációson (pl. S0 0,757 → 0,705, G 0,770 → 0,754). A dúsított teszthalmazban jóval több a nehéz, ritka elváltozás (pl. csonttörés 38 a validációs halmaz 2 esetével szemben), és a makró átlagban minden elváltozás azonos súllyal számít.

### 7.4 A fő eredmény: G a specialista ellen

**G valószínűségei jobban elkülönítik a pozitív és a negatív eseteket, mint S0 pontszámai, vagyis G megkülönböztető képessége jobb: makró AUROC 0,754 vs. 0,705, különbség +0,049 [+0,013; +0,087].** Az intervallum nem tartalmazza a nullát, így az előre rögzített feltétel szerint G jobb. Ez a küszöbtől független osztályozási minőségre vonatkozik, nem arra, hogy G több helyes igen/nem választ adna; a kimondott ítéleteket a 7.6 rész veti össze.

A különbség két elváltozásból jön: **tüdőlézió +0,19 [+0,10; +0,29]** és kardiomegália +0,07 [+0,01; +0,12]. A tüdőárnyék, az atelektázia, a pleurális folyadék és a csonttörés esetén a két rendszer azonos szinten van.

**Érzékenységvizsgálat** (a tesztfuttatás előtt rögzített vállalás: ha a ritka, nehéz elváltozások elhagyása megváltoztat egy következtetést, a makróátlagot nélkülük is meg kell adni):

| Makró AUROC-különbség [95% CI] | Mind a hat | Tüdőlézió nélkül | Tüdőlézió és csonttörés nélkül |
|---|---|---|---|
| G − S0 | **+0,049 [+0,013; +0,087]** | +0,020 [−0,020; +0,063] | +0,008 [−0,028; +0,044] |
| G − G-img | +0,039 [+0,008; +0,071] | +0,040 [+0,005; +0,077] | +0,030 [−0,003; +0,065] |

**Tüdőlézió nélkül G és S0 nem különböztethető meg.** A fő eredmény nagyrészt a tüdőlézión múlik, azon az elváltozáson, amelyen az S0 minden halmazon a véletlen közelében teljesít. Ez a modell gyengesége, nem definíciós kérdés: az S0 tanítócímkéje lényegében ugyanazt tekinti tüdőléziónak, mint a referenciacímke (5.2). A gyengeség pontos oka nem ismert. Ezzel szemben az, amit az esetek és az indikáció G saját felépítésén belül hozzáadnak (G − G-img), tüdőlézió nélkül is megmarad. Az előre rögzített eredmény változatlanul érvényes; az érzékenységvizsgálat mellette szerepel, nem helyette.

### 7.5 Honnan jön G előnye?

| Összevetés | Makró AUROC-különbség [95% CI] | Értelmezés |
|---|---|---|
| G-img − S0 | +0,010 [−0,029; +0,050] | a Gemini csak a képből ≈ a specialista |
| G − G-img | **+0,039 [+0,008; +0,071]** | az esetek + indikáció egyértelmű javulást hoznak |
| G − G−ind | +0,020 [−0,008; +0,046] | az indikáció önálló hatása nem mutatható ki |
| G − R0 | +0,028 [−0,004; +0,060] | az LLM hozzáadott értéke a puszta visszakereséshez nem mutatható ki (határeset) |

A fő különbség felbontható: **0,010 + 0,039 = 0,049.** A Gemini a képből egyedül nagyjából annyit tud, mint a specialista; G előnye a visszakeresett esetekből és a klinikai kontextusból fakad, nem abból, hogy jobban olvasná a röntgenképet. Ezek másodlagos összevetések, ezért leíró jelleggel értelmezendők.

### 7.6 G és a kiinduló alkalmazások

A kiinduló alkalmazások csak ítéletet adnak, pontszámot nem, ezért itt az AUROC helyett a szenzitivitás, a specificitás és az F1 a közös mérce. Hogy mindkét oldalt ugyanaz a szabály pontozza, G is a saját kimondott ítéletével szerepel (csak a „present” számít igennek), nem a küszöbbel képzett változatával:

| Rendszer | Makró szenzitivitás | Makró specificitás | Makró F1 | G − rendszer, F1 [95% CI] |
|---|---|---|---|---|
| P1-adapted-ns | 0,52 | 0,86 | 0,42 | −0,022 [−0,078; +0,029] |
| P2-adapted | 0,42 | 0,90 | 0,39 | −0,000 [−0,049; +0,049] |
| P3-adapted | 0,43 | 0,75 | 0,30 | **+0,097 [+0,040; +0,149]** |
| **G (saját „present”)** | 0,32 | 0,95 | 0,39 | |
| G (valószínűség ≥ 0,08, tájékoztatásul) | 0,62 | 0,78 | 0,42 | |
| S0 (küszöbbel) | 0,72 | 0,58 | 0,33 | |
| R0 (küszöbbel) | 0,70 | 0,63 | 0,37 | |

- **G kimondott ítéletei óvatosabbak, nem pontosabbak.** G minden kiinduló alkalmazásnál ritkábban mond igent: szenzitivitása 0,09–0,20-szal alacsonyabb, specificitása 0,05–0,20-szal magasabb (minden intervallum kizárja a nullát).
- F1-ben a P1-gyel és a P2-vel egy szinten van, a P3-nál jobb; a P3 sosem látja a beteg képét, és túl sokat jelez.
- A csonttörést minden LLM-alapú rendszer kimondott ítélete elhibázza (szenzitivitás 0,00–0,18).
- G előnye nem az igen/nem válaszokban van, hanem a valószínűségeiben (AUROC) és az ellenőrizhető hivatkozásaiban.

Egy szemléltető eset (CXR2130: tüdőárnyék, kardiomegália és pleurális folyadék): G mind a hat elváltozást helyesen ítélte meg, a P1 ötöt, az R0 négyet, az S0, a P2 és a P3 hármat-hármat. A P2 itt a visszakeresett, normális szívméretű szomszédra támaszkodva tévedett; a P3 a szomszédok leleteiben szereplő elváltozásokat jelentette a betegre.

### 7.7 Hogyan használja G az eseteket?

- **Visszakeresés:** G 10 szomszédja 1,2–7,2-szer gyakrabban tartalmazza a beteg elváltozását, mint a véletlenszerű esetek (pleurális folyadék ×7,2, kardiomegália ×3,3, csonttörés ×1,2). Egyetlen szomszédnál (k = 1) a BiomedCLIP az atelektáziánál, a tüdőléziónál és a pleurális folyadéknál jobb esetet talál, mint a P3 Gemini-beágyazása, a P2 egyetlen találata pedig még gyengébb.
  *Miért ez a mérési mód?* A mérőszám elváltozásonként, csak azokon a tesztvizsgálatokon készül, amelyekben az elváltozás megvan: a visszakeresett esetek hányadában van meg szintén. Ha a teljes címkekészlet egyezése lenne a mérce, az eredményt a normális–normális párok uralnák, hiszen a vizsgálatok többsége normális. A nyers arány az elváltozás tanítóhalmazbeli gyakoriságával van elosztva (ezt adná a véletlen visszakeresés), így a kapott szorzó (×1 = véletlen) a ritka és a gyakori elváltozásoknál egyformán olvasható.
- **Másolás:** ahol a 10 szomszéd szavazata téves (2 400 párból 524), G kimondott ítélete az esetek 42,6%-ában egyezik vele, a szomszédokat sosem látó G-img-é 39,9%-ban. **A szomszédok kb. 3 százalékponttal húzzák G-t**, az atelektáziánál (0,64 vs. 0,54) és a kardiomegáliánál (0,30 vs. 0,23) erősebben. G tehát többnyire nem másol.
  *Miért ez a mérési mód?* A szomszédok „szavazata” az R0 szabálya k = 10-nél (a szomszédok arányának küszöbe, a tanítóhalmazon hangolva). Egyszerű többségi szavazás nem használható, mert ezeknél a ritka elváltozásoknál 10 szomszédból szinte soha nincs 6 pozitív, így a szavazat szinte mindig „nem” lenne. Önmagában az egyezés sem bizonyít másolást, hiszen G és a szomszédok egyszerűen ugyanazt a képet is láthatják hasonlónak. Ezért kell a G-img mint kontroll: ugyanaz a modell és prompt, de a szomszédokat nem látja. A két egyezési arány különbsége mutatja a szomszédok tényleges húzását, és a legárulkodóbb ott, ahol a szomszédok tévednek.
- **Hivatkozások:** G válogatva hivatkozik. A „present” ítéleteihez megjelölt esetek 81–96%-ában valóban megvan az adott elváltozás (tüdőárnyék, kardiomegália, atelektázia, pleurális folyadék), miközben az összes megkapott szomszédnak csak 24–43%-ában. A gyenge pont a tüdőlézió (43% vs. 7%); a csonttörés túl ritka a megítéléshez (6 ítélet). G 210 „present” ítéletéből 8 hivatkozás nélküli.
  *Miért kell az összehasonlító arány?* Egy önmagában álló szám (pl. „a hivatkozott esetek 40%-ában megvan az elváltozás”) nem értelmezhető: válogatott hivatkozást jelent, ha a 10 szomszédnak csak 10%-ában van meg, de semmilyen válogatást, ha 40%-ában. Az összes megkapott szomszéd aránya ezért a viszonyítási alap. A vizsgálat csak a „present” ítéletekre terjed ki, mert egy „absent” ítélethez hivatkozott normális eset mindig „helyes” lenne, így annak nincs értelmes ellenőrzése.
- **Kitalált hivatkozások a G-img-nél:** a G-img nem kap eseteket, de a prompt és a JSON-séma hivatkozást kér. 450 válaszából 2-ben soha meg nem kapott, de formailag helyes esetazonosítókat adott meg. A kód ellenőrzése kiszűrte őket. Tanulság: ha a kimenetben van hely bizonyítéknak, az kitalált bizonyítékot csalhat elő ott, ahol nincs bizonyíték.

### 7.8 Hibaelemzés, emlékezésvizsgálat, G indoklásainak ellenőrzése

Ezek az elemzések a teszthalmaz tárolt válaszain futottak, további API-hívás nélkül. A kézi ítéleteket egy MI-asszisztens hozta első körben, szöveg alapján (lelet, indikáció, válasz); radiológus nem ellenőrizte őket, és képet senki nem nézett.

**Hibaelemzés (23 rétegzett mintavétellel kiválasztott hiba):**

| Fő kategória | Darab |
|---|---|
| Kihagyott elváltozás | 6 |
| Szomszéd átmásolása | 6 |
| Kitalált elváltozás | 4 |
| Definíciós eltérés | 3 |
| Egyéb (küszöbhatás 2, félreosztályozott valódi eltérés 1, kitalált hivatkozás helyes ítélettel 1) | 4 |

Visszatérő minták:

- **régi és nem bordán lévő törések kimaradnak:** G indoklásai „akut” törést keresnek, holott a definíció a régit is ide sorolja;
- **a másolás látszik a szövegen:** ilyenkor az indoklás kimondja, hogy a hasonló esetek miatt döntött, vagy szinte szó szerint átvesz egy mondatot egy szomszéd leletéből;
- **téves kardiomegália** alacsony tüdőtérfogatú vagy AP-irányú felvételeken;
- **definíciós eltérések** a tüdőárnyék körül (csomók, terimék, hegesedés);
- **küszöbhatás:** a 0,08-as küszöb G gyakori „absent, 0,10” válaszát pozitívvá teszi.

**Emlékezésvizsgálat:** a Gemini valószínűleg látta az IU X-Ray-t a tanítása során. A kérdés az volt, hogy a rendszerek szövege jobban hasonlít-e a vizsgálat saját leletére, mint 20 ugyanilyen címkéjű másik leletre (szóhármasok egyezése; 384 vizsgálat). **Egyik rendszernél sincs nyoma szó szerinti felidézésnek:** minden intervallum tartalmazza a nullát, és a saját lelet csak a vizsgálatok 4,9–6,8%-ában előzi meg mind a 20 összehasonlító leletet, ami a véletlen 5%-ának felel meg. Ez a csak képet kapó G-img-re is igaz. A korlát: a vizsgálat a megfogalmazás felidézését zárja ki, a címkék felidézését nem.

*Miért ez a mérési mód?* A mérőszám az, hogy a saját lelet egymást követő szóhármasainak hányada fordul elő a rendszer szövegében. A szóhármasok azért alkalmasak, mert a felidézés a megfogalmazásban látszik; egy jelentésalapú hasonlóság a helyes képolvasást is „emlékezésnek” mérné. Az összehasonlító leletek ugyanazokat a címkéket viselik, mert az IU-leletek erősen sablonosak: két normális lelet véletlenül is sok közös fordulatot tartalmaz, így csak az azonos címkéjű leletekhez mért többlet utal emlékezésre. A leletből az INDICATION szakasz kimarad, mert G azt bemenetként megkapta. Ha a saját lelet nem kiemelt a többi 20 között, akkor véletlenül 21-ből 1 eséllyel (kb. 5%) lesz a legjobb – ez a viszonyítási szint.

**G indoklásainak ellenőrzése (15 válasz, 90 elváltozás):** a státusz 88 esetben egyezik a saját indoklással, 2 esetben nem egyértelmű, ellentmondás nincs. A 17 „present” ítélet indoklása mind a képből indul ki: 10 csak a képre hivatkozik, 4 a képre és a hivatkozott esetekre, 3 a képre és az indikációra. Olyan ítélet, amely csak az esetekre vagy csak az indikációra épül, nem volt. Az indoklás azonban önbevallás, nem feltétlenül tükrözi, hogyan született a döntés.

**Költség:** az API-költség összesen 36,00 USD volt a havi 45 USD-s keretből, ebből a tesztfuttatások 23,21 USD-t tettek ki. Az S0 és az R0 helyben, a projekt grafikus kártyáján futott.

---

## 8. Következtetések

**Q1 – Alkalmasság:** egyik kiinduló alkalmazás sem működött mellkasröntgenen eredeti formájában: megszűnt modellek, egy eltávolított eszközt előíró prompt (62%-os hibaarány) és egy gondolkodást is magában foglaló kimeneti korlát akadályozta őket. Átalakítva minden esetre válaszolnak, de ítéleteik durvák (makró szenzitivitás / specificitás: P1 0,52 / 0,86, P2 0,42 / 0,90, P3 0,43 / 0,75). A P3 sosem látja a beteg képét, és valójában a szomszédairól nyilatkozik.

**Q2 – Erősségek és hiányok:** érdemes átvenni a P1 rendszerezett radiológiai olvasását, a P2 visszakereső mintázatát és a P3 hivatkozási objektumait. Mindegyikből hiányzik a klinikai szöveg mint bemenet, a beteg képének és a szöveges bizonyítéknak az együttes átadása, az elváltozásonkénti pontszám, a referenciacímkékhez igazított definíció és a géppel ellenőrizhető hivatkozás. A G rendszer ezeket pótolja.

**Q3 – Összevetés:**

- Az előre rögzített fő összevetésben **G valószínűségei jobban megkülönböztetik a pozitív és a negatív eseteket, mint a tervezett zero-shot specialista pontszámai** (+0,049 makró AUROC), de ez **nagyrészt a tüdőléziónak köszönhető**, amelyen a specialista a véletlen közelében teljesít; enélkül a különbség nem mutatható ki. A specialista lényegében ugyanazzal a tüdőlézió-definícióval tanult, mint amellyel az értékelés történt; gyengeségének pontos oka nem ismert.
- A legszilárdabb eredmény a G felépítésén belüli: a visszakeresett esetek és a klinikai indikáció együtt mérsékelten javították az LLM megkülönböztető képességét a csak képes működéshez képest (+0,039 makró AUROC), és ez tüdőlézió nélkül is megmaradt; a két tényező külön hatása nem választható szét. Csak a képből a Gemini és a specialista között nem mutatható ki különbség.
- A kiinduló alkalmazásokkal szemben G kimondott ítéletei óvatosabbak, nem pontosabbak. Valódi többlete a jól megkülönböztető valószínűség és az ellenőrizhető hivatkozás: a „present” ítéletekhez megjelölt esetek 81–96%-ában megvan az adott elváltozás.
- A szomszédok csak kis mértékben húzzák G ítéleteit (kb. 3 százalékpont). Ha G mégis másol, az látszik az indoklásán.

**A fő állítás korlátai:**

- az S0 a terv szerinti DenseNet, nem a legerősebb elérhető specialista (egy ResNet-50 a tanítóhalmazon 0,808-at ért el a DenseNet 0,728-ával szemben, a teszten nem futott); az S0 ráadásul zero-shot, IU-adaton finomhangolt képmodell nem szerepelt;
- a Gemini valószínűleg látta az IU X-Ray-t; szó szerinti felidézés nem mutatható ki, a címkék felidézése viszont nem zárható ki;
- G olyan elváltozás-definíciókat kap, amelyeket a többi rendszer nem;
- egyetlen kórházi gyűjtemény és egyetlen 400 esetes tesztminta, a ritka elváltozásoknál széles intervallumokkal; külső validálás nem volt;
- a kézi ítéletek egy MI-asszisztens első körös, szöveg alapú értékelései; radiológus nem nézte át őket.

**Összegezve:** az eredmény nem azt mondja, hogy „a multimodális visszakeresés legyőzi a képmodelleket”, hanem azt, hogy **egy bizonyítékra épülő kép + szöveg rendszer ezen a gyűjteményen felülmúlta a tervezett zero-shot specialistát, főként egy olyan elváltozáson, amelyen a specialista a véletlen szintjén teljesít**, és hogy **a visszakeresett esetek és a klinikai kontextus mérhetően javítják az LLM képértelmezését a csak képes működéshez képest.**

**Továbblépési lehetőségek (szakdolgozat):**

- erősebb vagy finomhangolt specialista (pl. a ResNet-50) bevonása, illetve G és a specialista kombinálása;
- a valószínűségek kalibrálása és elváltozásonkénti küszöbök;
- a csonttörés-definíció pontosítása a promptban;
- a nagy k-s szomszédszavazás jelének átadása G-nek;
- mechanizmusvizsgálatok: üres kép, kevert szöveg, szándékosan rossz címkéjű visszakeresés;
- helyben futó modellel, hitelesített adaton (MIMIC-CXR) végzett megismétlés.
