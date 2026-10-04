# robotek-2 — robot 20 cm, ciągły ruch i kamera 3D

**robotek-2** to wariant z wyborem terenów generatora, modelami osobnymi dla
gałęzi Git i małą, stałą karą za kolizje −0,01. Nie ma kary za stanie w miejscu.

Python 3.10+, Tkinter, NumPy i PyTorch.

```powershell
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -t . -v
```

## Katalogi i ścieżki

Konfiguracja: `config/levels.json` i `config/bptt_settings.json`.
`config/config.py` wyznacza ścieżki względem projektu, niezależnie od katalogu
uruchomienia. GUI i trening używają `models/<branch>/` dla modelu, `.latest.pt`,
walidacji, checkpointów i historii sesji. Nowa gałąź bez modelu zaczyna nową
sieć; nie wczytuje modelu innej gałęzi. Konsolowe `--model` nadpisuje ten wybór.
Nazwy z `/` są kodowane, np. `feature/plain` → `models/feature%2Fplain/`.
Detached HEAD używa `models/detached-<commit>/`. Po zmianie brancha uruchom
aplikację ponownie; działająca sesja zachowuje katalog wybrany przy starcie.

Testy uruchamiaj z katalogu projektu jako pakiet `tests`, np.
`python -m unittest tests.test_paths -v`.

### Zmiany po przeniesieniu katalogów — 4 października 2026

- Przeniesione konfiguracje są czytane i zapisywane w `config/`; projektant
  poziomów i okno ustawień BPTT korzystają z nowych ścieżek.
- GUI i trening mają jedną wspólną ścieżkę modelu z `config/config.py`.
  Model, zapis najnowszy, checkpointy, wyniki walidacji i historie sesji
  pozostają w katalogu konkretnej gałęzi `models/<branch>/`.
- Odczyt gałęzi Git jest wykonywany w katalogu repozytorium, więc uruchomienie
  programu z innego katalogu nie zmienia wyboru modelu. Jeśli nie można ustalić
  gałęzi, program zgłasza błąd zamiast wybrać wspólny katalog modeli.
- Dodano pliki pakietów `config/__init__.py` i `tests/__init__.py`, testy
  ścieżek w `tests/test_paths.py` oraz poprawiono odwołania w dokumentacji.

Weryfikacja po przeniesieniu ścieżek: 51 testów ścieżek, kolizji, podglądu,
walidacji i batcha przeszło. Po kolejnych zmianach generatora i uporządkowaniu
kar pełny zestaw zawiera **125 testów — wszystkie przechodzą**.

Po utworzeniu i przełączeniu nowego brancha zamknij poprzednią sesję aplikacji
i uruchom ją ponownie. Przed treningiem możesz sprawdzić wybrane ścieżki:

```powershell
python -c "from config.config import MODEL_VERSION, MODEL_PATH, PLAN_PATH, SETTINGS_PATH; print(MODEL_VERSION); print(MODEL_PATH); print(PLAN_PATH); print(SETTINGS_PATH)"
python -m unittest tests.test_paths -v
python main.py
```

`MODEL_PATH` powinien wskazywać katalog nowej gałęzi. Jeśli nie ma w nim modelu,
pierwszy trening utworzy nową sieć; zapis w `models/main/` pozostanie osobny.
Nie kopiuj modelu z poprzedniej gałęzi, jeśli celem jest eksperyment od zera.

## Trenuj i oglądaj

### Pola dostępne dla losowych map

W projektancie pole `pola` wybiera tereny generatora. Wymagane są `S`, `E`
i `#` (ściana). Domyślne `SE#.` oznacza START, END, ściany i zwykłe podłoże.
Przykłady: `SE#.P` dodaje bruk, `SE#.PBWF` dopuszcza wszystkie tereny.
Samo `SE#` również używa zwykłego podłoża jako wypełnienia korytarzy.
Jeśli podasz tereny podłoża, losowane są tylko te wymienione w polu.

Wybrana paleta działa w podglądzie poziomu, treningu i walidacji, a zapis
modelu oraz podsumowanie walidacji zachowują `pola`. Generator buduje połączony
labirynt, po czym losuje tereny jednakowo na całej mapie — **nie chroni trasy
do END przed zagrożeniami**. Wagi dla wybranych terenów: ziemia 55, bruk 15,
bagno 15, woda 8, ogień 7; wagi są przeliczane względem dostępnej palety.
**Zawsze istnieje co najmniej jedno połączenie S → E przez pola inne niż
ściany.** Generator DFS łączy komórki korytarzami, a END wybiera spośród pól
osiągalnych od START. Losowanie terenów zmienia rodzaj podłoża, ale nie
zamienia istniejących korytarzy w ściany. Korytarze mają szerokość co najmniej
jednego pola (1 m), więc robot o średnicy 20 cm mieści się na trasie po ich
środkach.

Ta gwarancja dotyczy geometrii losowej mapy. Przy wybranych zagrożeniach
nie gwarantuje przeżycia ani ukończenia w limicie czasu, a wyuczona polityka
może nie znaleźć trasy. Szczególnie duży krok lub ograniczony obrót również
mogą utrudniać wykonanie geometrycznie istniejącej trasy. Dla map ręcznych
walidacja planu sprawdza połączenie S → E przez pola przechodnie.

Dawne bezpośrednie wywołanie `generate_maze()` bez argumentu `pola` zachowuje
stary generator dla zgodności; trening i walidacja przekazują paletę jawnie.
W istniejących konfiguracjach jawne `SE` należy zmienić na `SE#.`.
Plan bez klucza `pola` dostaje nową domyślną wartość.

Testy: `python -m unittest tests.test_generator_fields tests.test_map_fields -v`.

Przycisk **Trenuj i oglądaj** pokazuje cały batch robotów na wspólnej mapie.
Kolory rozróżniają boty; biały obrys i numer oznaczają wybranego robota.
Kliknięcie bota lub wybór numeru nad kamerą przełącza podgląd 3D.
Szare roboty zakończyły przebieg, czerwone zostały ubite łapką.
Tempo podglądu steruje wspólnymi krokami batcha; MAX odświeża podgląd
bez celowego spowalniania treningu.

**Walidacja** podczas treningu dodaje osobną ocenę do kolejki. Oceny wykonują
się kolejno między grupami; kliknięcia podczas oceny dodają następne oceny.
Co 1000 globalnych grup nadal wykonywana jest automatyczna walidacja na
100 stałych mapach, bez eksploracji i aktualizacji wag. Każda ukończona ocena
uzupełnia tabelę i wykres skuteczności Q (ostatnie 40 ocen; pełny log w tabeli).
Wyniki poszczególnych map trafiają do `models/<branch>/agent_continuous.validation.csv`,
a podsumowania do `models/<branch>/agent_continuous.validation.jsonl`; GUI wczytuje
historię po ponownym uruchomieniu. Stop przerywa ocenę i usuwa oczekujące żądania.

**Q oznacza jakość bota: procent map walidacyjnych zakończonych sukcesem**
(`wygrane / liczba map × 100%`). **Q aktualne** pochodzi z ostatniej ukończonej
walidacji, ręcznej lub automatycznej. **Q średnie** to średnia skuteczności
z ostatnich czterech automatycznych walidacji co 1000 grup. Jeśli dostępnych
jest mniej ocen, GUI uśrednia dostępne i pokazuje ich liczbę (np. 2/4).
Ręczne walidacje uzupełniają Q aktualne, wykres i log, ale nie Q średnie.
Starsze logi też dostarczają Q na podstawie liczby wygranych i map.
Wartości funkcji Q sieci DQN nie są wyświetlane pod tym oznaczeniem.

Włącz **Łapka −20** i kliknij zapętlonego bota. Kara kończy jego przebieg,
odejmuje dokładnie 20 od wyniku i nagrody oraz oznacza ostatnie przejście
jako terminalne w historii BPTT. Bot nie dostaje dodatkowej kary za zwykłą
śmierć. Pozostałe boty trenują dalej. Kara dotyczy wyłącznie żywych przebiegów
w bieżącej grupie; przy nakładających się botach preferowany jest wybrany bot.
Przy START kara czeka na pierwszy ruch, aby istniało przejście do uczenia.
Przebieg ukaranego bota podlega dotychczasowym zasadom doboru do BPTT.

Testy nowych funkcji: `python -m unittest tests.test_training_dashboard tests.test_live_gui tests.test_validation tests.test_batching -q`.

## Zatrzymanie eksperymentu — 4 października 2026

Decyzja prowadzącego: **zatrzymać dalszy trening obecnego wariantu** ze względu
na błąd założenia generatora. Generator zawsze chroni bezpieczną drogę
START → END, a zagrożenia umieszcza poza nią. To daje botom wskazówkę:
odnogi z zagrożeniami można odrzucać zamiast uczyć się ogólnej nawigacji.

Konkluzja z obserwacji prowadzącego: boty nauczyły się przede wszystkim
eliminować odnogi z zagrożeniami. Jeśli poprawne przejście leży obok zagrożenia,
mogą również je odrzucać i powtarzać nieskuteczne akcje. Dalszy trening na tym
rozkładzie map grozi utrwaleniem tej niepożądanej strategii. To interpretacja
obserwowanego zachowania; dotychczasowe testy nie analizowały decyzji sieci.

Walidacja korzysta z tego samego generatora, więc wysokie Q na jej mapach
nie wystarcza do wykazania ogólnej umiejętności przechodzenia labiryntów.
Stałe, osobne seedy chronią przed oceną na mapach treningowych, ale nie usuwają
wspólnego błędu założenia obu zbiorów.

Audyt kolizji opisany w `tests/COLLISION_AUDIT.md` sprawdzał mechanikę ruchu,
cofania i obrys robota. Jego pozytywny wynik nie wyklucza zapętlenia strategii
wyuczonej przez sieć ani błędu projektu środowiska treningowego.

**Plan kolejnego eksperymentu na nowym branchu:** generator map bez
zagrożeń i specjalnych terenów, z polami START, END i zwykłym podłożem.
Pierwszym celem będzie sprawdzenie nauki nawigacji bez wskazówki wynikającej
z rozmieszczenia zagrożeń. Wprowadzono wybór palety opisany w sekcji
„Pola dostępne dla losowych map”; poniższy opis dotyczy wcześniejszego eksperymentu.

## Dotychczasowy wariant treningu

Obecnie każda wykonana próba ruchu zablokowana przez ścianę lub granicę mapy
daje małą, stałą karę **−0,01** (`COLLISION_COST` w `episode.py`). Kara nie rośnie
z serią kolizji. Obrót i udany ruch, także cofanie, nie dostają tej kary.
Akcja niewykonana z powodu przekroczenia limitu czasu nie nalicza kolizji.
Koszt czasu nalicza się osobno. Kara wpływa na nagrodę do uczenia, a punktowy
wynik oceny zachowuje dotychczasową formułę.

Kary za przebywanie w miejscu usunięto wraz z ich nieużywanymi obliczeniami
i testami. Nie ma dawnej narastającej kary za kolizje. W poprzednim eksperymencie
agresywne kary według obserwacji utrudniały naukę; obecne −0,01 jest świadomie
małym sygnałem odróżniającym uderzenie w ścianę od poprawnego ruchu.

Przy zatrzymaniu eksperymentu `config/levels.json` zawiera jeden losowy poziom 11×11:
krok 0,1 m, obrót 15°, limit 300 s i losowy kierunek startowy.
Próg 9999 wygranych grup z rzędu
celowo utrzymuje naukę na tym poziomie. Każda grupa dostaje nową mapę;
dalszy rozwój ma polegać na zwiększaniu szczegółowości ruchu, bez powiększania map.
Zmiana kroku lub kąta może chwilowo pogorszyć wyniki; porównuj wyniki walidacji
w obrębie tej samej konfiguracji zapisanej w CSV.

Zapisany dobór BPTT w `config/bptt_settings.json` to podział rankingu 10/20/70%
i losowanie 2/3/2 pełnych przebiegów. GUI pozwala zmienić te wartości.
Jedna zakończona grupa oznacza jeden krok optymalizatora, niezależnie od batcha
i liczby przebiegów wybranych do BPTT. W poprzednim eksperymencie logiczne
zachowanie GRU zaobserwowano około 16 tys. aktualizacji; nie jest to gwarantowany próg.

Testy kolizji sprawdzają stałą karę, jej sumowanie, cofanie, obrót i brak kary
za niewykonaną akcję po limicie czasu. Dawne oczekiwania usuniętych kar zostały
uporządkowane. Testy kamery porównują obrazy w pamięci i nie zapisują PNG.

## Przestrzeń i sterowanie

Każde pole mapy ma **1×1 m**. Robot jest kołem o średnicy **20 cm**.
Współrzędne `(x,y)` oznaczają środek robota w metrach, licząc od lewego
górnego rogu mapy. Start na polu `(i,j)` to `(i+0.5,j+0.5)` — 50 cm
od jego krawędzi. Kamera znajduje się w środku robota.

- **W / ↑**: przód o 10 cm.
- **S / ↓**: tył o 10 cm, bez zmiany kierunku patrzenia.
- **A / ←**: obrót w lewo o 10°.
- **D / →**: obrót w prawo o 10°.
- **R**: restart tej samej mapy; **Esc**: zamknięcie.

Kąt 0° oznacza prawo, 90° dół; startowy kąt wynosi 0°.
Po obrocie ruch odbywa się po rzeczywistym kierunku, bez przyciągania do siatki.
Kolizja obejmuje całą tarczę i całą drogę jej środka, również przy narożnikach.
Dotknięcie styczne jest dozwolone; nakładanie na ścianę lub wyjście poza mapę
blokuje cały krok, nalicza jego czas oraz stałą karę −0,01 za kolizję.
Kolizje i ich serie są nadal liczone; udany ruch (także cofnięcie) zeruje serię,
a obrót jej nie zeruje. Robot nie ślizga się po ścianie.
Obrót koła nie zmienia obrysu i nie powoduje kolizji.

Cofanie nie ma dodatkowej kary. Nie ma kar za ponowne odwiedziny lub powrót.
Koszt czasu i obrażeń jest taki sam w obu kierunkach.
END jest osiągnięty, gdy **środek** żywego robota wchodzi na pole E.

## Czas i teren

Podstawowa prędkość wynosi 1 m na jednostkę czasu: 10 cm zajmuje 0.1.
Obrót o 10° kosztuje `0.5/9` jednostki na zwykłym podłożu.
Prędkość zależy od terenu; krok przecinający granicę pól uwzględnia długość
każdej części. Obrót i zablokowany krok nie naliczają efektów terenu.
Domyślny limit próby wynosi 200 jednostek czasu; poziom może ustawić własny limit.
Akcja przekraczająca limit nie jest
wykonywana; osiągnięcie END dokładnie w limicie jest dozwolone.

| Teren | Symbol | Mnożnik prędkości | Efekt przebytej drogi |
|---|---|---|---|
| Ściana | # | — | Blokada obrysu i kamery |
| Ziemia / START / END | . / S / E | 1 | Zwykła powierzchnia |
| Bruk | P | 1.2 | Szybszy ruch |
| Bagno | B | 0.5 | Wolniejszy ruch |
| Woda | W | 0.8 | 10 obrażeń/m, gasi ogień; śmierć po 3 m bez opuszczania wody |
| Ogień | F | 1 | 25 obrażeń/m; śmierć po 2 m bez opuszczania ognia |

Po opuszczeniu ognia robot otrzymuje 10 obrażeń/m i ginie po 3 m drogi
z podpaleniem, chyba że wejdzie do wody. Efekty powierzchni zależą od drogi
środka robota. Liczniki kontaktu z wodą/ogniem zerują się po opuszczeniu terenu.
Wygrana wymaga przeżycia ruchu. Obrót nie zwiększa tych liczników.

Nagroda jest sumą: +10 za END, −3 za śmierć, −1.5 za limit czasu,
−0.005 za jednostkę czasu, −0.02 za punkt obrażeń,
+0.2 za pierwsze wejście środka na nowe pole 1×1 m (maksymalnie +5 na próbę),
**−0.01 za kolizję**, a przy użyciu łapki −20 i zakończenie przebiegu.
Nie ma dodatkowej kary za przebywanie w polu ani powrót do niego.
Premia nie jest wypłacana za każdy krok 10 cm. Nie ma nagrody za odległość
od END ani premii za odkrywanie starego FOV.

GUI pokazuje osobno sumę nagród treningowych i premię eksploracji.
GUI pokazuje również skumulowaną karę za kolizje.
Punktowy wynik oceny jest liczony osobno od nagrody treningowej.

Wynik oceny: `1000*wygrana − czas − 2*obrażenia − 250*śmierć − 100*limit`.

## Kamera i sieć

Kamera Wolf3D generuje **RGB 80×60**, kąt **120°**, głębokość widoku 6 m.
GUI pokazuje te same piksele w powiększeniu do 320×240. Ściany zasłaniają
teren. Fioletowa szachownica oznacza END. Obraz zmienia się przy każdym
przesunięciu 10 cm i obrocie 10°.

Sieć: **CNN 32/64/64 → 256 cech → GRU 1024 → 512 → 512 → 256 → 128 → 4 Q**.
Akcje: przód, lewo, prawo, tył. GRU dostaje także poprzednią wykonaną akcję
lub znacznik START. Nie dostaje pozycji, mapy, odwiedzin ani kierunku do celu.
Pamięć GRU zeruje się przed każdym epizodem, również w ocenie i grze AI.

## Batch na jednej mapie i wybór najlepszego życia

Pole **Roboty na mapę (batch)** wybiera 1–128 robotów; domyślnie 5.
Pole **Grupy / mapy** określa liczbę kolejnych grup: 300 grup przy batchu 5
oznacza 1500 przebiegów na 300 mapach, nie 300 pojedynczych robotów.

1. Generujemy jedną mapę i N robotów ze wspólnymi, zamrożonymi na czas grupy wagami.
2. Każdy robot ma osobny seed eksploracji, poprzednią akcję oraz zerowy początkowy stan GRU.
3. Jeden forward CNN/GRU przetwarza aktywne roboty w rzeczywistym wymiarze batch.
   Ukończone roboty są usuwane z aktywnego batcha. Fizyka pozostaje na CPU;
   kamery powstają tensorowo na urządzeniu sieci (CUDA lub CPU).
4. Po zakończeniu wszystkich przebiegów sortujemy roboty: najpierw wygrana,
   następnie najwyższa suma nagród; wynik punktowy i mniej kroków rozstrzygają remisy.
5. Domyślnie uczymy na najlepszym przebiegu. Opcja **Dobór do BPTT…** pozwala
   włączyć losowanie z trzech części rankingu: najlepszych, średnich i pozostałych.
   Ustaw trzy procenty (suma 100%) oraz trzy liczby przebiegów do wylosowania.
6. Pełne BPTT każdego wybranego życia zaczyna się od własnego zerowego stanu GRU.
   Strata jest uśredniana po krokach danego życia, następnie po wybranych życiach.
   Gradienty są akumulowane, przycinane raz, a wspólny optymalizator wykonuje
   **jeden krok na całą grupę**. Nie sklejamy historii ani stanów pamięci robotów.

Przykład: batch 128, podział 10/20/70 daje 13/26/89 robotów (największe reszty,
przy remisie pierwszeństwo ma wcześniejsza grupa). Losowanie 3/2/3 wybiera osiem
różnych pełnych żyć, po jednym głosie na życie: udziały grup w stracie to
37,5%/25%/37,5%. Długa przegrana próba nie dominuje tylko dlatego, że ma więcej kroków.
Losowanie jest odtwarzalne, bez powtórzeń. Nie można wybrać więcej robotów, niż
mieści grupa; komunikat prosi o zmianę batcha lub ustawień. Zero oznacza pominięcie
grupy w BPTT. Pierwsza grupa to czołówka rankingu, nie gwarancja wygranej.

Ustawienia zapisują się w `config/bptt_settings.json`; konfiguracja sesji trafia także do
modelu i raportu JSON. CSV przebiegów ma `rank_group` i `selected`, a CSV grup
`bptt_count`. W trybie domyślnym `rank_group` jest puste. W konsoli:
`python train.py --batch-size 128 --split 10 20 70 --samples 3 2 3`.
CLI bez tych opcji używa najlepszego robota; GUI odczytuje zapisane ustawienia.

To wybór z przebiegów treningowych. Osobna walidacja odbywa się co 1000 grup.
Licznik zaliczeń nadal zależy od najlepszego robota całej grupy, niezależnie od
tego, czy został wylosowany do BPTT. Każde wybrane życie wnosi własne nagrody
oraz kary; sprzeczne gradienty mogą częściowo się znosić we wspólnej sieci.
Jedna grupa to jedna wygrana lub przegrana, niezależnie od wielkości batcha.
Licznik jest stale widoczny i zapisywany z modelem. Pełne wyniki robotów pozostają w CSV.
Podgląd pokazuje pierwszego aktywnego robota, po jego zakończeniu przechodzi do
kolejnego. Nagłówek podaje jego numer, rozmiar batcha i liczbę aktywnych robotów.
Tempo „kroków/s” to suma ruchów wszystkich robotów, z uwzględnieniem opóźnienia
podglądu, ale bez późniejszego BPTT; czas BPTT jest pokazywany osobno.

Full-episode BPTT nie odcina gradientu GRU. CNN używa checkpointingu aktywacji
w porcjach po 64 obrazy — to oszczędza pamięć, ale nie skraca życia użytego do nauki.
Historia kandydatów przechowuje tylko pozycję środka i wektor kierunku kamery
oraz akcje, nagrody i zakończenia. Nie przechowuje tysięcy obrazów w RAM ani VRAM.
Mapa i paleta są przesyłane na urządzenie raz na grupę. Każdy krok renderuje
aktywny batch na urządzeniu sieci, a tensor RGB uint8 trafia bezpośrednio do CNN,
bez pobierania obrazu na CPU i ponownego wysyłania na GPU.
Po wyborze zwycięzcy jego obrazy są deterministycznie odtwarzane na GPU z historii
pozycji, porcjami po maksymalnie 64 kamery, i używane do pełnego BPTT.
Fizyka, decyzja zakończenia i wybór losowych akcji nadal są obsługiwane na CPU.
Nie losujemy dodatkowych epizodów z dawnego replay w tym trybie treningu.

### Kamera tensorowa i podgląd

`tensor_camera.TensorCamera` to deterministyczne operacje PyTorch na podłodze,
promieniach i prostokątach ścian. Nie jest uczoną siecią generującą obrazy.
CNN nadal dostaje wyłącznie piksele, nie mapę ani pozycję robota. Architektura
użytkownika (większe CNN, GRU i głowica) pozostaje bez zmian.
Referencyjny renderer NumPy w `camera.py` jest zachowany do testów zgodności;
nie jest wywoływany w pętli batchowego treningu.
Porównania dopuszczają niewielkie różnice na krawędziach tekstur i narożnikach
wynikające z arytmetyki CPU/GPU i sposobu rozstrzygania trafień na granicach.

Dodatkowy obraz Tkinter powstaje tylko gdy zaznaczone jest **Pokaż kamerę**
i użytkownik ogląda grę lub trening. Wyłączenie oglądania nie generuje obrazu
dla GUI. Wyłączenie samej kamery zachowuje stały rozmiar jej panelu.
Podgląd używa tego samego renderera tensorowego na CPU; nie pobiera wszystkich
kamer z GPU. Dla CPU wejście sieci również jest generowane tensorowo.

**Stop** przerywa grupę bez uczenia na niepełnych kandydatach i zapisuje model.
Jeśli Stop nastąpi już podczas BPTT, dokańcza się bieżąca aktualizacja i zapis.
Przy braku RAM/VRAM trening kończy się komunikatem i zachowuje ostatni zapis
na dysku; niezapisane grupy trzeba powtórzyć po zmniejszeniu batcha.
Wagi, optymalizator, poziom map oraz liczniki grup i przebiegów są zapisywane.
Bieżące obrazy i stany GRU nie są zapisywane. Obecne wagi v6 można kontynuować.
Double DQN, gamma 0.99, Adam 0.0002, Huber, obcięcie gradientu do 10,
aktualizacja sieci docelowej 1% po kroku optymalizatora.

### Obciążenie sprzętu

Panel pod kamerą odświeża około raz na sekundę CPU systemu i aplikacji,
RAM systemu i aplikacji, obciążenie GPU 0 oraz zajęty/całkowity VRAM.
CPU aplikacji jest przeliczony względem wszystkich logicznych procesorów.
VRAM/GPU dotyczą całej karty, również innych procesów. To pomiary chwilowe,
a nie gwarancja zmieszczenia kolejnego batcha. Brak odczytu jest oznaczony,
nie wyświetlany jako zero. NVIDIA jest odczytywana przez `nvidia-smi`, CPU/RAM
przez `psutil`, w osobnym wątku bez blokowania GUI.

Urządzenie **auto / cuda / cpu** wybiera miejsce obliczeń sieci, domyślnie auto
(CUDA, jeśli dostępna). Po Stop można zmienić batch i wznowić istniejące wagi.
Porównuj kroki/s przy MAX, VRAM/RAM i czas BPTT. Duży batch nie musi przyspieszać
symulacji, bo fizyka i sterowanie nadal mają część szeregową na CPU.

## Trening i mapy

**Trenuj dalej** / **Trenuj i oglądaj** pozwalają wybrać do miliona prób.
Suwak określa tempo podglądu; **MAX** usuwa sztuczne opóźnienia.
Podgląd pomija klatki, jeśli trening działa szybciej niż odświeżanie.
**Ukryj podgląd** kontynuuje trening bez rysowania. **Stop** kończy pracę
z zapisem. Między próbami nie ma pauzy na wygraną/przegraną;
uczenie i zapis nadal wymagają czasu.

Mapy zaczynają od 5×5. Po 20/20 wygranych na bieżącym poziomie następuje
awans: 7×7, 9×9, 11×11, 13×13, 15×15. Do tego liczymy 20 kolejnych grup na bieżącym poziomie: po jednym wyniku
wybranego robota. Po przejściu ze starego licznika indywidualnych robotów
okno awansu jest zerowane, ale rozmiar mapy i wyuczone wagi pozostają. Po awansie okno wyników poziomu się
zeruje. Wznowienie przywraca poziom i jego ostatnie wyniki.
Generator zapewnia trasę po ziemi i bruku, wykonalną przez ruchy 10 cm
po środkach korytarzy i obroty 10°, w limicie czasu.

Co **1000 grup / aktualizacji treningowych** (według globalnego licznika, również
po wznowieniu) trening zapisuje osobny checkpoint i ocenia jednego robota na
**100 stałych mapach randomowych 11×11**. Seedy są ujemne i odrębne od treningowych;
inny seed nie gwarantuje braku podobnych układów. Stały zestaw pozwala porównywać wyniki.
Ocena działa bez eksploracji, w trybie eval/inference, bez BPTT i aktualizacji wag.
GRU jest zerowane dla każdej mapy; stan pamięci i RNG treningu są przywracane po ocenie.
Krok (m), kąt obrotu (°), limit czasu i kierunek startowy pochodzą z ocenianego poziomu.
Przycisk **Walidacja** ocenia najnowszy zapis na tym samym zestawie.
Działa w tle; Stop go przerywa. Niepełna ocena nie trafia do CSV.
Po zakończeniu lub błędzie pracy przyciski ponownie stają się aktywne.
`--evaluate` uruchamia ręcznie tę samą ocenę na 100 mapach 11×11.
`--width` / `--height` nie zmieniają tego zestawu oceny.
Plik `models/<branch>/agent_continuous.validation.csv` jest dopisywany przy kolejnych
ocenach i sesjach. Każda mapa ma osobny wiersz: seed, wynik, wygrana, czas, obrażenia,
kroki, nagroda i przyczyna zakończenia, licznik grup/aktualizacji/przebiegów,
krok robota, kąt obrotu, limit czasu, kierunek startowy, checkpoint oraz skuteczność zestawu.
Checkpointy co 1000 grup są w `models/<branch>/checkpoints/`, z numerem grupy i identyfikatorem
sesji w nazwie. Zawierają wagi, optymalizator, plan i postęp oraz konfigurację treningu
i oceny. Dotychczasowy zapis latest co 10 grup oraz po Stop pozostaje aktywny.
Nie ma automatycznego wyboru najlepszego modelu; AI i wznowienie używają
najnowszego zapisu, a przy jego braku pliku bazowego.
Eksploracja: `0.1 + 0.6*exp(-(numer_próby-1)/250)`.

```powershell
python train.py --episodes 300 --batch-size 5
python train.py --episodes 300 --batch-size 5 --device cuda
python train.py --evaluate
```

GUI treningu i CLI domyślnie używają auto (CUDA, jeżeli dostępna). `--device cpu|cuda|auto` wybiera
urządzenie sieci. Kamera wejściowa działa na urządzeniu sieci, symulacja na CPU.
`--fresh` zaczyna od nowych wag; `--model` wybiera osobny plik.

Model **v6** zapisuje geometrię, kamerę i listę czterech akcji.
Trening zapisuje **models/<branch>/agent_continuous.latest.pt** (najnowszy).
**models/<branch>/agent_continuous.pt** jest opcjonalnym wcześniejszym zapisem bazowym. Raport JSON obok modelu,
historie CSV i raporty sesji w `models/<branch>/runs/`. Główny CSV zawiera wybranego
robota każdej grupy; plik `.rollouts.csv` zawiera wszystkie przebiegi, seedy
i oznaczenie wybranego robota.
Dawne wytrenowane pliki są zachowane, ale nie są automatycznie wczytywane:
mają inne wejście GRU, trzy akcje i inne znaczenie ruchu. Nowy wariant
zaczyna naukę od nowych wag. Wagi ciągłego modelu v6 są zgodne z nowym batchem.
Zapis następuje co 10 grup, na końcu i po Stop.

Kod: `player.py` — geometria; `episode.py` — akcje i nagrody;
`tensor_camera.py` — renderer tensorowy; `camera.py` — paleta i renderer referencyjny; `visual_agent.py` — CNN/GRU/BPTT; `train.py` — nauka;
`gui.py` — okno; `world.py` — pola; `maps.py` — generator.
Usunięto dawny agent MLP, kodowanie obserwacji FOV, kierunki kardynalne,
ruch po całych polach i liczniki zagrożeń oparte na liczbie ruchów.

## Projektant map i plan poziomów

Przycisk **Projektant poziomów** otwiera edytor planu treningowego. Początkowy plan ma 10 poziomów; można dodawać i usuwać kolejne. Wybierz poziom na liście, ustaw mapę **Losowa** lub **Moja**, rozmiar **5×5, 7×7, 9×9 lub 11×11** oraz liczbę zaliczeń z rzędu.

Dla mapy własnej wybierz teren z palety i maluj kliknięciem lub przeciągnięciem. START i END przenoszą się na wskazane pole. Zmiana rozmiaru czyści rysunek po potwierdzeniu. Zapis wymaga dokładnie jednego START i END oraz połączenia między nimi bez ścian; nie gwarantuje przeżycia na wodzie lub ogniu.

**Zapisz plan** zapisuje wszystkie poziomy w `config/levels.json`. Następne uruchomienie „Trenuj dalej” lub „Trenuj i oglądaj” korzysta z tego planu. Zmiany zapisane podczas treningu dotyczą dopiero następnego uruchomienia.

Roboty trenują poziomy w kolejności. Mapa własna jest powtarzana, a losowa powstaje od nowa dla każdej grupy. Tak jak wcześniej, wynik najlepszego robota w grupie stanowi jedno zaliczenie. Próg 20 oznacza 20 wygranych grup z rzędu; porażka przerywa serię. Po awansie licznik kolejnego poziomu zaczyna się od zera. Po ukończeniu ostatniego poziomu trening kończy się i zapisuje model.

Plan i postęp są również częścią zapisu modelu. Wznowienie tego samego planu zachowuje poziom i wyniki. Zmiana planu rozpoczyna poziom 1, zachowując wyuczone wagi sieci. W konsoli plan można podać przez `python train.py --plan config/levels.json`; bez tej opcji pozostaje wcześniejszy automatyczny dobór rozmiaru.

Każdy poziom ma także własny **Limit czasu mapy (s)**, od 2 do 1 000 000 sekund czasu symulacji. Limit obowiązuje każdego robota w każdej próbie tego poziomu, niezależnie od tempa podglądu. Generator losowy uwzględnia go przy wyznaczaniu bezpiecznej trasy. Starsze plany otrzymują domyślne 200 s bez utraty postępu; zmiana limitu jest zmianą planu i rozpoczyna go od poziomu 1, zachowując wagi sieci.

Opcja **Start od lvl** przy ustawieniach treningu pozwala wybrać poziom początkowy. **Kontynuuj** zachowuje zapisany postęp. Wybranie numeru rozpoczyna wskazany poziom z zerową serią zaliczeń, zachowując wagi modelu; następnie trening przechodzi do kolejnych poziomów planu. Wybór numeru dotyczy jednego uruchomienia — pole wraca do Kontynuuj, aby kolejne wznowienie nie resetowało postępu. W konsoli: python train.py --plan config/levels.json --start-level 4.

## Ruch i nazwa osobno dla każdego poziomu

Projektant ma pola **Krok robota (m)** (0,01–1 m), **Kąt obrotu (°)** (1–180°) oraz **Nazwa / opis** (1–120 znaków). Przykład: krok 1 m i obrót 90° pozwalają przechodzić między środkami sąsiednich pól i skręcać pod kątem prostym. Domyślne wartości starszych poziomów pozostają 0,1 m i 10°.

Parametry dotyczą wszystkich robotów na danym poziomie, zarówno map własnych, jak i losowych. Przód i tył mają tę samą długość kroku, lewo i prawo ten sam kąt. Czas akcji wynika z przebytej drogi lub kąta i rodzaju terenu. Kolizje sprawdzają cały odcinek ruchu wraz z obrysem robota — większy krok nie przeskakuje ścian. Średnica robota pozostaje 20 cm; kąt obrotu nie zmienia jego promienia.

**Wypróbuj poziom** ładuje aktualnie edytowany poziom do głównego widoku z jego limitem czasu i ruchem, bez konieczności zapisywania całego planu. Można sterować samemu lub uruchomić AI. Restart zachowuje parametry testowanego poziomu. Nowy zwykły labirynt i mapa demonstracyjna wracają do domyślnego ruchu. Podgląd treningu pokazuje aktualny krok i kąt.

Zmiana samej nazwy/opisu zachowuje postęp. Zmiana parametrów ruchu zmienia plan i rozpoczyna go od początku (chyba że wskażesz poziom startowy), zachowując wyuczone wagi. Parametry są zapisane w planie i modelu wraz z postępem. Sieć nadal ma cztery akcje; ich fizyczną wielkość określa poziom. Własne nietypowe kombinacje kroku i kąta warto sprawdzić przyciskiem podglądu: połączenie pól na mapie nie jest gwarancją osiągalności przy dowolnej geometrii ruchów.

W projektancie pole **Kierunek startowy** wybiera Prawo, Lewo, Górę, Dół lub Losowy. Losowy oznacza niezależne losowanie jednego z czterech kierunków dla każdego robota i każdej próby; w treningu jest odtwarzalne z seeda przebiegu. Wypróbuj poziom, restart i AI korzystają z tej samej opcji; przy losowym kierunku restart losuje ponownie. Starsze poziomy zachowują start w prawo. Opcja zapisuje się w planie; jej zmiana jest zmianą reguł poziomu.
