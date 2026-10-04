# Audyt blokowania botów — 4 października 2026

## Wynik

W sprawdzonych przypadkach nie znaleziono błędu powodującego wejście koła
w ścianę lub utratę możliwości odwrócenia poprawnego ruchu. Nie zmieniono
fizyki, nagród, modeli ani ustawień treningu. Dodano 14 powtarzalnych testów
w `test_collision_escape.py`.

To wynik dla opisanych prób, nie dowód braku wszystkich możliwych błędów.
Nie badano decyzji aktualnie zapisanej sieci. Test batcha używa zadanych akcji,
aby odróżnić działanie mechaniki od zachowania polityki.

## Zakres i liczby

| Próba | Zakres | Wynik |
|---|---|---|
| Losowe spacery w labiryntach | 20 seedów map 11×11 × 4 długości kroku × 500 prób = 40 000 | 27 900 ruchów dozwolonych, wszystkie odwracalne; 12 100 kolizji bez zmiany pozycji |
| Geometria odcinek–ściana | 6000 losowych odcinków, niezależna minimalizacja odległości od prostokąta | Zgodność do tolerancji 1e-9; zgodność po odwróceniu kierunku do 1e-12 |
| Pełna decyzja o dopuszczeniu ruchu | 4000 losowych ścieżek, trzy przeszkody i zewnętrzne granice mapy | Zgodność z niezależną geometrią |
| Styk ze ścianami | Cztery strony, kroki 0,01 / 0,1 / 0,5 / 1 m, po 20 zablokowanych próbach | Cofanie działa, seria kolizji zeruje się po udanym ruchu |
| Granice mapy | Cztery brzegi i cztery narożniki, cztery długości kroku | Ruch poza mapę blokowany; cofanie do wnętrza możliwe |
| Wypukły narożnik ściany | 91 kątów kontaktu: 0–90° | Styk dozwolony, ruch w przeszkodę blokowany, cofanie działa |
| Przecinanie narożnika | Oba końce ruchu wolne, obrys przecina narożnik w środku drogi | Prawidłowa blokada bez przemieszczenia; cofanie działa |
| Obrót przy dwóch ścianach | Po 720 obrotów dla kątów 10 / 15 / 45 / 90° | Obrót nie przesuwa bota ani nie wpycha go w ścianę |
| Ślepy zaułek | Dojazd do blokady, 10 ponownych kolizji, cofnięcie całej drogi | Powrót do pozycji startowej dla czterech długości kroku |
| Błędy zaokrągleń przy styku | 1000 cykli: kolizja → cofnięcie → powrót | Brak narastającego wnikania w ścianę |
| Granice terenów | Ziemia, bruk, bagno, woda, ogień; cztery długości kroku | Granice terenów nie blokują ruchu jak ściany |
| Batch treningowy | 8 botów, zadane uderzanie w ścianę, potem cofanie | Wszystkie cofają się; historia przebiegu zapisuje zmianę pozycji |
| Koniec czasu | Limit 0,5; geometrycznie wolna droga wstecz | Epizod terminalny odrzuca dalszą akcję zgodnie z regułą zakończenia |

W `levels.json` podczas audytu ustawiono krok 0,1 m, obrót 15°, losowy kierunek
startowy i limit 300. Testy obejmują te wartości oraz większe i mniejsze kroki.
„Krawężnik” nie jest osobnym typem przeszkody w tym projekcie: sprawdzono
krawędzie pól ścian, narożniki, granice mapy oraz przejścia między terenami.

## Co może wyglądać jak zakleszczenie

1. **Bot nadal wybiera ruch w ścianę.** Kolizja nie przemieszcza bota, ale
   zużywa czas. Sama mechanika pozwala cofnąć lub obrócić żywego bota.
   Testy nie ustalają, dlaczego aktualna sieć wybiera konkretną akcję.
2. **Przód i tył są oba zablokowane.** W korytarzu o szerokości 1 m bot
   ustawiony w poprzek z krokiem 0,5 m nie może wykonać żadnej z tych akcji.
   Sześć obrotów po 15° ustawia go wzdłuż korytarza i pozwala ruszyć.
   Poprawnie zablokowany przód nie oznacza, że przeciwny kierunek zawsze
   będzie wolny — to zależy od otoczenia i długości kroku.
3. **Epizod już zakończony.** Koniec czasu, śmierć, wygrana i łapka kończą
   przebieg. Dalszych ruchów nie wykonuje się, nawet gdy droga jest wolna.

## Odtworzenie

```powershell
python -m unittest test_collision_escape -v
python -m unittest test_collision_escape test_training_dashboard test_live_gui test_validation test_batching -q
```

Pierwsze polecenie uruchamia 14 testów audytu, drugie 46 testów audytu,
podglądu, walidacji i batcha. Oba zestawy przeszły.
Sześć starszych testów w `test_game.py` opisanych w README nadal oczekuje
wyłączonych kar `collision` i `staying`; audyt ich nie zmienia.
