# Korištenje TiPuM-a

Za instalaciju i pokretanje pogledajte [README](../README.md).

## Obrada razgovora

U izborniku odaberite novu snimku ili transkripciju postojeće. Postojeći WAV stavite u `recordings/` s nazivom `<broj>_snimka.wav`, a zatim unesite taj broj. Mapa za nove snimke nastaje pri radu aplikacije.

Za svaku snimku rezultati se spremaju zajedno u `../output/<broj>_snimka/`:

- `<naziv>.txt` — transkript za čitanje;
- `<naziv>_strukturirano.json` — tekst s vremenima, govornicima i ulogama;
- `<naziv>_medicinski_sazetak.txt` i `.json` — izdvojene medicinske informacije;
- `<naziv>_lijecnicki_zapis.txt` i `.json` — zapis u trećem licu.

TXT služi za čitanje, a JSON za daljnju obradu. Postojeći izlazi ne prepisuju se.

## Ispravci

Odaberite **3 – Uredi transkript**, uredite tekst i spremite ga. Prethodni i novi tekst ostaju zapisani u povijesti, dostupnoj kroz **4 – Prikaži povijest**. Posebno provjerite lijekove, doze, brojeve, negacije i pripadnost iskaza govorniku. Nakon izmjene transkripta provjerite odgovaraju li mu i sažetak i zapis.

Uređuje se radna kopija. Zadržite jedan redak po iskazu, postojeći broj redaka i oznaku govornika ispred dvotočke. Nakon spremanja i zatvaranja uređivača automatski se usklađuju strukturirani transkript, medicinski sažetak i zapis u trećem licu. Ponovna izrada koristi deterministički postupak, bez pozivanja Qwena. Ako provjera izmjene ne uspije, izvorni izlazi ostaju sačuvani, a radna kopija ostaje za sljedeći pokušaj uređivanja.

Vremena iskaza i izvorni tekst ostaju sačuvani. Izmijenjene riječi nemaju ponovno izračunate vremenske oznake; stare oznake čuvaju se odvojeno kao izvorni podaci. Stari samostalni TXT bez strukturiranog zapisa može se urediti, ali iz njega se ne izmišljaju govornici ni vremenske oznake.

## Ako obrada ne radi

| Problem | Postupak |
|---|---|
| Pyannote model nije moguće učitati | Ako model još nije preuzet, prihvatite njegove uvjete na Hugging Faceu i slijedite postupak preuzimanja iz README-a. |
| Nedostaje snimka | Provjerite naziv i mapu `recordings/`. |
| Izlaz već postoji | Upotrijebite novi broj izlaza ili sačuvajte postojeći skup na drugom mjestu. |
| Nedostaje FFmpeg ili Python paket | Pokrenite `../development/testiraj.cmd` i provjerite instalaciju. |
| Obrada je prekinuta | Sačuvajte nastale datoteke i ponovite obradu pod novim brojem; djelomičan izlaz nije dovršen rezultat. |

`pokreni.cmd` automatski pokreće lokalni LLM. Sažetak mu daje najviše tri pokušaja, a nakon neuspjelih ili nepotpunih odgovora koristi deterministički rezultat.

Snimajte uz dopuštenje sudionika, koristite broj umjesto imena i čuvajte snimke i transkripte lokalno uz ograničen pristup. Automatski zapis treba ljudsku provjeru.
