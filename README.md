# Video till Signal på Raspberry Pi

Programmet övervakar MP4-filer på en monterad disk, analyserar färdigskrivna
filer och skickar en annoterad JPG per identifierad **objektklass och video**
till den konfigurerade Signal-gruppen. Om inga objekt hittas skickas ett
textmeddelande med videons filnamn. Vid första starten inventeras redan befintliga
MP4-filer och ignoreras; nya filer som dyker upp efter inventeringen analyseras.
Inventeringen sparas i SQLite och gäller även efter omstarter.

## Körning

Kör kommandona på Raspberry Pi från `/home/pelle/video_analyzer`.
`./.venv-pi` är den isolerade körmiljön med CPU-byggnaden av PyTorch och
TorchVision som fungerar på Pi:ns Cortex-A72. Den återanvänder projektets
installerade Ultralytics, OpenCV och övriga Python-beroenden. Om miljön behöver
skapas på nytt, kör på Pi:n:
`python3 -m venv --system-site-packages .venv-pi`
`.venv-pi/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu`.
Signal-kommandot använder det länkade kontot och grupp-ID:t i YAML-konfigurationen.

1. Kopiera `config.example.yaml` till `config.yaml`.
2. Ange diskens sökväg, ommonteringskommando och Signal-konto i YAML-filen.
3. Kontrollera YAML utan att köra externa kommandon:
   `./.venv-pi/bin/python video_monitor.py --config config.yaml --check-config`
4. Starta: `./run_monitor.sh` (eller `./run_monitor.sh config.yaml`).

Standardmodellen är `runs/detect/train-4/weights/best.pt`, som körs på CPU.
Ett modelltest utan Signal eller ommontering kan köras med
`python3 video_analyzer.py --no-isolation --self-test`.
Analysera en enskild film med `.venv-pi/bin/python video_analyzer.py /sökväg/film.mp4`.
Analysatorn skriver antal analyserade bildrutor var tionde ruta och när videon är klar.
TFLite-alternativet kräver den separata modulen `tflite_detector` och dess
runtime; dessa följer för närvarande inte med projektet.

`remount.sh` förutsätter en diskavbildning med partition 1 och använder
`sudo -n` för `umount`, `losetup`, `partx` och `mount`. Användaren som kör
monitorn måste ha lösenordsfri sudo-behörighet för dessa kommandon. Övriga
delar av monitorn körs som den vanliga användaren. Starta inte hela analysprogrammet som root för att lösa
monteringsbehörigheter. Anpassa monteringskommandot till din befintliga miljö.
Avbildningen monteras skrivskyddad. Ingen film läses medan ommontering pågår.

## Lokal installation av signal-cli

signal-cli 0.14.9 är installerat i `tools/signal-cli-native` från
[ARM64-paketet signal-cli-native](https://packaging.gitlab.io/signal-cli/).
Det är paketprojektets experimentella GraalVM-variant utan Java. Inga
systempaket eller tjänster har installerats. Paketets SHA256 kontrollerades
mot paketindexet före uppackning.

Kör `./tools/signal-cli --version` från projektkatalogen.
Startskriptet lagrar kontot under `state/signal-cli` med katalogbehörighet 700.
Exempelkonfigurationen använder detta startskript. Kontot behöver fortfarande
kopplas till Signal och exempelnumret ersättas innan sändning kan fungera.

## Konfiguration och standardvärden

Relativa filvägar räknas från YAML-filens katalog. Externa kommandon körs från
projektkatalogen som argumentlistor utan skal. Platshållare ersätts bara när de
utgör ett helt argument: `{mountpoint}`, `{image}`, `{message}` respektive `{group_id}`.

| Inställning | Standardvärde |
| --- | --- |
| watch.path | Obligatorisk; exemplet använder /mnt/video |
| watch.scan_seconds | 10 sekunder |
| watch.stable_seconds | 30 sekunder utan ändrad storlek eller ändringstid |
| watch.recursive | true; scans the mount and subdirectories |
| mount.remount_command | Obligatoriskt; exemplet använder remount.sh med /video/virtual_disk.img |
| mount.interval_seconds | 60 sekunder, efter avslutad ommontering |
| mount.timeout_seconds | 30 sekunder |
| analysis.model | runs/detect/train-8/weights/best.pt |
| analysis.data | Tom sträng; valfri klassnamnsfil för TFLite |
| analysis.confidence | 0.45 |
| analysis.min_object_hits | 3 detektioner per klass |
| analysis.max_bad_frames | 25 |
| analysis.threads | 1 |
| analysis.iou | 0.7 |
| analysis.timeout_seconds | 1800 sekunder |
| analysis.max_attempts | 3 |
| signal.command | signal-cli till angiven grupp; se YAML |
| signal.group_id | Obligatoriskt; Signal-gruppens ID, konfigurerat i YAML |
| signal.timeout_seconds | 120 sekunder |
| storage.path | state, utanför den övervakade katalogen |
| storage.max_spool_mb | 256 MiB; stoppar start av nya analyser när gränsen nåtts |
| retry.initial_seconds | 30 sekunder |
| retry.maximum_seconds | 3600 sekunder |

Ommontering väntar tills aktiv analys eller sändning är avslutad. Intervallet
kan därför bli längre än konfigurerat. Spoolgränsen kontrolleras före en ny
analys och är ingen hård gräns för pågående bildskrivning.

## Återhämtning och loggning

SQLite-kön i state lagrar behandlade filer och väntande sändningar över
omstarter. Filidentitet bygger på sökväg, storlek och ändringstid. Misslyckade
sändningar försöks igen med exponentiellt ökande väntetid. Bilder tas bort
först efter lyckad sändning. En processkrasch precis efter att Signal tagit
emot en bild kan leda till en dubblett vid nästa försök.

Fel loggas till standardfel. Efter tre misslyckade analysförsök markeras filmen
som misslyckad; ändrad filidentitet skapar ett nytt jobb. SIGINT/SIGTERM stoppar
programmet och dess aktiva underprocess. Ett lås hindrar dubbla monitorer med
samma state-katalog. Kör en analys åt gången på Pi:n med 1 GB RAM.

## Tester

Kör på Raspberry Pi: `PYTHONDONTWRITEBYTECODE=1 .venv-pi/bin/python -m unittest discover -s tests -v`.
Testerna använder tillfälliga filer under tests och simulerar ommontering och
Signal. Ett riktigt modelltest behövs också för att kontrollera inferensmiljön.
