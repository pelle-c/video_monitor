# Video till Signal på Raspberry Pi

Programmet övervakar MP4-filer på en monterad disk, analyserar färdigskrivna
filer och skickar en annoterad JPG per identifierad **objektklass och video**
till den konfigurerade Signal-gruppen. Om inga objekt hittas skickas ett
textmeddelande med videons filnamn och en bildruta från en sekund in i videon. Om
videon är kortare än en sekund används dess sista tillgängliga bildruta. Vid första starten inventeras redan befintliga
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

## Köra en TFLite-modell

Installera LiteRT-runtime i projektets virtuella miljö på Raspberry Pi:n:
`.venv-pi/bin/python -m pip install -r requirements-tflite.txt`.
Kör sedan en film direkt med exempelvis:

```sh
.venv-pi/bin/python video_analyzer.py /sökväg/film.mp4 \
  --model runs/detect/train-8/weights/best_saved_model/best_float32.tflite
```

För övervakaren ändrar du `analysis.model` i `config.yaml` till önskad
`.tflite`-fil; den befintliga standardmodellen ändras inte automatiskt. Ultralytics-
exporter använder modellens metadata för klassnamn och bildstorlek. För andra
modeller kan du ange en klassnamnsfil med `analysis.data` eller `--data`.
Vanliga `.tflite`-filer körs på CPU. Filer med `edgetpu` i namnet
kräver i stället Coral-installationen nedan.

## Köra en modell på Coral Edge TPU

Pi:n ser redan USB-acceleratorn. För en modell som använder den måste filen vara
kompilerad för Edge TPU och ha `edgetpu` i filnamnet. Exempelmodellen i projektet
är `runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite`.
Installera Python-runtime i `.venv-pi`:

```sh
.venv-pi/bin/python -m pip install -r requirements-edgetpu.txt
```

Filen låser NumPy under 2 och OpenCV 4.11 eftersom
`tflite-runtime 2.14.0` använder NumPy 1.x-ABI:t.

Den lokalt byggda throttled-runtime finns här:

`tools/edge-tpu-build/libedgetpu/out/throttled/k8/libedgetpu.so.1.0`

För att Python ska hitta `libedgetpu.so.1` vid start behöver runtime-katalogen
finnas i `LD_LIBRARY_PATH`. Ange den för varje kommando som kör Coral-modellen:

```sh
LD_LIBRARY_PATH=/home/pelle/video_analyzer/tools/edge-tpu-build/libedgetpu/out/throttled/k8 \
  .venv-pi/bin/python video_analyzer.py /sökväg/film.mp4 \
  --model runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite
```

För monitorn används samma miljövariabel:

```sh
LD_LIBRARY_PATH=/home/pelle/video_analyzer/tools/edge-tpu-build/libedgetpu/out/throttled/k8 \
  ./run_monitor.sh config.yaml
```

Detta gäller när runtime-biblioteket ligger i projektets byggkatalog. Om
`libedgetpu.so.1` i stället installeras i en standardsökväg av systemets
pakethanterare behövs normalt inte `LD_LIBRARY_PATH`.

Installera sedan Coral-systembiblioteket och udev-regeln på Raspberry Pi:n:

```sh
sudo install -d -m 755 /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install libedgetpu1-std
```

Koppla ur och in USB-acceleratorn efter paketinstallationen så att udev-regeln
aktiveras. Kontrollera att `ldconfig -p` visar `libedgetpu.so.1` och att den
vanliga användaren kan öppna USB-enheten. Kör en enskild video med:

```sh
.venv-pi/bin/python video_analyzer.py /sökväg/film.mp4 \
  --model runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite
```

För övervakaren anger du samma modellfil i `analysis.model` i `config.yaml`.
Modeller utan `edgetpu` i namnet körs på CPU även om Coral är ansluten.

`remount.sh` förutsätter en diskavbildning med partition 1 och använder
`sudo -n` för `umount`, `losetup`, `partx` och `mount`. Användaren som kör
monitorn måste ha lösenordsfri sudo-behörighet för dessa kommandon. Övriga
delar av monitorn körs som den vanliga användaren. Starta inte hela analysprogrammet som root för att lösa
monteringsbehörigheter. Anpassa monteringskommandot till din befintliga miljö.
Avbildningen monteras skrivskyddad. Ingen film läses medan ommontering pågår.

## USB-gadget-tjänst

Projektet innehåller systemd-enhetsfilen [`systemd/usb-gadget.service`](systemd/usb-gadget.service)
och dess startskript [`systemd/usb_gadget_start.sh`](systemd/usb_gadget_start.sh).
Tjänsten exponerar diskavbildningen `/video/virtual_disk.img` som USB-masslagring.
Startskriptet använder ConfigFS, en loop-enhet och USB Device Controller (UDC);
systemd kör tjänsten som root. Skriptet förutsätter att avbildningen finns på
den angivna sökvägen.

Från projektkatalogen på Raspberry Pi:n installeras eller uppdateras båda filerna
med:

```sh
sudo install -m 755 systemd/usb_gadget_start.sh /usr/local/bin/usb_gadget_start.sh
sudo install -m 644 systemd/usb-gadget.service /etc/systemd/system/usb-gadget.service
sudo systemctl daemon-reload
```

Aktivera vid uppstart med
`sudo systemctl enable usb-gadget.service`; starta eller starta om med
`sudo systemctl start usb-gadget.service` respektive
`sudo systemctl restart usb-gadget.service`. Kontrollera den med
`systemctl status usb-gadget.service`. Denna projektfil dokumenterar enheten;
att lägga till filerna i projektet ändrar inte automatiskt den installerade
systemd-konfigurationen.

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
| analysis.max_bad_frames | 25; varnar efter tröskeln men fortsätter hoppa över oläsbara bildrutor |
| analysis.threads | 1; används även för Edge TPU:s CPU-fallback |
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

Innan analys kopieras MP4-filen till jobbets katalog under `storage.path`. En
kontroll utan modell verifierar sedan att kopian kan öppnas och att en bild nära
slutet går att avkoda. Om kontrollen misslyckas väntar jobbet och försöker igen
utan att förbruka analysens maxförsök. Filens identitet kontrolleras också igen
efter kopiering och validering. Detta minskar risken att analysera en MP4 medan
Arlo fortfarande skriver den, men kan inte garantera att hubben har flushat all
data till disken. Analysen läser den lokala kopian, så pågående inferens är
oberoende av USB-mounten. Kopian tas bort när analysen är klar; spoolgränsen
räknar även med videokopian, så stora filer kan få vänta tills lagringsutrymme
finns.

Fel loggas till standardfel. Efter tre misslyckade analysförsök markeras filmen
som misslyckad; ändrad filidentitet skapar ett nytt jobb. SIGINT/SIGTERM stoppar
programmet och dess aktiva underprocess. Ett lås hindrar dubbla monitorer med
samma state-katalog. Kör en analys åt gången på Pi:n med 1 GB RAM.

## Tester

Kör på Raspberry Pi: `PYTHONDONTWRITEBYTECODE=1 .venv-pi/bin/python -m unittest discover -s tests -v`.
Testerna använder tillfälliga filer under tests och simulerar ommontering och
Signal. Ett riktigt modelltest behövs också för att kontrollera inferensmiljön.

## Utveckling

Codex har gjort mesta jobbet via codex cli på en laptop. Montering av rpi projektmapp har gjorts med sshfs - `sshfs user@rpi:/home/user/video_analyzer /home/user/rpi-project -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3`
