#!/bin/bash

# 1. Ladda nödvändiga kärnmoduler
modprobe configfs
modprobe libcomposite
sleep 0.5

# 2. Skapa loop-enheten om den inte redan är aktiv
if ! losetup -a | grep -q "virtual_disk.img"; then
    echo "Skapar virtuell loop-enhet..."
    losetup -fP /video/virtual_disk.img
fi

# NY DEBIAN 12 SÄKRING: Hämta enheten via losetup -j (ger garanterat rätt /dev/loopX-sökväg)
LOOP_DEV=$(losetup -j /video/virtual_disk.img | awk -F: '{print $1}')

if [ -z "$LOOP_DEV" ]; then
    echo "ERROR: Kunde inte hitta loop-enhet!"
    exit 1
fi
echo "Använder virtuell enhet: $LOOP_DEV"

# 3. Gå till ConfigFS och städa om g1 redan existerar
cd /sys/kernel/config/usb_gadget/ || exit 1
if [ -d "g1" ]; then
    echo "Städar gammal konfiguration..."
    echo "" > g1/UDC 2>/dev/null
    rm -f g1/configs/c.1/mass_storage.usb0 2>/dev/null
    rmdir g1/configs/c.1/strings/0x409 2>/dev/null
    rmdir g1/configs/c.1 2>/dev/null
    rmdir g1/functions/mass_storage.usb0 2>/dev/null
    rmdir g1/strings/0x409 2>/dev/null
    rmdir g1 2>/dev/null
    sleep 0.5
fi

# 4. Skapa ny gadget-struktur
mkdir -p g1
cd g1 || exit 1

# 5. Sätt standard USB-ID
echo "0x7614" > idVendor
echo "0x1000" > idProduct
echo "0x0100" > bcdDevice
echo "0x0200" > bcdUSB

mkdir -p strings/0x409
echo "123456789" > strings/0x409/serialnumber
echo "Raspberry Pi" > strings/0x409/manufacturer
echo "USB Mass Storage Device" > strings/0x409/product

# 6. Skapa konfiguration
mkdir -p configs/c.1/strings/0x409
echo "Mass Storage" > configs/c.1/strings/0x409/configuration
echo "250" > configs/c.1/bmAttributes

# 7. Skapa masslagringsfunktionen
mkdir -p functions/mass_storage.usb0

echo "0" > functions/mass_storage.usb0/stall
echo "1" > functions/mass_storage.usb0/lun.0/removable
echo "0" > functions/mass_storage.usb0/lun.0/ro
echo "0" > functions/mass_storage.usb0/lun.0/cdrom
echo "0" > functions/mass_storage.usb0/lun.0/nofua

# Skriv den verifierade loop-sökvägen
echo "$LOOP_DEV" > functions/mass_storage.usb0/lun.0/file

# 8. Länka funktionen till konfigurationen
ln -sf functions/mass_storage.usb0 configs/c.1/

# 9. Hitta rätt UDC-namn och aktivera hårdvaran
sleep 0.5
UDC_NAME=$(ls /sys/class/udc | head -n 1)

if [ -n "$UDC_NAME" ]; then
    echo "Aktiverar USB på kontroller: $UDC_NAME"
    echo "$UDC_NAME" > UDC
else
    echo "ERROR: Hittade ingen USB Device Controller (UDC)!"
fi
