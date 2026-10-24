#!/bin/bash
# Called by video_monitor only when no video is being read.
set -euo pipefail

image=${1:-/video/virtual_disk.img}
target=${2:-/mnt/video}

if ! test -f "$image"; then
    echo "Disk image not found: $image" >&2
    exit 1
fi
if ! test -d "$target"; then
    echo "Mountpoint not found: $target" >&2
    exit 1
fi

# Stop immediately on a busy mount; never replace a mount still in use.
if mountpoint -q -- "$target"; then
    sudo -n umount -- "$target"
fi

loop_device=$(sudo -n losetup --associated "$image" --noheadings --output NAME)
if test -z "$loop_device"; then
    loop_device=$(sudo -n losetup --find --nooverlap --show --partscan --read-only "$image")
fi
if test -z "$loop_device" || test "${loop_device#*$'\n'}" != "$loop_device"; then
    echo "Expected exactly one loop device for $image" >&2
    exit 1
fi
partition="${loop_device}p1"

# An existing loop device can survive while its partition mapping disappears.
if ! test -b "$partition"; then
    sudo -n partx --add --nr 1 "$loop_device"
fi

# Allow device-node creation to catch up, without an unbounded wait.
for attempt in {1..25}; do
    if test -b "$partition"; then
        sudo -n mount -o ro -- "$partition" "$target"
        exit 0
    fi
    sleep 0.2
done
echo "Partition did not appear: $partition" >&2
exit 1
