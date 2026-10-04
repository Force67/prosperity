#!/usr/bin/env python3
"""Add a non-Steam shortcut for Prosperity to every Steam user on this Deck.

Steam rewrites shortcuts.vdf on exit, so run this while Steam is shut down
(`steam -shutdown`). Prints the shortcut's game id for
`steam steam://rungameid/<id>`. Re-running is a no-op for users that have it.
"""
import os
import struct
import sys
import zlib

NAME = "Prosperity"
HOME = os.path.expanduser("~")
EXE = f'"{HOME}/prosperity/prosperity.sh"'
START_DIR = f'"{HOME}/prosperity/"'
ICON = f"{HOME}/prosperity/logo.png"


def read_map(data, pos):
    result = {}
    while data[pos] != 0x08:
        kind = data[pos]
        end = data.index(b"\0", pos + 1)
        key = data[pos + 1:end].decode()
        pos = end + 1
        if kind == 0x00:
            result[key], pos = read_map(data, pos)
        elif kind == 0x01:
            end = data.index(b"\0", pos)
            result[key] = data[pos:end].decode()
            pos = end + 1
        elif kind == 0x02:
            result[key] = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unknown vdf type {kind:#x}")
    return result, pos + 1


def write_map(value):
    out = b""
    for key, item in value.items():
        name = key.encode() + b"\0"
        if isinstance(item, dict):
            out += b"\x00" + name + write_map(item)
        elif isinstance(item, int):
            out += b"\x02" + name + struct.pack("<I", item)
        else:
            out += b"\x01" + name + item.encode() + b"\0"
    return out + b"\x08"


def main():
    appid = zlib.crc32((EXE + NAME).encode()) | 0x80000000
    root = f"{HOME}/.local/share/Steam/userdata"
    for user in os.listdir(root):
        path = f"{root}/{user}/config/shortcuts.vdf"
        shortcuts = {}
        if os.path.exists(path):
            data = open(path, "rb").read()
            shortcuts = read_map(data, 0)[0].get("shortcuts", {})
        if any(s.get("appid") == appid for s in shortcuts.values()):
            continue
        shortcuts[str(len(shortcuts))] = {
            "appid": appid, "AppName": NAME, "Exe": EXE,
            "StartDir": START_DIR, "icon": ICON, "ShortcutPath": "",
            "LaunchOptions": "", "IsHidden": 0, "AllowDesktopConfig": 1,
            "AllowOverlay": 1, "OpenVR": 0, "Devkit": 0, "DevkitGameID": "",
            "DevkitOverrideAppID": 0, "LastPlayTime": 0, "FlatpakAppID": "",
            "tags": {},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(write_map({"shortcuts": shortcuts}))
        print(f"added to user {user}", file=sys.stderr)
    print((appid << 32) | 0x02000000)


main()
