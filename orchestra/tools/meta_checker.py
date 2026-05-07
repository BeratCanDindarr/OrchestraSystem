"""Tool to scan .meta files for GUID duplicates in Unity project."""
import os
import sys
import re

def run_meta_check(project_path="."):
    guid_map = {}
    duplicates = []
    
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".meta"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        match = re.search(r"guid: ([a-z0-9]+)", content)
                        if match:
                            guid = match.group(1)
                            if guid in guid_map:
                                duplicates.append((guid, path, guid_map[guid]))
                            else:
                                guid_map[guid] = path
                except: pass
    
    if not duplicates:
        print("✅ No duplicate GUIDs found.")
    else:
        print(f"❌ Found {len(duplicates)} duplicate GUIDs!")
        for guid, p1, p2 in duplicates:
            print(f"Conflict: {guid}\n  File 1: {p1}\n  File 2: {p2}\n")

if __name__ == "__main__":
    run_meta_check()
