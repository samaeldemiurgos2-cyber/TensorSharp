"""Manifest-derived shard selection shared by Muse Spark validation tools."""
import re


def shard_entries(manifest, quantization):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', quantization):
        raise ValueError('Invalid quantization directory')
    entries = sorted((row for row in manifest['siblings']
                      if row['rfilename'].startswith(quantization + '/')
                      and row['rfilename'].endswith('.gguf')), key=lambda row: row['rfilename'])
    pattern = re.compile(rf'{re.escape(quantization)}/Muse-Spark-Open-816B-A42B-'
                         rf'{re.escape(quantization)}-(\d{{5}})-of-(\d{{5}})\.gguf')
    matches = [pattern.fullmatch(row['rfilename']) for row in entries]
    if not entries or not all(matches):
        raise ValueError('Manifest contains no complete named shard set for ' + quantization)
    count = int(matches[0][2])
    if count < 1 or len(entries) != count or any(int(m[2]) != count or int(m[1]) != n
                                                for n, m in enumerate(matches, 1)):
        raise ValueError('Missing, duplicate or inconsistent shards for ' + quantization)
    return entries
