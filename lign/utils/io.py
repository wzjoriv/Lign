import json as jn
import os
import pickle as pk

def unpickle(fl):
    with open(fl, 'rb') as f:
        dict = pk.load(f)
    return dict

def pickle(data, fl):
    with open(fl, 'wb') as f:
        pk.dump(data, f)

def unjson(fl):
    with open(fl, 'r') as f:
        dict = jn.load(f)
    return dict

def json(data, fl):
    with open(fl, 'w') as f:
        jn.dump(data, f, indent=4)

def move_file(fl1, fl2):
    os.rename(fl1, fl2)

def move_dir(dir1, dir2):
    os.renames(dir1, dir2)

def to_iter(data):
    try:
        iter(data)
    except TypeError:
        data = [data]
    if type(data) == str:
        data = [data]
    return data

def is_primitve(data):
    return type(data) in (int, str, bool, float)