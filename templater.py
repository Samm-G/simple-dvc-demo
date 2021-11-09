import os


# Make Dirs..
dirs = [
    os.path.join('data','raw'),
    os.path.join('data','processed'),
    os.path.join('notebooks'),
    os.path.join('saved_models'),
    os.path.join('src'),
]
for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, '.gitkeep'), 'w') as f:
        pass

# Make Files..
files = [
    'dvc.yaml',
    'params.yaml',
    '.gitignore',
    os.path.join('src','__init__.py'),
    'README.md'
]
for file_ in files:
    with open(file_,'w') as f:
        pass