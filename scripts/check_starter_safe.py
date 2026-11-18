"""Run starter checks without opening final labels."""
import argparse
from pathlib import Path
import sys


def run(starter):
    """Run unchanged starter checks while skipping the final-label hash."""
    starter = Path(starter).resolve()
    sys.path.insert(0, str(starter))
    path = starter / 'check_starter.py'
    source = path.read_text(encoding='utf-8')
    marker = "for filename, expected in provenance['hashes'].items():"
    if source.count(marker) != 1:
        raise RuntimeError('Unexpected checker version; inspect before adapting')
    source = source.replace(marker, marker +
                            "\n        if filename == 'final_labels.json':\n            continue")
    source = source.replace('split separation and data hashes.',
                            'split separation and permitted data hashes (final-label hash skipped).')
    exec(compile(source, str(path), 'exec'),
         {'__name__': '__main__', '__file__': str(path)})


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'course_starter')
    run(parser.parse_args().starter)
