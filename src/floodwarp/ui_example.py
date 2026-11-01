"""Copy the installed tiny synthetic input set into the current directory."""
from pathlib import Path
import shutil


def copy_example(*, output):
    source = Path(__file__).with_name('ui_assets') / 'example'
    shutil.copytree(src=source, dst=output)
    print(f'Synthetic local-interface inputs: {output.resolve()}')


if __name__ == "__main__":
    output = Path('floodwarp-example')
    copy_example(output=output)
