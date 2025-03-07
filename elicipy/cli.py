import click
import sys

from .core import run_elicitation

@click.command()
def main():
    """Run EliciPy from the command line."""
    run_elicitation(sys.argv[1:])

if __name__ == "__main__":
    main()

