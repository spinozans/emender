"""SAL-2 stage entry: rebind the SAL-1 machinery to SAL-2's registered
campaign constants, then run the proven driver unchanged."""

from . import sal2


def main():
    sal2.rebind()
    sal2.sal_run.main()


if __name__ == "__main__":
    main()
