"""SAL-3 stage entry: rebind the SAL-1 machinery to SAL-3's registered
campaign constants, then run the proven driver unchanged."""

from . import sal3


def main():
    sal3.rebind()
    sal3.sal_run.main()


if __name__ == "__main__":
    main()
