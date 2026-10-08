from engine.cli_engine.gbensh_cmd import GBensh


def main():
    try:
        GBensh().cmdloop()
    except KeyboardInterrupt:
        print("\nKilling GBensh.")


if __name__ == "__main__":
    main()