import shlex
from cmd import Cmd


class GBensh(Cmd):

    prompt = "(GBensh) "
    intro = "Welcome to GBensh, the Graph Benchmarking Shell. Type 'help' or '?' to list available commands.\n"

    def onecmd(self, line):
        try:
            return super().onecmd(line)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            self.stdout.write(f"Error: {exc}\n")
            return False

    def do_quit(self, arg):
        """Politely exiting the program."""
        self.stdout.write("Thank you for using GBensh. Until next time!\n")
        return True

    do_q = do_quit
    do_EOF = do_quit

    def do_plant(self, arg):
        """Run one model on one dataset; use 'plant --help' for options."""

        from engine.cli_engine.parsers.plant_parser import parse_planting_args

        parser = parse_planting_args()

        try:
            args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            return

        from engine.training_engine.training.commands.run_planted_seed import run_planted_seed

        result = run_planted_seed(
            model_str=args.model,
            dataset_str=args.dataset,
            seed=args.seed,
            seed_index=args.seed_index,
            training_device=args.device,
        )
        self.stdout.write(
            f"{result.model.upper()} on {result.dataset.upper()} | "
            f"seed={result.seed} | device={result.device} | "
            f"test {result.metric}={result.test_metric:.4f} | "
            f"test loss={result.test_loss:.4f}\n"
        )
