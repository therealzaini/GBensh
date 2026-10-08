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
            print(f"Error: {exc}")
            return False

    def do_quit(self, arg):
        """Politely exiting the program."""
        print("Thank you for using GBensh. Until next time!")
        return True

    do_q = do_quit
    do_EOF = do_quit

    def do_plant(self, arg):
        """Runs a single model on a single dataset with a single seed."""

        from engine.cli_engine.parsers.plant_parser import parse_planting_args


        parser = parse_planting_args()

        try:
            import shlex
            args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            return

        model, dataset, device = args.model, args.dataset, args.device

        from engine.repositories.models import MODELS
        from engine.repositories.datasets import DATASETS

        if DATASETS[dataset]['task'] not in MODELS[model]['task']:
            print(f"Model {model.upper()} incompatible or not yet implemented for {dataset.upper()} task.")
            return

        
        if args.seed is not None:
            seed = args.seed
        else:
            from engine.training_engine.utils.seed_gen import generate_seeds
            seed_schedule = generate_seeds()
            seed = seed_schedule[args.seed_index]

        from engine.training_engine.training.run_planted_seed import run_planted_seed

        run_planted_seed(model_str=model, dataset_str=dataset, seed=seed, training_device=device)
