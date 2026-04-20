"""CLI entrypoint for training CREATE-Pone."""

from create_pone.trainer import build_arg_parser, train_create_pone


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    print(f"Starting training with args: {vars(cli_args)}")
    train_create_pone(cli_args)
