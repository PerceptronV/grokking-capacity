"""Shared argparse argument groups for the experiment scripts."""


def add_model_args(parser, *, dropout_default: float):
    parser.add_argument('--p', type=int, default=97,
                        help='Prime number (vocabulary size = p + 2)')
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=dropout_default,
                        help='Dropout rate')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Weight initialisation scale factor')
    parser.add_argument('--dim', type=int, required=True, help='Model dimension')


def add_optimizer_args(parser, *, weight_decay_default: float, epochs_default: int):
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=weight_decay_default,
                        help='AdamW weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=epochs_default,
                        help='Maximum training epochs')
    parser.add_argument('--mup-base', type=int, default=0,
                        help='Per-layer Adam lr scaling (muP, lr-only). 0 = off. N > 0: '
                             'embeddings and norm gains at --lr, all matrix parameters at '
                             'lr * N / dim. Identifying field (wallow.toml: mup_base).')


def add_device_args(parser):
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override (e.g. "cuda:0", "cpu", "mps")')


def add_io_args(parser, *, data_dir: str):
    parser.add_argument('--data-dir', type=str, default=data_dir,
                        help='Root directory for artefact output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if a completed wallow row exists')


def add_registry_args(parser):
    """Args used by the dispatcher to plumb wallow state into a worker."""
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Multi-node shard index (recorded as annotation)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Override path to the wallow runs.db (default: ./runs.db)')
