"""Shared argparse argument groups for capacity.py, speed.py, and groks.py."""


def add_model_args(parser, *, dropout_default):
    """Architecture hyperparameters shared across all experiment types."""
    parser.add_argument('--p', type=int, default=97,
                        help='Prime number (vocabulary size = p)')
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=dropout_default,
                        help='Dropout rate')
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help='Weight initialisation scale factor (default: 1.0, no change)')
    parser.add_argument('--dim', type=int, required=True,
                        help='Model dimension')


def add_optimizer_args(parser, *, weight_decay_default, epochs_default):
    """Optimiser hyperparameters shared across all experiment types."""
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=weight_decay_default,
                        help='AdamW weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=epochs_default,
                        help='Maximum training epochs')


def add_device_args(parser):
    """Device selection flags shared across all experiment types."""
    parser.add_argument('--cpu', action='store_true', help='Force CPU only')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0", "cuda:1", "cpu", "mps"). '
                             'Overrides --cpu if specified.')


def add_io_args(parser, *, data_dir):
    """I/O and run-control flags shared across all experiment types."""
    parser.add_argument('--data-dir', type=str, default=data_dir,
                        help='Data output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
