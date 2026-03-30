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


def add_vis_output_args(parser, *, plot_dir_default):
    """Output/directory args shared by all visualise.py subparsers."""
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root data directory for ResultsIndex scan (default: data)')
    parser.add_argument('--plot-dir', type=str, default=plot_dir_default,
                        help=f'Plot output directory (default: {plot_dir_default})')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to plot-dir')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')


def add_vis_file_selection_args(parser):
    """File selection and prime/seed filter args shared by groks/capacity/speed."""
    parser.add_argument('--p', nargs='+', type=int, default=[97],
                        help='Prime number(s) (default: 97)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed filter (default: 42)')
    parser.add_argument('--files', nargs='+', metavar='FILE',
                        help='Explicit result files to analyze (bypasses index)')
    parser.add_argument('--pattern', type=str,
                        help='Glob pattern escape hatch for file selection (advanced use)')
    parser.add_argument('--all', action='store_true',
                        help='(Legacy no-op) Loading all matching results is now the default')
    parser.add_argument('--list', action='store_true',
                        help='List available results matching current filters and exit')


def add_vis_model_filter_args(parser):
    """Architecture filter args (None = no filter)."""
    parser.add_argument('--depth', type=int, default=2,
                        help='Filter by model depth (default: 2)')
    parser.add_argument('--heads', type=int, default=1,
                        help='Filter by attention heads (default: 1)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Filter by dropout rate (default: no filter)')
    parser.add_argument('--init-scale', type=float, default=None,
                        help='Filter by init scale (default: no filter)')


def add_vis_optimizer_filter_args(parser):
    """Optimizer filter args (None = no filter)."""
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Filter by weight decay (default: no filter — shows all)')


def add_vis_task_args(parser, *, op_default='/', op_choices=None):
    """Task specification args shared by groks/speed/primes."""
    parser.add_argument('--op', type=str, default=op_default,
                        choices=op_choices or ['*', '/', '+', '-'],
                        help=f'Operation (default: {op_default})')
    parser.add_argument('--training-fraction', type=float, default=0.5,
                        help='Training fraction (default: 0.5)')


def add_vis_dim_args(parser):
    """Dimension filter args shared by groks/capacity/speed/primes."""
    parser.add_argument('--dims', nargs='+', type=int, metavar='DIM',
                        help='Filter by model dimensions (e.g., --dims 20 40 80)')
    parser.add_argument('--dims-start', type=int, default=None,
                        help='Start of dimension range')
    parser.add_argument('--dims-end', type=int, default=None,
                        help='End of dimension range (inclusive)')
    parser.add_argument('--dims-step', type=int, default=None,
                        help='Step size for dimension range')
