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
    parser.add_argument('--run-uuid', type=str, default=None,
                        help='Pre-generated wallow run uuid (set by dispatcher). '
                             'When omitted the worker generates its own and registers '
                             'the run from scratch.')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Multi-node shard index (recorded as annotation)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Override path to the wallow runs.db (default: ./runs.db)')


# ---------------------------------------------------------------------------
# Visualisation helpers — used by the legacy root visualise.py / plotting.py.
# Kept here so that whole subtree still has a single argparse module.
# ---------------------------------------------------------------------------

def add_vis_output_args(parser, *, plot_dir_default: str):
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root directory for legacy .meta.json scan')
    parser.add_argument('--plot-dir', type=str, default=plot_dir_default,
                        help=f'Plot output directory (default: {plot_dir_default})')
    parser.add_argument('--save', action='store_true', help='Save plots to plot-dir')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')


def add_vis_file_selection_args(parser):
    parser.add_argument('--p', nargs='+', type=int, default=[97],
                        help='Prime number(s) (default: 97)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed filter')
    parser.add_argument('--files', nargs='+', metavar='FILE',
                        help='Explicit result files (bypasses index)')
    parser.add_argument('--pattern', type=str, help='Glob pattern escape hatch')
    parser.add_argument('--all', action='store_true', help='(Legacy no-op)')
    parser.add_argument('--list', action='store_true', help='List results matching filters')


def add_vis_model_filter_args(parser, *, dropout_default=None, init_scale_default=None):
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=dropout_default)
    parser.add_argument('--init-scale', type=float, default=init_scale_default)


def add_vis_optimizer_filter_args(parser, wd_nargs=None, wd_default=None):
    parser.add_argument('--weight-decay', type=float, nargs=wd_nargs, default=wd_default,
                        help='Weight-decay filter (single value or "+" list).')


def add_vis_task_args(parser, *, op_default='/', op_choices=None):
    parser.add_argument('--op', type=str, default=op_default,
                        choices=op_choices or ['*', '/', '+', '-'])
    parser.add_argument('--training-fraction', type=float, default=0.5)


def add_vis_dim_args(parser):
    parser.add_argument('--dims', nargs='+', type=int, metavar='DIM')
    parser.add_argument('--dims-start', type=int, default=None)
    parser.add_argument('--dims-end', type=int, default=None)
    parser.add_argument('--dims-step', type=int, default=None)
