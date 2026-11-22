"""Resume a version-two campaign, including failed audits, without AST recovery."""
import sys
from ..campaigns.run_p97_capacity_scaling import main

if __name__ == '__main__':
    if '--resume' not in sys.argv:
        sys.argv.append('--resume')
    main()
