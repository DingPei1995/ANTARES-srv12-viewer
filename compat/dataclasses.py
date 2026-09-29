"""``dataclasses`` from the standard library, or its 3.6 backport."""
try:
    from dataclasses import *                      # noqa: F401,F403  (3.7+)
    from dataclasses import dataclass, field, fields, asdict, replace  # noqa: F401
except ImportError:                                # Python 3.6
    from compat.dataclasses36 import *             # noqa: F401,F403
    from compat.dataclasses36 import dataclass, field, fields, asdict, replace  # noqa: F401
