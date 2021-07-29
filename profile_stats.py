import pstats
from pstats import SortKey
p = pstats.Stats('profile_output')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_callers(30)

# print(p.print_callers('5.168 wandb_run.py:2498(finish)').print_stats())
