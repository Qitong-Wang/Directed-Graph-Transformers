import sys
from lib.training.execute import get_configs_from_args, execute
import os
import time
if __name__ == '__main__':
    config = get_configs_from_args(sys.argv[:2])

    print("train")
    if config['scheme'] == "MALNETSub":
        if not os.path.exists("./temp_malnet"):
            from lib.data.MALNETSub.parallel_cache import parallel_cache
            parallel_cache(config)
    execute('train', config)
