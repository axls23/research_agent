import sys
import fast_rlm
import inspect

with open("fast_rlm_inspect.txt", "w") as f:
    f.write("--- dir(fast_rlm) ---\n")
    f.write(str(dir(fast_rlm)) + "\n")

    f.write("\n--- help(fast_rlm.run) ---\n")
    try:
        f.write(str(inspect.getdoc(fast_rlm.run)) + "\n")
    except Exception as e:
        f.write(str(e) + "\n")

    f.write("\n--- help(fast_rlm.RLMConfig) ---\n")
    try:
        f.write(str(inspect.getdoc(fast_rlm.RLMConfig)) + "\n")
    except Exception as e:
        f.write(str(e) + "\n")
