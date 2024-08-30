import glob
import os

config_files_list = []


config_files_list += sorted(glob.glob("*.json", recursive=True, root_dir="configFiles/"))

config_files_list_no_exc = [i[:-5] for i in config_files_list if "(excluded)" not in i]
if len(config_files_list_no_exc) == 0:
    config_files_list = [i[:-5] for i in config_files_list]
else:
    config_files_list = config_files_list_no_exc

print("Loading Config Files:")

print("\n".join(config_files_list))

for config_file in config_files_list:
    if "test" not in config_file:
        print("==========================================================================================")
        os.system(
            f"python3 main.py --config_file 'configFiles/{config_file}.json' --mode train"
        )

for config_file in config_files_list:
    if "train" not in config_file:
        print("==========================================================================================")
        os.system(
            f"python3 main.py --config_file 'configFiles/{config_file}.json' --mode test"
        )
        # os.system(
        #     f"python3 main.py --config_file 'configFiles/{config_file}.json' --mode test --vis"
        # )
