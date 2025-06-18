import pickle
import numpy as np

def print_detailed_structure(data, indent=0):
    prefix = ' ' * indent
    if isinstance(data, dict):
        print(f"{prefix}Dict with keys:")
        for k, v in data.items():
            v_type = type(v)
            print(f"{prefix}  Key: {k} -> Type: {v_type}", end='')
            if isinstance(v, np.ndarray):
                print(f"; shape: {v.shape}, dtype: {v.dtype}")
            elif isinstance(v, dict):
                print()
                print_detailed_structure(v, indent + 4)
            elif isinstance(v, (float, np.floating)):
                print(f"; value: {v:.6f}")
            elif isinstance(v, (int, np.integer)):
                print(f"; value: {v}")
            else:
                # For other types, just print a short repr
                print(f"; value: {repr(v)[:50]}")
    elif isinstance(data, list):
        print(f"{prefix}List of length {len(data)}")
        if len(data) > 0:
            print(f"{prefix}First element structure:")
            print_detailed_structure(data[0], indent + 4)
    else:
        print(f"{prefix}{type(data)}: {repr(data)[:50]}")

# Replace this with your pickle file path
pickle_file = '/data/xinyu/demo_dexart_Jun18/laptop/demo_0.pkl'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

#print(data[1]["obs"]["palm_pose.q"])
for d in data:
    print(d["obs"]["progress"])

print(f"Total number of samples: {len(data)}\n")
print_detailed_structure(data)

#print("Count of 1s in each seg_vector column:", np.sum(data[0]["obs"]["observed_pc_seg-gt"] == 1, axis=0))
#print("Count of 1s in each seg_vector column:", np.sum(data[0]["obs"]["imagined_robot_pc_seg-gt"] == 1, axis=0))