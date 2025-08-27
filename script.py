# Script to modify dataset annotations by removing certain labels and remapping others

input_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_train(old).txt"
output_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_train.txt"

# Labels to remove
remove_labels = {6, 9, 11}

# Label mapping
label_map = {
    3: 4,
    4: 5,
    5: 8,
    7: 9,
    8: 10,
    10: 12,
    12: 13
}

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # skip malformed lines
        path, label_str = parts
        try:
            label = int(label_str)
        except ValueError:
            continue  # skip lines with non-integer labels
        if label in remove_labels:
            continue  # skip lines with labels to remove
        label = label_map.get(label, label)  # map label if needed
        fout.write(f"{path} {label}\n")
