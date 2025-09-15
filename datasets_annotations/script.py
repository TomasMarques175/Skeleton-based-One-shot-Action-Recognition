# Script to modify dataset annotations by removing certain labels and remapping others

input_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_train(old).txt"
output_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_train.txt"
child_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_childs_val.txt"
therapist_file = "d:/The God Folder/Tese/Skeleton-based-One-shot-Action-Recognition-mp/datasets_annotations/mp_therapists_val.txt"

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

with open(input_file, "r") as fin, \
     open(output_file, "w") as fout_all, \
     open(child_file, "w") as fout_child, \
     open(therapist_file, "w") as fout_therapist:

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
            continue  # skip unwanted labels

        # remap label
        label = label_map.get(label, label)

        # shift label by +1
        label = label + 1

        # write to main file
        fout_all.write(f"{path} {label}\n")

        # also write to child/therapist files depending on path
        if "child" in path.lower():
            fout_child.write(f"{path} {label}\n")
        elif "therapist" in path.lower():
            fout_therapist.write(f"{path} {label}\n")
