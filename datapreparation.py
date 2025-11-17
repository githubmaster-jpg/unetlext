# create file
try:
    with open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/dataset.csv", "x") as file:
        file.write("")
except FileExistsError:
    print("File exists.")

# write to file
with open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/dataset.csv", "r+") as file:
    file.write("xrays,masks\n")
    xray_dir = "/Users/tiia-/Downloads/cnn/Tibia/data/xrays"
    mask_dir = "/Users/tiia-/Downloads/cnn/Tibia/data/masks"
    for img in range(150):
        file_list1 = [xray_dir, f"/{img}.png"]
        file_list2 = [mask_dir, f"/{img}.png"]
        file_path1 = "".join(file_list1)
        if img <= 100:
            file_path2 = "".join(file_list2)
        else:
            file_path2 = None
        file.write('{},{}\n'.format(file_path1, file_path2))
    file.close()

# split file
dataset = open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/dataset.csv", "r").readlines()
csv_names = ["test", "train", "val"]
for name in csv_names:
    with open(f"/Users/tiia-/Downloads/cnn/Tibia/data/csv/{name}.csv", "w") as file:
        file.write("xrays,masks\n")
        file.close()

for i in range(1, len(dataset)):
    if i <= 20:
        with open(f"/Users/tiia-/Downloads/cnn/Tibia/data/csv/{csv_names[2]}.csv", "a") as file:
            file.write(dataset[i])
            file.close()
    elif i > 20 and i <= 101:
        with open(f"/Users/tiia-/Downloads/cnn/Tibia/data/csv/{csv_names[1]}.csv", "a") as file:
            file.write(dataset[i])
            file.close()
    elif i > 101 and i <= 150:
        with open(f"/Users/tiia-/Downloads/cnn/Tibia/data/csv/{csv_names[0]}.csv", "a") as file:
            file.write(dataset[i])
            file.close()
    else:
        print("error")

# bar chart

"""trainset = open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/train.csv", "r").readlines()
valset = open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/val.csv", "r").readlines()
testset = open("/Users/tiia-/Downloads/cnn/Tibia/data/csv/test.csv", "r").readlines()

mainlen = len(dataset)-1
trainlen = len(trainset)-1
vallen = len(valset)-1
testlen = len(testset)-1

import matplotlib.pyplot as plt
labels = ["dataset", "test", "train", "val"]
data = [mainlen, testlen, trainlen, vallen]

plot, ax = plt.subplots()
bars = ax.bar(labels, data)
ax.bar_label(bars, label_type="edge", color="red")
plt.show()"""