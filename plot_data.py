import matplotlib.pyplot as plt
import csv

path = ""       #path to csv file
y_max = 500     # height y axis

val_loss = []
loss = []

with open(path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        loss.append(row[1])
        val_loss.append(row[2])

loss_temp = []
val_temp = []

plt.xlim(0, len(loss))
plt.ylim(0, y_max)

for i in range(1, len(loss)):
    loss_temp.append(loss[i])
    val_temp.append(val_loss[i])

plt.plot(loss_temp)
plt.plot(val_temp)

plt.show()
