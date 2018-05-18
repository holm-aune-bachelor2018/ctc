import matplotlib.pyplot as plt
import csv

filename = ""                       # Name of file
path = "" + filename + ".csv"       # Path to save image of graph
savepath = "" + filename + ".png"   # Name to save file

title = ""                          # Unique model name
y_max = 300                         # height y axis

val_loss = []
loss = []
wer = []

with open(path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        loss.append(row[1])
        val_loss.append(row[2])
        wer.append(row[3])

loss_temp = []
val_temp = []
wer_temp = []

#plt.xlim(0, 70)
#plt.ylim(0, y_max)

for i in range(1, len(loss)):
    loss_temp.append(float(loss[i]))
    val_temp.append(float(val_loss[i]))
    wer_temp.append(float(wer[i])*100)
plt.xlabel("Epochs")
plt.title(title)
plt.plot(loss_temp, label="loss", color='blue',linewidth=2)
plt.plot(val_temp, label="val_loss", color='red', linewidth=2)
plt.plot(wer_temp, label="wer", color='black', linewidth=2)
plt.grid()
plt.legend()
plt.savefig(savepath)
plt.show()
