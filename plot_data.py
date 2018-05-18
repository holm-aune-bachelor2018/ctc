import matplotlib.pyplot as plt
import csv
import argparse


def main(args):
    try:
        input_path = args.path
        savepath = args.save + ".png"
        title = args.title

        val_loss = []
        loss = []
        wer = []

        with open(input_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                loss.append(row[1])
                val_loss.append(row[2])
                wer.append(row[3])

        loss_temp = []
        val_temp = []
        wer_temp = []

        #plt.xlim(0, 70)
        #plt.ylim(0, 300)

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

        if(savepath):
            plt.savefig(savepath)
        plt.show()

    except (Exception) as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path of file to load')
    parser.add_argument('--save', type=str, help='Path to save graph as image')
    parser.add_argument('--title', type=str, help='Plot title')

    args = parser.parse_args()

    main(args)