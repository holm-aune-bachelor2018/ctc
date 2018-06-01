import argparse
import csv
import matplotlib.pyplot as plt


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
            next(csv_reader)
            for row in csv_reader:
                loss.append(float(row[1]))
                val_loss.append(float(row[2]))
                wer.append(float(row[3])*100)

        plt.xlim(0, 100)
        plt.ylim(0, 250)

        plt.xlabel("Epochs")
        plt.title(title)
        plt.plot(loss, label="loss", color='blue',linewidth=2)
        plt.plot(val_loss, label="val_loss", color='red', linewidth=2)
        plt.plot(wer, label="wer", color='black', linewidth=2)
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
    