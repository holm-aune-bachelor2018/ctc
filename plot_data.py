"""
LICENSE

This file is part of Speech recognition with CTC in Keras.
The project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.
The project is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this project.
If not, see http://www.gnu.org/licenses/.

"""

import argparse
import csv
import matplotlib.pyplot as plt


def main(args):
    try:
        plot_graph_from_csv(args.path, args.save, args.title)

    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print message


def plot_graph_from_csv(path, save, title, x_axis=100, y_axis=250):
    """
    Plots data from a csv file to a graph, and saves the graph as a .png
    :param path: Path to csv file
    :param save: Path, where to save graph
    :param title: Title of graph
    :param x_axis: Max value x-axis, Default=100
    :param y_axis: Max value y-axis, Default=250
    :return:
    """
    input_path = path
    save_path = save + ".png"
    title = title

    val_loss = []
    loss = []
    wer = []

    with open(input_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            loss.append(float(row[1]))
            val_loss.append(float(row[2]))
            wer.append(float(row[3]) * 100)

    plt.xlim(0, x_axis)
    plt.ylim(0, y_axis)

    plt.xlabel("Epochs")
    plt.title(title)
    plt.plot(loss, label="loss", color='blue', linewidth=2)
    plt.plot(val_loss, label="val_loss", color='red', linewidth=2)
    plt.plot(wer, label="wer", color='black', linewidth=2)
    plt.grid()
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path of file to load')
    parser.add_argument('--save', type=str, default="model_graph", help='Path to save graph as image')
    parser.add_argument('--title', type=str, default="Model graph",help='Plot title')

    args = parser.parse_args()

    main(args)

