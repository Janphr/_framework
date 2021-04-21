import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import base64
import matplotlib
import cv2
from time import sleep


matplotlib.use("TkAgg")
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          "text.color" : 'w'}
plt.rcParams.update(params)

class DynPlotter:
    def __init__(self, it_per_epoch, emit, thread_nr, lr, alpha, dropout):
        self.emit = emit
        self.plt = plt
        self.plt.ioff()
        self.fig, self.ax = self.plt.subplots(1, 1, figsize=(5, 5), facecolor='#2B2B2B')
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
        self.ax.set_ylabel("Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_facecolor('#000000')
        self.ax.grid()
        self.legend = None
        self.it_per_epoch = int(it_per_epoch)
        self.last_thresh = -1
        self.thread_nr = thread_nr
        self.default_title = 'Learning rate: ' + str(lr) + ' Alpha: ' + str(alpha) + ' Dropout: ' + str(dropout) + '\n' + \
                             ('' if thread_nr == -1 else 'Thread: ' + str(thread_nr) + ' ') + 'Epoch '

    def plt_dynamic(self, data):
        for i in range(len(data)):
            self.ax.plot(data[i][0], data[i][1], label=data[i][2] + str(data[i][0][-1:]))
            if self.legend is not None:
                self.legend.get_texts()[i].set_text(data[i][2] + str(data[i][0][-1:]))
        if self.legend is None:
            self.legend = self.ax.legend(loc="upper right", facecolor='#2B2B2B')
        self.fig.canvas.draw()
        # self.plt.pause(0.0001)
        emit_plot(self.fig, self.emit, self.thread_nr)

    # sets title corresponding to number of batches processed in this epoch
    def update_process(self, epoch, process):
        thresh = int((process / self.it_per_epoch) * 10)
        if thresh > self.last_thresh:
            self.last_thresh = thresh
            title = self.default_title + str(epoch) + ' ['
            for i in range(10):
                if i <= thresh:
                    title += '*'
                else:
                    title += '-'
            title += ']'
            self.ax.title.set_text(title)
            self.fig.canvas.draw()
            # self.plt.pause(0.0001)
            emit_plot(self.fig, self.emit, self.thread_nr)
        if self.last_thresh == 10:
            self.last_thresh = -1


def plotFeatures(input_data, targets, start_index, end_index, emit, labels):
    number_of_plots = end_index-start_index+1

    for i in range(number_of_plots):
        fig, axs = plt.subplots(1, 1,facecolor='#2B2B2B', figsize=(5, 3))
        axs.set_facecolor('#000000')
        axs.scatter(np.arange(0,len(targets)*0.03,0.03), input_data.T[start_index+i],
                    marker = '.', s = 1, c=np.array(["red", "blue", "green","white"])[targets])
        #axs.set_xlabel("Feature "+str(labels[start_index+i]))
        axs.set_xlabel("Feature "+str(start_index+i))
        axs.grid()
        axs.set_axisbelow(True)

        plt.tight_layout()
        fig.canvas.draw()
        emit_plot(fig, emit, id=i)

class Animations:
    def __init__(self):
        self.stop = False
        self.plt = plt

    def animate_feature(self, feature_x, feature_y, feature_name,emit):
        plot_len = 10
        step_size = 3
        for i in range(0,len(feature_x)-plot_len,step_size):
            # plt.close()
            if(self.stop):
                self.stop = False
                return
            fig, axs = self.plt.subplots(1, 1,facecolor='#2B2B2B', figsize=(5, 5))
            axs.scatter(feature_x[i:i+plot_len], feature_y[i:i+plot_len])#, marker = '.', s = 1)
            axs.grid()
            axs.title.set_text("Movement of " + feature_name + " over Time")
            self.plt.xlim([np.min(feature_x)-5, np.max(feature_x)+5])
            self.plt.ylim([np.min(feature_y)-5, np.max(feature_y)+5])
            self.plt.gca().invert_yaxis()
            self.plt.tight_layout()
            fig.canvas.draw()
            emit_plot(fig, emit, id=0)
            #sleep(0.1)




def plot(data, xlabel, ylabel, title, emit=None, id=None):
    fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 5), facecolor='#2B2B2B')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    i = 0
    for d in data:
        if len(data) > 1:
            a = ax[i]
        else:
            a = ax
        a.plot(d[0][0], 'r', label='train')
        if len(d[0]) > 1:
            a.plot(d[0][1], 'b', label='validation')
        a.grid()
        i += 1
    fig.suptitle(title, fontsize=16)
    fig.canvas.draw()
    emit_plot(fig, emit, id)

def plot_prediction_errors(correct_wrong_predictions, emit):
    fig, ax = plt.subplots(1, 1,facecolor='#2B2B2B', figsize=(5, 5))
    [correct_predictions, wrong_predictions] = correct_wrong_predictions

    for gesture_idx in range(len(correct_predictions)):
        plt.scatter(np.array(correct_predictions[gesture_idx])*0.03, np.full(len(correct_predictions[gesture_idx]), gesture_idx), c=np.full(len(correct_predictions[gesture_idx]), 'green'), marker = '.', s = 5)

    for gesture_idx in range(len(wrong_predictions)):
        plt.scatter(np.array(wrong_predictions[gesture_idx])*0.03, np.full(len(wrong_predictions[gesture_idx]), gesture_idx), c= np.full(len(wrong_predictions[gesture_idx]),3), marker = '.', s = 5)

    ax.set_ylabel("Gesture")
    ax.set_xlabel("Time [s]")
    ax.grid
    fig.suptitle("Prediction Error", fontsize=16)
    fig.canvas.draw()
    emit_plot(fig, emit, 67)

def heatmap(data, xlabel, ylabel, xtick, ytick, thread, emit):
    # sns.set(font_scale=.65)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#2B2B2B')
    g = sns.heatmap(data, annot=True, ax=ax, fmt=".0f", xticklabels=xtick, yticklabels=ytick)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=7, rotation='vertical', ha='center')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=7, rotation='horizontal', va='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.title.set_text(('Heatmap' if thread == -1 else 'Heatmap Thread: ' + str(thread)))
    ax.figure.tight_layout()
    fig.canvas.draw()
    emit_plot(fig, emit, -2)


def print_results(avg_tps, avg_confidence, correct_wrong_predictions, confusion_matrix, emit):
    emit('to_console', "Accuracy: " + str(avg_tps))
    emit('to_console', "Confidence: " + str(avg_confidence))

    np.set_printoptions(precision=4)
    for i in range(len(correct_wrong_predictions[0])):
        correct = len(correct_wrong_predictions[0][i])
        wrong = len(correct_wrong_predictions[1][i])

        # F1 Score
        precision = confusion_matrix[i][i] / np.sum(confusion_matrix[i])
        recall = confusion_matrix[i][i] / np.sum(confusion_matrix.T[i])
        f1_score = 2 * precision * recall / (precision + recall)
        emit('to_console',
             "Class " + str(i) + "\n" + str(correct) + " correct and " + str(wrong) + " wrong predictions -> " +
             str(100 * wrong / ((correct + wrong) or 1)) + " % wrong.")
        emit('to_console',
             "F1 Score of Class " + str(i) + ": " + str(f1_score))


def emit_plot(figure, emit, id):
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape((-1, 1))
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    frame = cv2.imencode('.png', data)[1]
    emit('plot', {'data': base64.b64encode(frame.tobytes()).decode('utf-8'), 'id': int(id)})

# for i in range(len(frames)):
#     frame = frames[i]
#     error = 0
#     for j in range(-check_range, check_range):
#         if i+j >= len(frames) or i+j < 0 or frames[i+j]-j == frame:
#             error += 1
#     error /= 2*check_range+1
#     if error >= 0.6:
#         frame_indices.append(frame/total_length)
#
# extract_frames("./data/demo_video.mp4", frame_indices)

# # plots Features incl end_index
# def plotFeaturesAsSubPlots(input_data, targets, start_index, end_index):
#     number_of_plots = end_index-start_index+1
#     fig, axs = plt.subplots(round(number_of_plots/2), 2,facecolor='#2B2B2B', figsize=(5, 5))
#
#     if number_of_plots > 2:
#         for i in range(int(number_of_plots/2)):
#             axs[i, 0].scatter(range(len(targets)), input_data.T[start_index+i*2], c=targets[0], marker = '.', s = 1)
#             axs[i, 1].scatter(range(len(targets)), input_data.T[start_index+i*2+1], c=targets[0], marker = '.', s = 1)
#             axs[i, 0].set_xlabel("Feature "+str(start_index+i*2))
#             axs[i, 1].set_xlabel("Feature "+str(start_index+i*2+1))
#             axs[i, 0].grid()
#             axs[i, 1].grid()
#
#         if((number_of_plots%2)==1):
#             i = round(.5 + number_of_plots/2)-1
#             axs[i, 0].scatter(range(len(targets)), input_data.T[start_index+i*2], c=targets[0], marker = '.', s = 1)
#             axs[i, 0].set_xlabel("Feature "+str(start_index+i*2))
#             axs[i, 0].grid()
#
#     else:
#         axs[0].scatter(range(len(targets)), input_data.T[start_index], c=targets[0], marker = '.', s = 1)
#         axs[number_of_plots-1].scatter(range(len(targets)), input_data.T[end_index], c=targets[0], marker = '.', s = 1)
#         axs[0].set_xlabel("Feature "+str(start_index))
#         axs[number_of_plots-1].set_xlabel("Feature "+str(end_index))
#         axs[0].grid()
#         axs[number_of_plots-1].grid()
#
#     #plt.tight_layout()
#     #plt.show()
#
#     return fig