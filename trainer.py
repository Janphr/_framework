from .utils.evaluations import *


class Trainer():
    def __init__(self, network, loss, optimizer, dropout=1):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.stop = False
        self.current_epoch = 0
        self.dropout = dropout

    def train_batch(self, x, target):
        # go through all layers in forward pass of the network
        y, backward = self.network.forward(x)
        # take the prediction (y) and evaluate with the target
        loss, delta = self.loss(y, target)
        # take the gradient of the evaluation (delta)
        # and go through all backward functions to update the weights and biases
        backward(delta)
        # go through the tensor references of the layers in the network and update the weights and biases
        self.network.update(self.optimizer)
        return loss

    def train(self, x, target, epochs, batch_size, live_eval=None, dyn_plot=False, thread_nr=-1, emit=None):
        losses = []
        validation_losses = []
        dyn_plotter = None
        if dyn_plot:
            dyn_plotter = DynPlotter(
                len(x) / batch_size, emit, thread_nr, self.optimizer.lr, self.optimizer.alpha, self.dropout)

        for epoch in range(epochs):
            self.current_epoch = epoch
            # creates an array of indices of input length and shuffles them.
            # this way each batch in each epoch is different
            p = np.random.permutation(len(x))
            loss = 0
            for i in range(0, len(x), batch_size):
                # get the random input values and their targets and send them through the network
                x_batch = x[p[i: i + batch_size]]
                target_batch = target[p[i: i + batch_size]]
                loss += self.train_batch(x_batch, target_batch)
                if dyn_plotter:
                    dyn_plotter.update_process(epoch, i / batch_size)
            # since loss is the sum of all batches
            loss = loss * (batch_size / len(x))
            losses.append(loss)
            emit('to_console', 'Epoch ' + str(epoch + 1) + (
                ' of thread ' + str(thread_nr) if thread_nr != -1 else '') + ' loss: ' + str(round(loss, 4)))
            if live_eval:
                y, b = self.network.forward(live_eval[0])
                l, d = self.loss(y, live_eval[1])
                validation_losses.append(l)
                if dyn_plotter:
                    dyn_plotter.plt_dynamic([(losses, 'r', "train: "), (validation_losses, 'b', "validation: ")])
                # if thread_nr != -1:
                    # plot([[(losses, l)]], "epochs", "loss", "Thread" + str(thread_nr), emit, thread_nr)
                emit('save_best', {'val_loss': l, 'train_loss': loss, 'epoch': epoch, 'thread': int(thread_nr)})
            if self.stop:
                break
        return losses, validation_losses

    def get_network(self):
        return self.network

    def get_optimizer(self):
        return self.optimizer

    def get_current_epoch(self):
        return self.current_epoch

    def stop_run(self):
        self.stop = True
