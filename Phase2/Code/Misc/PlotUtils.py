import cv2


class Logger:
    """ Class to log the training process

    Args:
        log_dir (str): path to the log directory

    Attributes:
        log_dir (str): path to the log directory
        plot_dir (str): path to the plot directory
        media_dir (str): path to the media directory
        log_file_path (str): path to the log file
        log_file (file): log file
        train_data (list): list of training data
        val_data (list): list of validation data

    Methods:
        log(tag, **kwargs): log the data
        plot(data, name, path): plot the data
        plot_both(data1, data2, name, path): plot the data
        draw(epoch, img): draw the image
    """

    import os
    import time
    import matplotlib.pyplot as plt

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not self.os.path.exists(log_dir):
            self.os.makedirs(log_dir)

        self.plot_dir = self.os.path.join(log_dir, 'plots')
        if not self.os.path.exists(self.plot_dir):
            self.os.mkdir(self.plot_dir)

        self.media_dir = self.os.path.join(log_dir, 'media')
        if not self.os.path.exists(self.media_dir):
            self.os.mkdir(self.media_dir)

        self.log_file_path = self.log_dir + '/logs.txt'
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write('Logs date and time: '+self.time.strftime("%d-%m-%Y %H:%M:%S")+'\n\n')

        self.train_data = []
        self.val_data = []
        self.train_acc = []
        self.val_acc = []

    def log(self, tag, **kwargs):
        """ Log the data

        Args:
            tag (str): tag for the data
            **kwargs: data
        """

        self.log_file = open(self.log_file_path, 'a')

        if tag == 'args':
            self.log_file.write('Training Args:\n')
            for k, v in kwargs.items():
                self.log_file.write(str(k)+': '+str(v)+'\n')
            self.log_file.write('#########################################################\n\n')
            self.log_file.write(f'Starting Training... \n')

        elif tag == 'train':
            self.train_data.append([kwargs['loss']])
            self.train_acc.append([kwargs['acc']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Train Loss: {kwargs["loss"]} \t Train Accuracy: {kwargs["acc"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'test':
            self.val_data.append([kwargs['loss']])
            self.val_acc.append([kwargs['acc']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Test Loss: {kwargs["loss"]} \t Test Accuracy: {kwargs["acc"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'model':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving best model... Val Loss: {kwargs["loss"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'plot_loss':
            self.plot(self.train_data, name='Train Loss', path=self.plot_dir)
            self.plot(self.val_data, name='Test Loss', path=self.plot_dir)
            self.plot_both(self.train_data, self.val_data, name='Loss', path=self.plot_dir)

        elif tag == 'plot_acc':
            self.plot(self.train_acc, name='Train Acc', path=self.plot_dir)
            self.plot(self.val_acc, name='Test Acc', path=self.plot_dir)
            self.plot_both(self.train_acc, self.val_acc, name='Accuracy', path=self.plot_dir)


        self.log_file.close()

    def draw(self, epoch, img):
        """ Draw the image

        Args:
            epoch (int): epoch
            img (np.ndarray): image
        """

        cv2.imwrite(self.media_dir+'/'+str(epoch)+'.png', img)

    def plot(self, data, name, path):
        """ Plot the data

        Args:
            data (list): data
            name (str): name of the data
            path (str): path to the plot
        """

        self.plt.plot(data)
        self.plt.xlabel('Epochs')
        self.plt.ylabel(name)
        self.plt.title(name+' vs. Epochs')
        self.plt.savefig(self.os.path.join(path, name+'.png'), dpi=1200 ,bbox_inches='tight')
        self.plt.close()

    def plot_both(self, data1, data2, name, path):
        """ Plot data1 and data2 in the same plot

        Args:
            data1 (list): data1
            data2 (list): data2
            name (str): name of the data
            path (str): path to the plot
        """
        if name =='Loss':
            label1 = 'Train Loss'
            label2 = 'Test Loss'
        elif name =='Accuracy':
            label1 = 'Train Acc'
            label2 = 'Test Acc'

        self.plt.plot(data1, label=label1)
        self.plt.plot(data2, label=label2)
        self.plt.xlabel('Epochs')
        self.plt.ylabel(name)
        self.plt.title(name+' vs. Epochs')
        self.plt.legend()
        self.plt.savefig(self.os.path.join(path, name+'.png'), dpi=1200 ,bbox_inches='tight')
        self.plt.close()