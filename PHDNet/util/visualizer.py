import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from pdb import set_trace as st
class Visualizer():
    def __init__(self, opt, phase):
        # self.opt = opt
        self.display_id = opt.display_id

        self.use_html = True

        self.isTrain = opt.isTrain

        self.phase = phase
       

        self.win_size = opt.display_winsize
        # self.name = opt.name
        self.name = opt.checkpoints_dir.split('/')[-1]
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(server = opt.display_server,port = opt.display_port)
            #self.ncols = opt.ncols
            self.ncols = opt.display_ncols
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            if self.phase == 'train':
                self.img_dir = os.path.join(self.web_dir, 'TrainImages')
            elif self.phase == 'test':
                    self.img_dir = os.path.join(self.web_dir, 'TestImages/', 'epoch_' + opt.epoch)
            else:
                self.img_dir = os.path.join(self.web_dir, 'RealCompositeImages')

            print('images are stored in {}'.format(self.img_dir))



            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.log_name_test = os.path.join(opt.checkpoints_dir, opt.name, 'test_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            ncols = self.ncols
            if self.ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                '''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                '''
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                #label_html = '<table>%s</table>' % label_html
                #self.vis.text(table_css + label_html, win = self.display_id + 2,
                #              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if self.isTrain:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                else:
                    img_path = os.path.join(self.img_dir, '%d.png' % (epoch))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data_train'):
            self.plot_data_train = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data_train['X'].append(epoch + counter_ratio)
        self.plot_data_train['Y'].append([errors[k] for k in self.plot_data_train['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data_train['X'])]*len(self.plot_data_train['legend']),1),
            Y=np.array(self.plot_data_train['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data_train['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    def plot_test_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id+10)
        message = '(epoch: %d)' %(epoch)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name_test, "a") as log_file:
            log_file.write('%s\n' % message)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, visuals, image_path, image_name):
        print('save images to', image_path, image_name)
        if not os.path.exists(image_path):
            os.makedirs(image_path, exist_ok=True)
        for label, image_numpy in visuals.items():
            util.save_image(image_numpy, os.path.join(image_path, image_name))



