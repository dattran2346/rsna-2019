import logging
from pathlib import Path
from terminaltables import AsciiTable
import time
import sys
import numpy as np


class TrainingLogger:
    ## TRAINING LOGER

    def __init__(self, args):
        ## Setup training logger
        # checkdir = Path(f'{args.checkdir}/{args.checkname}')
        self.args = args
        checkdir = Path(f'runs/{args.checkname}')
        checkdir.mkdir(exist_ok=True, parents=True)

        self.terminal = sys.stdout
        if args.resume:
            mode = 'a' # append to prev
        else:
            mode = 'w+' # open and write new
        self.file = open(checkdir/"training.log", mode)

        ## Print training setting
        

        if self.args.start_epoch == 0:

            # only print log at the start of training
            self.write_all('='*20 + 'TRAINING START' + '='*20)
            self.write_all('\n\n')

            table_data = [['Setting', 'Value']]
            table_data.append(['Backbone', args.backbone])
            table_data.append(['Image size', args.image_size])
            table_data.append(['Fold', args.fold])
            table_data.append(['Epochs', args.epochs])
            table_data.append(['Warmup', args.warmup])
            table_data.append(["Batch size", args.batch_size])
            table_data.append(['Gradient accumulation', args.gds])
            table_data.append(['Lr', args.lr])
            table_data.append(['Optimizer', args.optim])
            table_data.append(['Scheduler', args.sched])
            table_data.append(['Mix window', args.mix_window])
            table_data.append(['#slies', args.nslices])
            table_data.append(['LSTM dropout', args.lstm_dropout])
            table_data.append(['Conv LSTM', args.conv_lstm])
            table_data.append(['Cut block', args.cut_block])
            table_data.append(['Dropblock', args.drop_block])
            table_data.append(['Auxilary', args.aux])
            table = AsciiTable(table_data)

            self.write_all(table.table)
            self.write_all('\n\n')
            # ['any', 'epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']
            self.file.write(f'Epoch |    LR    | Train Loss -  Any       -  Epidural  -  Subdural  -  Subara    -  Intraparen-  Intraven  | Valid Loss -  Any       -  Epidural  -  Subdural  -  Subara    -  Intraparen-  Intraven  | Time \n')
            self.file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    def write_all(self, log):
        self.terminal.write(f'{log}\n')
        self.file.write(f'{log}\n')

    def on_epoch_start(self, epoch):
        # record epoch start time
        self.start_time = time.time()
        self.terminal.write(f'EPOCH: {epoch}\n')

    def on_epoch_end(self, epoch, lr, train_losses, val_losses):

        ## Write log to file
        # train_time = (time.time() - self.start_time) / 60
        train_time = np.random.uniform(0) * 10
        train_loss = train_losses.mean()
        val_loss = val_losses.mean()

        self.file.write(f'{epoch:5d} |{lr:0.8f}| {train_loss:8.8f} - {train_losses[0]:8.8f} - {train_losses[1]:8.8f} - {train_losses[2]:8.8f} - {train_losses[3]:.8f} - {train_losses[4]:.8f} - {train_losses[5]:8.8f} | {val_loss:8.8f} - {val_losses[0]:8.8f} - {val_losses[1]:8.8f} - {val_losses[2]:8.8f} - {val_losses[3]:8.8f} - {val_losses[4]:8.8f} - {val_losses[5]:8.8f} | {train_time:3.1f}min\n')

        ## Write log to terminal
        self.terminal.write(f'Traininng loss: {train_loss:.5f}\n')
        self.terminal.write(f'Validation loss: {val_loss:.5f}\n')

    def on_training_end(self, best_loss, best_epoch):
        self.file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        self.write_all('\n\n')
        msg = f'====== Finish training, best loss {best_loss:.5f}@e{best_epoch+1} ======'
        self.file.write(msg)
        self.terminal.write(msg)

if __name__ == "__main__":
    ### Test logger
    from options import Options
    import numpy as np
    opt = Options()
    args = opt.parse()
    args.checkname = 'logger_test'
    args.checkdir = 'test'

    logger = TrainingLogger(args)

    for e in range(10):
        logger.on_epoch_start(e)

        train_loss, lr = np.random.uniform(0), np.random.uniform(0)
        val_loss = np.random.uniform(0)

        logger.on_epoch_end(e, lr, train_loss, val_loss)

    logger.on_training_end(0.1, 9)
