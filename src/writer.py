from torch.utils.tensorboard import SummaryWriter

class CustomWriter(SummaryWriter):
    def __init__(self, logdir):
        super(CustomWriter, self).__init__(logdir)

    def log_training(self, train_loss, epoch):
        self.add_scalar('train_loss', train_loss, epoch)

    def log_evaluation(self, test_loss, cer, epoch):
        self.add_scalar('test_loss', test_loss, epoch)
        self.add_scalar('test_cer', cer, epoch)
