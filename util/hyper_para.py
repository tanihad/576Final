import argparse

class HyperParameters():
    def parse(self):
        parser = argparse.ArgumentParser()
        # training parameters
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch-size', type=int, default=16)
        parser.add_argument('--optim_str', type=str, default='SGD')
        parser.add_argument('--decay', help='Decay lr every <input> epochs')
        parser.add_argument('--wd-id', type=int, default=0, help='Weight decay id')
        
        # dataset params
        parser.add_argument('--mode', type=str, default='fast', help='Selective search mode')
        
        # model parameters
        parser.add_argument('--arch', type=str, default='resnet18', help='Architecture of the network: resnet50, resnet18')
        parser.add_argument('--bbox-reg', action=argparse.BooleanOptionalAction, default=False, help='Bbox reg')
        parser.add_argument('--freeze', action=argparse.BooleanOptionalAction, default=False, help='Freeze backbone')
        parser.add_argument('--extra-layers', action=argparse.BooleanOptionalAction, default=False, help='Extra layers')
        # db loader parameters
        #arser.add_argument('--workers', type=int, default=8, help='Num workers for the pytorch dataloader')


        self.args = vars(parser.parse_args())

        assert self.args['optim_str'] in {'Adam', 'SGD', 'rmsprop'}, 'Invalid optimizer'
        assert self.args['arch'] in {'resnet50', 'resnet18', 'resnet101'}
        assert self.args['mode'] in {'fast', 'quality'}

    def __getitem__(self, key):
        return self.args[key]

    def __str__(self):
        return str(self.args)