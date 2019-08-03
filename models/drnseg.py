from models import drn
import torch
import torch.nn as nn
import math

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name='drn_d_22', classes=529, pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        # if(pretrained):
            # a = torch.load('./checkpoints/drn_d_22-4bd2f8ea.pth')
            # model.load_state_dict(a)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)

        base_layers = list(model.children())[:-2]

        # hack in 4-channel initial layer
        # init_weight = 2.5*torch.mean(base_layers[0][0].weight, dim=1, keepdim=False)
        base_layers[0][0] = torch.nn.Conv2d(4,16, kernel_size=7, stride=1, padding=3, bias=False)
        # base_layers[0][0].weight.data[...] = 0
        # base_layers[0][0].weight.data[:,0,:,:] = init_weight

        self.base = nn.Sequential(*base_layers)
        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.reg = nn.Conv2d(model.out_dim, 2,
                             kernel_size=1, bias=True)

        if use_torch_up:
            self.up8_class = nn.UpsamplingBilinear2d(scale_factor=8)
            self.up8_reg = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up8_reg = nn.ConvTranspose2d(2, 2, 16, stride=8, padding=4,
                                    output_padding=0, groups=2,
                                    bias=False)
            fill_up_weights(up8_reg)
            up8_reg.weight.requires_grad = False
            self.up8_reg = up8_reg

            up8_class = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up8_class)
            up8_class.weight.requires_grad = False
            self.up8_class = up8_class

        self.upsample4_ = nn.Upsample(scale_factor=4, mode='nearest')
        self.softmax_ = nn.Softmax(dim=1)

    def softmax(self, input):
        return self.softmax_(input)

    def upsample4(self, input):
        return self.upsample4_(input)

    def forward(self, input_A, input_B, mask_B):
        base = self.base(torch.cat((input_A, input_B, mask_B),dim=1))

        out_cl = self.softmax_(self.seg(base.detach()))
        out_cl = self.up8_class(out_cl)

        out_reg = self.reg(base)
        out_reg = self.up8_reg(out_reg)

        return (out_cl, out_reg)
