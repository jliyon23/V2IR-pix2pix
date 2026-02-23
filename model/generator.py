import functools

import torch
import torch.nn as nn

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu   = nn.ReLU(inplace=True)
        upnorm   = norm_layer(outer_nc)

        if outermost:

            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]

        elif innermost:

            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            model = [downrelu, downconv, uprelu, upconv, upnorm]

        else:

            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down  = [downrelu, downconv, downnorm]
            up    = [uprelu, upconv, upnorm]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)

        return torch.cat([x, self.model(x)], dim=1)

class UnetGenerator(nn.Module):

    def __init__(self, input_nc: int = 3, output_nc: int = 3,
                 num_downs: int = 8, ngf: int = 64,
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        super().__init__()

        block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8,
            innermost=True, norm_layer=norm_layer
        )

        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=block,
                norm_layer=norm_layer, use_dropout=use_dropout
            )

        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf,     ngf * 2, submodule=block, norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_generator(ngf: int = 64, use_dropout: bool = True) -> UnetGenerator:
    return UnetGenerator(
        input_nc=3, output_nc=3,
        num_downs=8, ngf=ngf,
        norm_layer=nn.BatchNorm2d,
        use_dropout=use_dropout,
    )
