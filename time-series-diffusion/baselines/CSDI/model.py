import torch 
import torch.nn as nn

class BasicErrorNet(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        time_dim = dim * 4
        fake_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(fake_dim),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.SiLU(), 
            nn.Linear(fake_dim, dim*2)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(dim, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, dim)
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(dim*2, fake_dim),
            nn.SiLU(),
            nn.Linear(fake_dim, dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )


    def forward(self, x, time):

        r = x.clone()

        x = self.state_mlp(x)
        x = torch.cat((x, r), dim=1)
        x = self.res_mlp(x)
        t = self.time_mlp(time)
        # t = rearrange(t, "b c -> b c 1")
        scale, shift = t.chunk(2, dim=1)
        x = x * (scale + 1) + shift

        x = self.final_mlp(x)

        return x