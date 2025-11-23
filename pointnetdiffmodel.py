import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- choose your maximum control net size (<=10x10) -----
MAX_CTRLS_U = 10
MAX_CTRLS_V = 10
CTRL_DIM     = 3     # x,y,z only (weights optional; see notes)
DEG_P        = 3
DEG_Q        = 3

# For SurfEval sampling resolution (match your eval grid)
OUT_U = 35
OUT_V = 35

# If you want to predict knots: lengths are m+p+1 and n+q+1 (clamped/open)
KNOTU_LEN = MAX_CTRLS_U + DEG_P + 1
KNOTV_LEN = MAX_CTRLS_V + DEG_Q + 1


def mlp(layers, bn=True, act=nn.ReLU):
    seq = []
    for i in range(len(layers)-1):
        seq += [nn.Linear(layers[i], layers[i+1])]
        if i < len(layers)-2:
            if bn:
                seq += [nn.BatchNorm1d(layers[i+1])]
            if act is not None:
                seq += [act(inplace=True)]
    return nn.Sequential(*seq)


class KnotHead(nn.Module):
    """
    Predict *nonnegative* deltas -> cumulative -> [0,1] normalized clamped knots.
    Uses softplus for positivity and tiny epsilon for numerical stability.
    """
    def __init__(self, in_ch, len_u=KNOTU_LEN, len_v=KNOTV_LEN):
        super().__init__()
        self.head = mlp([in_ch, 256, 128, len_u + len_v])
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)
        self.eps = 1e-4
        self.len_u = len_u
        self.len_v = len_v

    def _delta_to_knots(self, deltas):
        # deltas: (B, L) nonnegative
        cum = torch.cumsum(torch.clamp(deltas, min=self.eps), dim=-1)
        # normalize to [0,1]
        knots = (cum - cum[:, :1]) / (cum[:, -1:] - cum[:, :1] + 1e-8)
        # (Optional) explicitly clamp ends to 0 and 1 to emulate open clamping
        knots[:, 0]  = 0.0
        knots[:, -1] = 1.0
        return knots

    def forward(self, g):
        raw = self.head(g)                                        # (B, Ku+Kv)
        raw_u, raw_v = raw[..., :self.len_u], raw[..., self.len_u:]
        deltas_u = self.softplus(raw_u)
        deltas_v = self.softplus(raw_v)
        knot_u = self._delta_to_knots(deltas_u)                   # (B, Ku)
        knot_v = self._delta_to_knots(deltas_v)                   # (B, Kv)
        return knot_u, knot_v


class ControlNetHead(nn.Module):
    """
    Predict (Mu x Mv x 3) control net from the global PointNet++ feature.
    """
    def __init__(self, in_ch, Mu=MAX_CTRLS_U, Mv=MAX_CTRLS_V, dim=CTRL_DIM):
        super().__init__()
        self.Mu, self.Mv, self.dim = Mu, Mv, dim
        out_ch = Mu * Mv * dim
        self.head = mlp([in_ch, 512, 256, 128, out_ch])

    def forward(self, g):
        B = g.shape[0]
        ctrl = self.head(g)                                       # (B, Mu*Mv*dim)
        ctrl = ctrl.view(B, self.Mu, self.Mv, self.dim)
        return ctrl


# --------- PointNet++ backbone (MSG) ----------
# You already imported these utilities; using the same APIs.
# If you prefer your existing classes, you can swap this block out 1:1.
try:
    from pointnet2_utils import PointNetSetAbstractionMsg
except ImportError:
    # fallback: import from your local file if needed
    from pointnet2_utils import PointNetSetAbstractionMsg

class PointNet2Backbone(nn.Module):
    """
    A compact PointNet++ MSG encoder producing a global feature vector.
    Input: (B, N, 3)
    Output: (B, Cg)
    """
    def __init__(self, in_ch=3, g_ch=512):
        super().__init__()
        # 3 levels; adjust radii/K for your data scale (assumes normalized cloud)
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1, 0.2],
            nsample_list=[16, 32, 64],
            in_channel=0,                     # we only pass xyz, no extra feats
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 64],
            in_channel=64+128+128,           # concat from SA1
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=1,
            radius_list=[0.4, 0.8, 1.2],
            nsample_list=[32, 64, 128],
            in_channel=128+256+256,
            mlp_list=[[256, 256, 512], [256, 512, 512], [256, 512, 512]]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(512+512+512, g_ch, 1),
            nn.BatchNorm1d(g_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz):
        # xyz: (B, N, 3)
        xyz_in = xyz.transpose(1, 2).contiguous()   # (B, 3, N)
        l1_xyz, l1_points = self.sa1(xyz_in, None)  # (B, 3, 512), (B, C1, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # (B, 3, 1), (B, C3, 1)
        g = self.proj(l3_points).squeeze(-1)        # (B, g_ch)
        return g


# ---------- Full regressor ----------
class PointNet2SurfaceRegressor(nn.Module):
    """
    Encodes (B, N, 3) samples → predicts control net (Mu x Mv x 3)
    and optional knots. Designed to work with your SurfEval layer.
    """
    def __init__(self,
                 Mu=MAX_CTRLS_U, Mv=MAX_CTRLS_V,
                 predict_knots=True,
                 g_ch=512):
        super().__init__()
        self.Mu, self.Mv = Mu, Mv
        self.predict_knots = predict_knots

        self.backbone = PointNet2Backbone(g_ch=g_ch)
        self.ctrl_head = ControlNetHead(g_ch, Mu, Mv, CTRL_DIM)

        if predict_knots:
            self.knot_head = KnotHead(g_ch, len_u=Mu+DEG_P+1, len_v=Mv+DEG_Q+1)
        else:
            self.knot_head = None

        # Optional: tame initial outputs (helps early training stability)
        self.ctrl_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, xyz):
        """
        xyz: (B, N, 3)
        Returns:
            ctrl:   (B, Mu, Mv, 3)
            knot_u: (B, Mu+p+1) or None
            knot_v: (B, Mv+q+1) or None
        """
        g = self.backbone(xyz)                       # (B, g_ch)
        ctrl = self.ctrl_head(g) * self.ctrl_scale   # (B, Mu, Mv, 3)

        if self.predict_knots:
            knot_u, knot_v = self.knot_head(g)
        else:
            knot_u = knot_v = None

        return ctrl, knot_u, knot_v
