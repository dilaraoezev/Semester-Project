"""Pytorch implementation of KANN."""

import matplotlib.pyplot as plt
import torch
import tqdm

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LagrangeKANN(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANN, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max
        """
        self.weight = torch.nn.parameter.Parameter(torch.empty(self.n_width, self.n_nodes, dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.weight)
        """
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros((self.n_width, self.n_nodes)).to(device)
        )

    def lagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1).to(device)

        p_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        ).to(device)

        for j in range(n_order + 1):
            p = 1.0
            for m in range(n_order + 1):
                if j != m:
                    p *= (x - nodes[m]) / (nodes[j] - nodes[m])
            p_list[:, :, j] = p

        return p_list

    def dlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1).to(device)

        dp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        ).to(device)

        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k = torch.ones_like(x).to(device) / (nodes[j] - nodes[i])
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k *= (x - nodes[m]) / (nodes[j] - nodes[m])
                    y += k
            dp_list[:, :, j] = y

        return dp_list

    def ddlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1).to(device)

        ddp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        ).to(device)

        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k_sum = 0.0
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k_prod = torch.ones_like(x).to(device) / (nodes[j] - nodes[m])
                            for n in range(n_order + 1):
                                if n != i and n != j and n != m:
                                    k_prod *= (x - nodes[n]) / (nodes[j] - nodes[n])
                            k_sum += k_prod
                    y += (1 / (nodes[j] - nodes[i])) * k_sum
            ddp_list[:, :, j] = y

        return ddp_list

    def to_ref(self, x, node_l, node_r):
        """Transform to reference base."""
        return (x - (0.5 * (node_l + node_r))) / (0.5 * (node_r - node_l))

    def to_shift(self, x):
        """Shift from real line to natural line."""
        x_shift = (self.n_nodes - 1) * (x - self.x_min) / (self.x_max - self.x_min)

        return x_shift

    def forward(self, x):
        """Forward pass for whole batch."""
        if len(x.shape) != 2:
            x = x.unsqueeze(-1)
            x = torch.repeat_interleave(x, self.n_width, -1)
        x_shift = self.to_shift(x)

        id_element_in = torch.floor(x_shift / self.n_order)
        id_element_in[id_element_in >= self.n_elements] = self.n_elements - 1
        id_element_in[id_element_in < 0] = 0

        nodes_in_l = (id_element_in * self.n_order).to(int)
        nodes_in_r = (nodes_in_l + self.n_order).to(int)

        x_transformed = self.to_ref(x_shift, nodes_in_l, nodes_in_r)
        delta_x = 0.5 * self.n_order * (self.x_max - self.x_min) / (self.n_nodes - 1)

        delta_x_1st = delta_x
        delta_x_2nd = delta_x**2

        phi_local_ikp = self.lagrange(x_transformed, self.n_order)
        dphi_local_ikp = self.dlagrange(x_transformed, self.n_order)
        ddphi_local_ikp = self.ddlagrange(x_transformed, self.n_order)

        phi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes)).to(device)
        dphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes)).to(device)
        ddphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes)).to(device)
        for sample in range(self.n_samples):
            for layer in range(self.n_width):
                for node in range(self.n_order + 1):
                    phi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        phi_local_ikp[sample, layer, node]
                    )
                    dphi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        dphi_local_ikp[sample, layer, node] / delta_x_1st
                    )
                    ddphi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        ddphi_local_ikp[sample, layer, node] / delta_x_2nd
                    )

        t_ik = torch.einsum("kp, ikp -> ik", self.weight, phi_ikp).to(device)
        dt_ik = torch.einsum("kp, ikp -> ik", self.weight, dphi_ikp).to(device)
        ddt_ik = torch.einsum("kp, ikp -> ik", self.weight, ddphi_ikp).to(device)

        return {
            "t_ik": t_ik,
            "dt_ik": dt_ik,
            "ddt_ik": ddt_ik,
            "phi_ikp": phi_ikp,
            "dphi_ikp": dphi_ikp,
            "ddphi_ikp": ddphi_ikp,
            "delta_x": delta_x,
        }


class KANN(torch.nn.Module):
    """KANN class with Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(KANN, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        self.inner = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max).to(device)
        self.outer = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max).to(device)

        total_params = sum(p.numel() for p in self.parameters())
        nodes_per_width = n_elements * n_order + 1
        print(f"\n\nTotal parameters: {total_params}")
        print(f"Order: {n_order}")
        print(f"Number of elements: {n_elements}")
        print(f"Nodes per width: {nodes_per_width}")
        print(f"Samples: {n_samples}\n\n")

        return None

    def forward(self, x):
        """Forward pass for whole batch."""
        x = self.inner(x)["t_ik"]
        x = self.outer(x)["t_ik"]

        x = torch.einsum("ik -> i", x)

        return x

    def residual(self, x):
        """Calculate residual."""
        y = 1 + x * self.forward(x)

        inner_dict = self.inner(x)
        outer_dict = self.outer(inner_dict["t_ik"])

        inner_dt_ik = inner_dict["dt_ik"]

        outer_t_ik = outer_dict["t_ik"]
        outer_dt_ik = outer_dict["dt_ik"]

        dy = torch.einsum("i, ik, ik -> i", x, outer_dt_ik, inner_dt_ik).to(device) + torch.einsum(
            "ik -> i", outer_t_ik
        ).to(device)

        residual = dy - y

        return residual

    def linear_system(self, x):
        """Compute Ax=b."""
        b = self.residual(x)

        inner_dict = self.inner(x)
        outer_dict = self.outer(inner_dict["t_ik"])

        inner_dt_ik = inner_dict["dt_ik"]
        inner_phi_ikp = inner_dict["phi_ikp"]
        inner_dphi_ikp = inner_dict["dphi_ikp"]

        outer_dt_ik = outer_dict["dt_ik"]
        outer_ddt_ik = outer_dict["ddt_ik"]
        outer_phi_ikl = outer_dict["phi_ikp"]
        outer_dphi_ikl = outer_dict["dphi_ikp"]

        A_outer = (
            torch.einsum("i, ikl, ik -> ikl", x, outer_dphi_ikl, inner_dt_ik).to(device)
            - torch.einsum("i, ikl -> ikl", x, outer_phi_ikl).to(device)
            + torch.einsum("ikl -> ikl", outer_phi_ikl).to(device)
        )

        A_inner = (
            torch.einsum(
                "i, ik, ik, ikp -> kp",
                x,
                outer_ddt_ik,
                inner_dt_ik,
                inner_phi_ikp,
            ).to(device)
            + torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_dphi_ikp).to(device)
            + torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp).to(device)
            - torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_phi_ikp).to(device)
        )

        return {"A_inner": A_inner, "A_outer": A_outer, "b": b}


def main():
    """Execute main routine."""
    n_width = 10
    n_order = 1
    n_samples = 5
    autodiff = True
    n_elements = int((n_samples - 2) / n_order)
    # n_elements = 100
    n_epochs = 100
    tol = 1e-3

    x_min = 0.0
    x_max = 1.0
    x_i = torch.linspace(x_min, x_max, n_samples).requires_grad_().to(device)

    y_i = torch.exp(x_i).to(device)

    model = KANN(
        n_width=n_width,
        n_order=n_order,
        n_elements=n_elements,
        n_samples=1,
        x_min=x_min,
        x_max=x_max,
    ).to(device)

    with tqdm.trange(n_epochs) as pbar1:
        for _ in pbar1:
            loss_epoch = torch.zeros((n_samples,)).to(device)
            for sample in range(n_samples):

                x = x_i[sample].unsqueeze(-1)

                system = model.linear_system(x)
                A_inner = system["A_inner"]
                A_outer = system["A_outer"]
                residual = system["b"]

                loss = torch.mean(torch.square(residual))
                if autodiff is True:
                    g_lst = torch.autograd.grad(
                        outputs=residual,
                        inputs=model.parameters(),
                        grad_outputs=torch.ones_like(residual),
                        retain_graph=True,
                        create_graph=False,
                    )

                    norm = torch.linalg.norm(torch.hstack(g_lst)) ** 2

                else:
                    g_lst = [A_inner, A_outer]

                    norm = (
                        torch.linalg.norm(A_inner) ** 2
                        + torch.linalg.norm(A_outer) ** 2
                    )

                for p, g in zip(model.parameters(), g_lst):
                    update = (residual / norm) * torch.squeeze(g)
                    p.data -= update

                loss_epoch[sample] = loss

            loss_mean = torch.mean(loss_epoch)
            loss_str = f"{loss_mean.item():0.4e}"

            pbar1.set_postfix(loss=loss_str)

            if loss_mean.item() < tol:
                break

    y_hat = torch.zeros_like(x_i).to(device)
    for sample in range(n_samples):
        x = x_i[sample].unsqueeze(-1)

        y_hat[sample] = 1 + x * model(x)

    l2 = torch.linalg.norm(y_i - y_hat)
    print(f"\nL2-error: {l2.item():0.4e}")
    """
    plt.figure(figsize=(6, 4), dpi=300)

    plt.plot(y_hat.detach().numpy(),  label="Analytical Solution", linewidth=2.5, color="black")
    plt.plot(y_i.detach().numpy(), '--', label="KANN prediction", linewidth=2, color="red")

    plt.xlabel("x", fontsize=14, fontweight='bold')
    plt.ylabel("T(x)", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc="upper left")

    plt.tight_layout()

    plt.savefig("KANN_vs_Analytical.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.savefig("KANN_vs_Analytical.pdf", format='pdf', dpi=300, bbox_inches='tight')
    """
    return None


if __name__ == "__main__":
    main()
