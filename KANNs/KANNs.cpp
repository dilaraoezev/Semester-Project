// Copyright [2024] <Dilara Özev>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
/*
Kolmogorov- Arnold Neural Network with auto dif and ODE

*/
torch::Dtype default_dtype = torch::kFloat64;


torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

// A KANN layer using n-th order Lagrange polynomials.
struct LagrangeKANN : torch::nn::Module {

  int64_t n_width, n_order, n_elements, n_nodes, n_samples;
  double x_min, x_max;
  torch::Tensor weight;

  // Constructor
  LagrangeKANN(int64_t n_width, int64_t n_order, int64_t n_elements,
               int64_t n_samples, double x_min, double x_max)
      : n_width(n_width), n_order(n_order), n_elements(n_elements),
        n_samples(n_samples), x_min(x_min), x_max(x_max) {

    n_nodes = n_elements * n_order + 1;

    weight = register_parameter(
        "weight", torch::zeros({n_width, n_elements * n_order + 1},
                               torch::TensorOptions().dtype(torch::kFloat64).device(device))); 
    /*
    weight = register_parameter("weight",
                torch::empty({n_width, n_nodes}, default_dtype));
    torch::nn::init::xavier_uniform_(weight);
    */
  }

  // Lagrange Polynomials
  torch::Tensor lagrange(torch::Tensor x, int64_t n_order) {

    torch::Tensor nodes = torch::linspace(-1.0, 1.0, n_order + 1,
                                          x.options().dtype(default_dtype).device(device)); 
    torch::Tensor p_list = torch::zeros({x.size(0), x.size(1), n_order + 1},
                                        x.options().dtype(default_dtype).device(device)); 

    for (int j = 0; j <= n_order; ++j) {
      torch::Tensor p =
          torch::full(x.sizes(), 1.0, x.options().dtype(default_dtype).device(device)); 
      for (int m = 0; m <= n_order; ++m) {
        if (j != m) {
          p *= (x - nodes[m]) / (nodes[j] - nodes[m]);
        }
      }
      p_list.index({torch::indexing::Slice(), torch::indexing::Slice(), j}) = p;
    }
    return p_list;
  }

  torch::Tensor dlagrange(torch::Tensor x, int64_t n_order) {
    torch::Tensor nodes = torch::linspace(-1.0, 1.0, n_order + 1,
                                          x.options().dtype(default_dtype).device(device)); 
    auto dp_list = torch::zeros({x.size(0), x.size(1), n_order + 1},
                                x.options().dtype(default_dtype).device(device)); 

    for (int j = 0; j <= n_order; ++j) {
      torch::Tensor y =
          torch::full(x.sizes(), 0.0, x.options().dtype(default_dtype).device(device)); 
      for (int i = 0; i <= n_order; ++i) {
        if (i != j) {
          auto k = torch::ones_like(x, x.options().dtype(default_dtype).device(device)) / 
                   (nodes[j] - nodes[i]);

          for (int m = 0; m <= n_order; ++m) {
            if (m != i && m != j) {
              k *= (x - nodes[m]) / (nodes[j] - nodes[m]);
            }
          }
          y += k;
        }
      }
      dp_list.index({torch::indexing::Slice(), torch::indexing::Slice(), j}) =
          y;
    }
    return dp_list;
  }

  torch::Tensor ddlagrange(torch::Tensor x, int64_t n_order) {
    torch::Tensor nodes = torch::linspace(-1.0, 1.0, n_order + 1,
                                          x.options().dtype(default_dtype).device(device));
    torch::Tensor ddp_list = torch::zeros({x.size(0), x.size(1), n_order + 1},
                                          x.options().dtype(default_dtype).device(device)); 

    for (int j = 0; j <= n_order; ++j) {
      torch::Tensor y =
          torch::full(x.sizes(), 0.0, x.options().dtype(default_dtype).device(device)); 
      for (int i = 0; i <= n_order; ++i) {
        if (i != j) {
          torch::Tensor k_sum =
              torch::full(x.sizes(), 0.0, x.options().dtype(default_dtype).device(device)); 
          for (int m = 0; m <= n_order; ++m) {
            if (m != i && m != j) {
              auto k_prod =
                  torch::ones_like(x, x.options().dtype(default_dtype).device(device)) / 
                  (nodes[j] - nodes[m]);
              for (int n = 0; n <= n_order; ++n) {
                if (n != i && n != j && n != m) {
                  k_prod *= (x - nodes[n]) / (nodes[j] - nodes[n]);
                }
              }
              k_sum += k_prod;
            }
          }
          y += (1.0 / (nodes[j] - nodes[i])) * k_sum;
        }
      }
      ddp_list.index({torch::indexing::Slice(), torch::indexing::Slice(), j}) =
          y;
    }
    return ddp_list;
  }
  torch::Tensor to_ref(torch::Tensor x, torch::Tensor node_l,
                       torch::Tensor node_r) {
    return (x - 0.5 * (node_l + node_r)) / (0.5 * (node_r - node_l));
  }

  torch::Tensor to_shift(torch::Tensor x, int64_t n_nodes, double x_min,
                         double x_max) {
    return (n_nodes - 1) * (x - x_min) / (x_max - x_min);
  }
  std::map<std::string, torch::Tensor> forward(torch::Tensor x) {

    if (x.dim() != 2) {
      x = x.unsqueeze(-1);
      x = torch::repeat_interleave(x, n_width, -1);
    }
    torch::Tensor x_shift = to_shift(x, n_nodes, x_min, x_max);

    auto id_element_in = torch::floor(x_shift / n_order);

    id_element_in.index_put_({id_element_in >= n_elements}, n_elements - 1);
    id_element_in.index_put_({id_element_in < 0}, 0);
    // id_element_in = torch::clamp(id_element_in, 0, n_elements - 1);

    torch::Tensor nodes_in_l = (id_element_in * n_order).to(torch::kInt);
    torch::Tensor nodes_in_r = (nodes_in_l + n_order).to(torch::kInt);

    torch::Tensor x_transformed = to_ref(x_shift, nodes_in_l, nodes_in_r);
    double delta_x = 0.5 * n_order * (x_max - x_min) / (n_nodes - 1);
    double delta_x_1st = delta_x;
    double delta_x_2nd = delta_x * delta_x;

    torch::Tensor phi_local_ikp = lagrange(x_transformed, n_order);
    torch::Tensor dphi_local_ikp = dlagrange(x_transformed, n_order);
    torch::Tensor ddphi_local_ikp = ddlagrange(x_transformed, n_order);

    auto phi_ikp =
        torch::zeros({n_samples, n_width, n_nodes}, torch::TensorOptions().dtype(default_dtype).device(device)); 
    auto dphi_ikp =
        torch::zeros({n_samples, n_width, n_nodes}, torch::TensorOptions().dtype(default_dtype).device(device)); 
    auto ddphi_ikp =
        torch::zeros({n_samples, n_width, n_nodes}, torch::TensorOptions().dtype(default_dtype).device(device)); 

    for (int sample = 0; sample < n_samples; ++sample) {
      for (int layer = 0; layer < n_width; ++layer) {
        for (int node = 0; node <= n_order; ++node) {

          int idx = nodes_in_l.index({sample, layer}).item<int>() + node;
          // std::cout << "Everything is fine until here, layer number is :"<<
          // layer << std::endl;
          phi_ikp.index_put_({sample, layer, idx},
                             phi_local_ikp.index({sample, layer, node}));

          dphi_ikp.index_put_({sample, layer, idx},
                              dphi_local_ikp.index({sample, layer, node}) /
                                  delta_x_1st);
          ddphi_ikp.index_put_({sample, layer, idx},
                               ddphi_local_ikp.index({sample, layer, node}) /
                                   delta_x_2nd);
        }
      }
    }
    torch::Tensor t_ik = torch::einsum("kp, ikp -> ik", {weight, phi_ikp}).to(device); 
    torch::Tensor dt_ik = torch::einsum("kp, ikp -> ik", {weight, dphi_ikp}).to(device); 
    torch::Tensor ddt_ik = torch::einsum("kp, ikp -> ik", {weight, ddphi_ikp}).to(device); 

    return {{"t_ik", t_ik},
            {"dt_ik", dt_ik},
            {"ddt_ik", ddt_ik},
            {"phi_ikp", phi_ikp},
            {"dphi_ikp", dphi_ikp},
            {"ddphi_ikp", ddphi_ikp},
            {"delta_x",
             torch::full({0}, delta_x, x.options().dtype(default_dtype).device(device))}}; 
  }
};
struct KANN : torch::nn::Module {

  int64_t n_width, n_order, n_elements, n_nodes, n_samples;
  double x_min, x_max;
  std::shared_ptr<LagrangeKANN> inner;
  std::shared_ptr<LagrangeKANN> outer;
  KANN(int64_t n_width, int64_t n_order, int64_t n_elements, int64_t n_samples,
       double x_min, double x_max)
      : n_width(n_width), n_order(n_order), n_elements(n_elements),
        n_samples(n_samples), x_min(x_min), x_max(x_max) {

    inner = std::make_shared<LagrangeKANN>(n_width, n_order, n_elements,
                                           n_samples, x_min, x_max);
    outer = std::make_shared<LagrangeKANN>(n_width, n_order, n_elements,
                                           n_samples, x_min, x_max);

    register_module("inner", inner);
    register_module("outer", outer);

    int64_t total_params = 0;
    for (const auto &param : this->parameters()) {
      total_params += param.numel();
    }

    int64_t nodes_per_width = n_elements * n_order + 1;

    std::cout << "\n\nTotal parameters: " << total_params << std::endl;
    std::cout << "Order: " << n_order << std::endl;
    std::cout << "Number of elements: " << n_elements << std::endl;
    std::cout << "Nodes per width: " << nodes_per_width << std::endl;
    std::cout << "Samples: " << n_samples << "\n\n" << std::endl;
  }

  torch::Tensor forward(torch::Tensor x) {

    x = inner->forward(x).at("t_ik");
    x = outer->forward(x).at("t_ik");

    x = torch::einsum("ik->i", {x});
    // x =  x.sum(dim=1);
    return x;
  }
  torch::Tensor residual(torch::Tensor x) {

    torch::Tensor y = 1 + x * forward(x);

    auto inner_dict = inner->forward(x);
    auto outer_dict = outer->forward(inner_dict.at("t_ik"));

    torch::Tensor inner_dt_ik = inner_dict.at("dt_ik");

    torch::Tensor outer_t_ik = outer_dict.at("t_ik");
    torch::Tensor outer_dt_ik = outer_dict.at("dt_ik");

    torch::Tensor dy =
        torch::einsum("i, ik, ik -> i", {x, outer_dt_ik, inner_dt_ik}).to(device) + 
        torch::einsum("ik -> i", {outer_t_ik}).to(device);
    /*
    torch::Tensor dy_first_term = (x.unsqueeze(1) * outer_dt_ik *
    inner_dt_ik).sum(1); torch::Tensor dy_second_term = outer_t_ik.sum(1);

    torch::Tensor dy = dy_first_term + dy_second_term;
    */

    torch::Tensor residual = dy - y;

    return residual;
  }

  std::map<std::string, torch::Tensor> linear_system(torch::Tensor x) {
    torch::Tensor b = residual(x);

    auto inner_dict = inner->forward(x);
    auto outer_dict = outer->forward(inner_dict.at("t_ik"));

    torch::Tensor inner_dt_ik = inner_dict.at("dt_ik");
    torch::Tensor inner_phi_ikp = inner_dict.at("phi_ikp");
    torch::Tensor inner_dphi_ikp = inner_dict.at("dphi_ikp");

    torch::Tensor outer_dt_ik = outer_dict.at("dt_ik");
    torch::Tensor outer_ddt_ik = outer_dict.at("ddt_ik");
    torch::Tensor outer_phi_ikl = outer_dict.at("phi_ikp");
    torch::Tensor outer_dphi_ikl = outer_dict.at("dphi_ikp");

    torch::Tensor A_outer =
        torch::einsum("i, ikl, ik -> ikl", {x, outer_dphi_ikl, inner_dt_ik}).to(device) - 
        torch::einsum("i, ikl -> ikl", {x, outer_phi_ikl}).to(device) + 
        torch::einsum("ikl -> ikl", {outer_phi_ikl}).to(device); 

    torch::Tensor A_inner =
        torch::einsum("i, ik, ik, ikp -> kp",
                      {x, outer_ddt_ik, inner_dt_ik, inner_phi_ikp}).to(device) + 
        torch::einsum("i, ik, ikp -> ikp", {x, outer_dt_ik, inner_dphi_ikp}).to(device) + 
        torch::einsum("ik, ikp -> ikp", {outer_dt_ik, inner_phi_ikp}).to(device) -
        torch::einsum("i, ik, ikp -> ikp", {x, outer_dt_ik, inner_phi_ikp}).to(device); 

    return {{"A_inner", A_inner}, {"A_outer", A_outer}, {"b", b}};
  }
};

/*
void save_data(const torch::Tensor& y_hat, const torch::Tensor& y_i, float l2) {
    std::ofstream file("data.csv");

    file << "y_hat,y_i\n";
    for (int i = 0; i < y_hat.size(0); ++i) {
        file << y_hat[i].item<double>() << ","
             << y_i[i].item<double>() << "\n";
    }

    file.close();

    // Save L2-error separately in another file
    std::ofstream l2_file("l2_error.txt");
    l2_file << l2;
    l2_file.close();
}
*/

int main() {
  int64_t n_width = 10;
  int64_t n_order = 1;
  int64_t n_samples = 5;
  int64_t n_elements = (n_samples - 2) / n_order;
  int64_t n_epochs = 100;
  bool autodif = true;
  double tol = 1e-3;

  double x_min = 0.0;
  double x_max = 1.0;
  torch::Tensor x_i = torch::linspace(x_min, x_max, n_samples).requires_grad_().to(device); 

  torch::Tensor y_i;
  y_i = torch::exp(x_i).to(device); 

  auto model =
      std::make_shared<KANN>(n_width, n_order, n_elements, 1, x_min, x_max);
  model->to(device); 

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    torch::Tensor loss_epoch = torch::zeros({n_samples}, torch::TensorOptions().dtype(default_dtype).device(device)); 
    for (int sample = 0; sample < n_samples; ++sample) {
      torch::Tensor x = x_i[sample].unsqueeze(0); 
      torch::Tensor residual, A_inner, A_outer;

      auto system = model->linear_system(x);
      residual = system.at("b");
      A_inner = system.at("A_inner");
      A_outer = system.at("A_outer");

      torch::Tensor loss = torch::mean(torch::square(residual));

      torch::Tensor norm;
      std::vector<torch::Tensor> g_lst;
      std::vector<torch::Tensor> outputs = {residual};
      if (autodif) {
        g_lst = torch::autograd::grad(outputs, model->parameters(), /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/false); 
        norm = torch::norm(torch::cat(g_lst)).pow(2);
      } else {
        g_lst = {A_inner, A_outer};
        norm = torch::norm(A_inner).pow(2) + torch::norm(A_outer).pow(2);
      }

      for (size_t i = 0; i < model->parameters().size(); ++i) {
        auto p = model->parameters()[i];
        auto g = (residual / norm) * torch::squeeze(g_lst[i]);
        p.data() -= g;
      }

      loss_epoch[sample] = loss;
    }

    double loss_mean = loss_epoch.mean().item<double>();
    std::cout << std::fixed << std::setprecision(4) << "Epoch: " << epoch
              << ", Loss: " << loss_mean << std::endl;

    if (loss_mean < tol) {
      break;
    }
  }

  // testing the model
  torch::Tensor y_hat = torch::zeros_like(x_i).to(device); 

  for (int sample = 0; sample < n_samples; ++sample) {
    torch::Tensor x = x_i[sample].unsqueeze(-1); 

    torch::Tensor model_output = model->forward(x);

    y_hat[sample] = 1 + x.item<double>() * model_output.item<double>();
  }

  auto l2 = torch::norm(y_i - y_hat);
  std::cout << "\nL2-error: " << l2.item<float>() << std::endl;

  /*
  // Lagrange test
  LagrangeKANN kann_layer(n_width, 2, n_elements, n_samples, x_min, x_max);

  torch::Tensor test_points = torch::linspace(-1.0, 1.0, 5,
  torch::dtype(default_dtype)); // Testing at -1, -0.5, 0, 0.5, 1 torch::Tensor
  lagrange_values = kann_layer.dlagrange(test_points.unsqueeze(0), 2);

  std::cout << "Lagrange polynomial values at test points:" << std::endl;
  std::cout << lagrange_values << std::endl;
  */
  // save_data(y_hat, y_i, l2.item<float>());

  return 0;
}
