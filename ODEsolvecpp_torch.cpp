#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
/*PINN to Approximate Solution of ODEs using torch
Pde for solving Poisson Equation T'(x)-T(x)=0 on x ϵ [0,1].

Boundary conditions: T(0)=1;
*/
// Neural Network definition with torch
struct NeuralNet : torch::nn::Module {
    // Layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Tanh tanh{nullptr};

    // Constructor
    NeuralNet(int64_t input_dim, int64_t output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 16));
        fc2 = register_module("fc2", torch::nn::Linear(16, 16));
        fc3 = register_module("fc3", torch::nn::Linear(16, output_dim));
        tanh = register_module("tanh", torch::nn::Tanh());
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        auto x_ann = tanh(fc1(x));
        x_ann = tanh(fc2(x_ann));
        x_ann = fc3(x_ann);

        auto y = 1 + x * x_ann; // Boundary condition
        return y;
    }
};
//training function
struct PINNTrainer {
    NeuralNet& net;
    torch::optim::LBFGS optimizer;
    torch::Tensor grid;
    int n_int;
    float T0;
    //contrcut variables
    PINNTrainer(NeuralNet& network, int n_int_) 
        : net(network), optimizer(net.parameters(), torch::optim::LBFGSOptions(0.05).history_size(30)), n_int(n_int_), T0(1.0) {
        //create grid for the domain
        grid = torch::linspace(0, 1, n_int).unsqueeze(1);
    }

    torch::Tensor compute_pde_residual(torch::Tensor input_int) {
        input_int.set_requires_grad(true);
        auto predictions = net.forward(input_int);
        auto grad_outputs = torch::ones_like(predictions);
        auto grad_T = torch::autograd::grad({predictions.sum()}, {input_int}, /*grad_outputs=*/{grad_outputs}, /*retain_graph=*/true, /*create_graph=*/true)[0];
        auto grad_T_x = grad_T.select(1, 0);
        // T'-T=0
        auto residual = grad_T_x - predictions.squeeze();
        return residual;
    }

    torch::Tensor compute_loss(torch::Tensor inp_train_int) {
        auto residual_int = compute_pde_residual(inp_train_int);
        auto loss = torch::mean(residual_int.pow(2));
        return loss;
    }
    //Fit
    std::vector<double> fit(int num_epochs, bool verbose = true) {
        std::vector<double> history;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            if (verbose) {
                std::cout << "################################ " << epoch << " ################################" << std::endl;
            }

            optimizer.step([&]() {
                optimizer.zero_grad();
                auto loss = compute_loss(grid);
                loss.backward();
                history.push_back(loss.item<double>());
                return loss;
            });
            //print the loss
            if (verbose) {
                std::cout << "Epoch Loss: " << history.back() << std::endl;
            }
        }
        std::cout << "Final Loss: " << history.back() << std::endl;
        return history;
    }

    void save_results(const std::string& filename) {

        // generate inputs and predict the output
        torch::Tensor inputs = torch::linspace(0, 1, 1000).unsqueeze(1);
        torch::Tensor output = net.forward(inputs);
        torch::Tensor T_pred = output.squeeze();
        

        // Save inputsa nd predicted values to a CSV file
        std::ofstream file(filename);
        file << "x,Predicted_T\n";
        for (int i = 0; i < inputs.size(0); ++i) {
            file << inputs[i].item<double>() << "," << T_pred[i].item<double>() << "\n";
        }
        file.close();

        std::cout << "Results saved to " << filename << std::endl;
    }
};

int main() {
    // initialize the neural network
    int input_dim = 1;
    int output_dim = 1;
    NeuralNet neural_net(input_dim, output_dim);
    int n_int = 20;

    // an instance of the PINNTrainer class
    PINNTrainer trainer(neural_net, n_int);

    // train
    int num_epochs = 5;
    auto history = trainer.fit(num_epochs, true);

    // Save the results to a CSV file
    trainer.save_results("results.csv");

    return 0;
}