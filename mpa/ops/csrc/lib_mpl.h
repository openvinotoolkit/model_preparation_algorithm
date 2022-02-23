void compute_weights(int size,
                     const torch::Tensor losses,
                     const torch::Tensor indices,
                     torch::Tensor weights,
                     float ratio,
                     float p);
