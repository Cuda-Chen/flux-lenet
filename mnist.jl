using Flux

function build_model(args; imgsize = (28, 28, 1), nclass = 10)
    cnn_output_size = Int.(floor.(imgsize[1]/8,imgsize[2]/8,32))

    return Chain(
    # First convolution, operating upon 28x28 image
    Conv((3, 3), imgsize[3]=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # second convolution, operating upon 14x14 image
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon 7x7 image
    Conv((3,3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(cnn_output_size), 10)
    )
end
