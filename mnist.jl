using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON

function build_model(args; imgsize = (28, 28, 1), nclasses = 10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)

    return Chain(
    # First convolution, operating upon 28x28 image
    Conv((5, 5), imgsize[end]=>6, relu),
    MaxPool((2, 2)),

    # second convolution, operating upon 14x14 image
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),

    # Third convolution, operating upon 7x7 image
    #Conv((3,3), 32=>32, pad=(1,1), relu),
    #MaxPool((2,2)),

    flatten,
    Dense(prod(out_conv_size), 120, relu),
    Dense(120, 84, relu),
    Dense(84, nclasses)
    )
end

function prepare_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end
