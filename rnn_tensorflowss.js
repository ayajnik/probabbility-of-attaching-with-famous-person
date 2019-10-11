//importing the library
import * as tf from ‘@tensorflow/tfjs’;
async function trainModel(inputs, outputs, size, window_size, n_epochs, learning_rate, n_layers, callback)
{   

    const input_layer_shape  = window_size;
    const input_layer_neurons = 100;//creating the input layer
                                    //using one dense layer with two-dimensional input shape as an input layer of the entire network

    const rnn_input_layer_features = 10;
    const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
   
    const rnn_input_shape  = [ rnn_input_layer_features, rnn_input_layer_timesteps ];
    const rnn_output_neurons = 20;

    const rnn_batch_size = window_size;
 
    const output_layer_shape = rnn_output_neurons;
    const output_layer_neurons = 1;

    //initializing the model
    const model = tf.sequential();

    inputs = inputs.slice(0, Math.floor(size / 100 * inputs.length));
    outputs = outputs.slice(0, Math.floor(size / 100 * outputs.length));

    //using tensors
    //as we have seen in the above two lines, we have divided the inputs and outputs by 10 so as to perform normalization in order to tackle the problem of 
    // outliers. So, reshaping the values between [0,1]
    
    const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10));
    const ys = tf.tensor2d(outputs, [outputs.length, 1]).reshape([outputs.length, 1]).div(tf.scalar(10));

    model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
    model.add(tf.layers.reshape({targetShape: rnn_input_shape}));

    var lstm_cells = [];
    for (let index = 0; index < n_layers; index++) {
         lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));           
    }

    model.add(tf.layers.rnn({cell: lstm_cells,
 inputShape: rnn_input_shape, returnSequences: false}));

    model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));

    const opt_adam = tf.train.adam(learning_rate);
    model.compile({ optimizer: opt_adam, loss: 'meanSquaredError'});

    const hist = await model.fit(xs, ys,
 { batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
     onEpochEnd: async (epoch, log) => { callback(epoch, log); }}});

    return { model: model, stats: hist };
}

function Predict(inputs, size, model)
{
    var inps = inputs.slice(Math.floor(size / 100 * inputs.length), inputs.length);
    const outps = model.predict(tf.tensor2d(inps, [inps.length,
 inps[0].length]).div(tf.scalar(10))).mul(10);

    return Array.from(outps.dataSync());
}

//according to the basic architecture of RNN the input of the following NN is a 3dimensional 
//tensor having the following shape is the actual number of samples passed from the output of the input dense layer to the correspondent
//inputs of RNN's timestep. The third dimension is the number of features in each sample.

//as the input layer is in one dimensional tensors of values, so to pass the following tensor
//values to the inputs of RNN we need to transform the structure of this data into 3-D tensors.
//So, we are going to "reshape the layer" which performs no computation.




//rnn_input_shape is the target shape for the specific dense layer output data transformation.
//the number of neurons in the input dense layer is divided by the number of features in each sample
//passed to the input of RNN to obtain the values of time steps during which the RNN is recursively
//trained. So, the following target shape is [input_layer_neuron, 10, 10].

//since we have computed the target shape for the RNN input, we're now appending the re-shape 
//layer to the model being constructed. So, during the process of learning and predicting the values
//by all computation means the following layer will transform data passed from the outputs of input dense layer to
//the input of RNN layer.



//the input shape of the RNN is as follows, the first argument is array of LASTM cells. The
//second argument is the RNN's input shape and the last one is being created which is used to specify
//if the RNN should output 3-D tensor of outputs. In our model we are passing the output of 
//RNN to another dense output layer, we must set the value of the argument as FALSE.

