"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This is a predictor for autoregressive RNNs constructed in tensorflowrol
This predictor is good only for one control input being first net input, all other net inputs in the same order
as net outputs, and all net outputs being closed loop, no dt, no target position
horizon cannot be changed in runtime
"""


"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN - it make take quite a bit of time
    During initialization you only need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_net
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optim
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN. It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

#TODO: for the moment it is not possible to update RNN more often than mpc dt
#   Updating it more often will lead to false results.

# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Network import get_internal_states, load_internal_states
from SI_Toolkit.load_and_normalize import *

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

import numpy as np

from types import SimpleNamespace
import yaml, os
import pickle

import tensorflow as tf

#import SNN.snn_dumy as snn
import SNN.snn_lmu as snn

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'), Loader=yaml.FullLoader)

NET_NAME = config['modeling']['NET_NAME']
PATH_TO_MODELS = config["paths"]["PATH_TO_EXPERIMENT_RECORDINGS"] + config['paths']['path_to_experiment'] + "Models/"

#PATH_TO_SNN_WEIGHTS = r'.\SNN\pre_trained_weights\weights_latest_LMU2.npy'
#PATH_TO_SNN_WEIGHTS = r'.\SNN\pre_trained_weights\weights_latest_LMU3.npy'
#PATH_TO_SNN_WEIGHTS = r'.\SNN\pre_trained_weights\model_weights_new.npy'      # Uses Predictor_LMU2
PATH_TO_SNN_WEIGHTS = r'.\SNN\pre_trained_weights\model_weights_new_forealyo_good_stuff.npy'
PATH_TO_STATE = r'.\SNN\pre_trained_weights\model_state.pkl'

class predictor_autoregressive_SNN:
    def __init__(self, horizon=None, batch_size=None, net_name=None):

        a = SimpleNamespace()
        self.batch_size = batch_size
        self._horizon = None  # Helper variable for horizon settoer
        self.horizon = horizon
        a.path_to_models = PATH_TO_MODELS

        a.net_name = net_name

        with open(PATH_TO_STATE, "rb") as f:
            model_state = pickle.load(f)
        self.scales = model_state["scales"]
        #print(self.scales)

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net_info = snn.NetInfo()

        samp_freq = 50  # cartpole data is recorded at ~50Hz
        dt = 0.01  # nengo time step
        learning_rate = 0  # lr
        t_delay = 0.02  # how far to predict the future (initial guess)
        seed = 4  # to get reproducible neuron properties across runs
        lmu_theta = 0.1  # duration of the LMU delay
        lmu_q = 5  # number of factorizations per dim in LMU

        #neurons_per_dim = 100  # number of neurons representing each dimension
        neurons_per_dim = 200

        #weights = 0.00003*np.ones((len(self.net_info.outputs), n_neurons)) # Weights should be read from file (pre-trained model)
        weights = np.load(PATH_TO_SNN_WEIGHTS)
        #print(weights.shape)

        #print(self.net_info.ctrl_inputs)
        #print(self.net_info.state_inputs)
        #print(self.net_info.inputs)

        #self.net = snn.Predictor(weights=weights, seed=seed, n_neurons=n_neurons, c_init=np.zeros((len(self.net_info.inputs))))

        self.net = snn.Predictor_LMU2(action_init=np.zeros((len(self.net_info.ctrl_inputs))),
                    state_init=np.zeros((len(self.net_info.state_inputs))),
                    scales=self.scales,
                    weights=weights,
                    seed=seed,
                    n=neurons_per_dim,
                    samp_freq=samp_freq,
                    t_delay=t_delay,
                    lmu_theta=lmu_theta,
                    lmu_q=lmu_q,
                    dt=dt,
                )
        self.ens = self.net.ens

        self.normalization_info = get_norm_info_for_net(self.net_info)

        # Make a prediction

        #self.rnn_internal_states = get_internal_states(self.net)

        self.net_initial_input_without_Q = np.zeros([len(self.net_info.inputs) - len(CONTROL_INPUTS)], dtype=np.float32)

        self.prediction_denorm = None # Set to True or False in setup, determines if output should be denormalized

        self.output_array = np.zeros([self.batch_size, self.horizon+1, len(STATE_VARIABLES)+len(CONTROL_INPUTS)], dtype=np.float32)
        Q_type = tf.TensorSpec((self.horizon,), tf.float32)

        initial_input_type = tf.TensorSpec((len(self.net_info.inputs)-1,), tf.float32)

        net_input_type = tf.TensorSpec((self.batch_size, 1, len(self.net_info.inputs)), tf.float32)

        self.output_array_single_step = np.zeros([self.batch_size, 2, len(STATE_VARIABLES)+1], dtype=np.float32)


        # Retracing tensorflow functions
        try:
            self.evaluate_net = self.evaluate_net_f.get_concrete_function(net_input=net_input_type)
        except:
            self.evaluate_net = self.evaluate_net_f

        try:
            self.iterate_net = self.iterate_net_f.get_concrete_function(Q=Q_type,
                                                                        initial_input=initial_input_type)
            print(self.iterate_net)
        except:
            self.iterate_net = self.iterate_net_f

        print('Init done')

        self.snn_internal_states_voltage = self.net.return_internal_states('voltage')
        self.snn_internal_states_refractory = self.net.return_internal_states('refractory_time')


    def setup(self, initial_state: np.array, prediction_denorm=True):
        self.batch_size = initial_state.shape[0]

        #print(initial_state.shape)
        #print('That was init_state')
        #print(self.output_array.shape)
        #print(initial_state.shape)

        #self.output_array[..., 0, :-1] = initial_state
        self.output_array[:self.batch_size, 0, :-1] = initial_state
        #print(self.output_array)

        initial_input_net_without_Q = initial_state[..., [STATE_INDICES.get(key) for key in self.net_info.inputs[1:]]]
        self.net_initial_input_without_Q = normalize_numpy_array(initial_input_net_without_Q, self.net_info.inputs[1:], self.normalization_info)
        #print(self.net_initial_input_without_Q.shape)

        # [1:] excludes Q which is not included in initial_state_normed
        # As the only feature written with big Q it should be first on each list.
        self.net_initial_input_without_Q_TF = tf.convert_to_tensor(self.net_initial_input_without_Q, tf.float32)
        self.net_initial_input_without_Q_TF = tf.reshape(self.net_initial_input_without_Q_TF, [-1, len(self.net_info.inputs[1:])])
        #print(self.net_initial_input_without_Q_TF.shape)
        if prediction_denorm:
            self.prediction_denorm = True
        else:
            self.prediction_denorm = False

        #print('Setup done')

    def predict(self, Q, single_step=False) -> np.array:
        #print(Q.shape)  #(batch,horizon)
        if single_step:
            output_array = self.output_array_single_step
        else:
            output_array = self.output_array

        #print(output_array.shape)

        #output_array[..., :-1, -1] = Q
        output_array[:self.batch_size, :-1, -1] = Q
        #print('out_array got Q')

        # load internal RNN state if applies
        #load_internal_states(self.net, self.rnn_internal_states)
        self.net.set_internal_states(self.snn_internal_states_voltage, 'voltage')
        self.net.set_internal_states(self.snn_internal_states_refractory, 'refractory_time')
        #print('internal states')

        net_outputs = self.iterate_net(Q, single_step=single_step)

        # Denormalize
        #output_array[..., 1:, [STATE_INDICES.get(key) for key in self.net_info.outputs]] = \
            #denormalize_numpy_array(net_outputs.numpy(), self.net_info.outputs, self.normalization_info)
        output_array[..., 1:, [STATE_INDICES.get(key) for key in self.net_info.outputs]] = \
            denormalize_numpy_array(net_outputs, self.net_info.outputs, self.normalization_info)

        #output_array[..., 1:, [STATE_INDICES.get(key) for key in self.net_info.outputs]] = net_outputs.numpy()

        #print(output_array.shape)
        #print(output_array)

        # Augment
        augment_predictor_output(output_array, self.net_info)

        return output_array


    # @tf.function
    def update_internal_state(self, Q0):
        #self.batch_size = Q0.shape[0]
        # load internal RNN state
        #load_internal_states(self.net, self.rnn_internal_states)

        # Run current input through network
        Q0 = tf.squeeze(tf.convert_to_tensor(Q0, dtype=tf.float32))
        Q0 = tf.reshape(Q0, [-1, 1])
        if self.net_info.net_type == 'Dense':
            net_input = tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1).numpy()
        else:
            net_input = (tf.reshape(tf.concat([Q0, self.net_initial_input_without_Q_TF], axis=1),
                                    [-1, 1, len(self.net_info.inputs)])).numpy()

        #print(net_input.shape)     # (batch,1,in_size)

        # self.evaluate_net(self.net_current_input) # Using tf.function to compile net
        #print('Evaluating net')
        #self.net(net_input)    # Using net directly
        #self.net.step(c=net_input)     # Using net directly
        for ii in range(self.batch_size):
            #self.net.step(a=net_input[ii,:,0],s=net_input[ii,:,1:])   # Using net directly
            self.evaluate_net(net_input[ii, :, :])
        #print('Net done')

        self.snn_internal_states_voltage = self.net.return_internal_states('voltage')
        self.snn_internal_states_refractory = self.net.return_internal_states('refractory_time')
        #print('Got internal states')
        #self.rnn_internal_states = get_internal_states(self.net)

    # @tf.function
    def iterate_net_f(self, Q, single_step=False):
        #print(Q.shape)
        #self.horizon = Q.shape[0]

        if single_step:
            horizon = 1
        else:
            horizon = self.horizon

        #net_outputs = tf.TensorArray(tf.float32, size=horizon)
        net_outputs = np.zeros(shape=(self.batch_size,horizon,len(self.net_info.outputs)))
        net_output = np.zeros(shape=(self.batch_size,len(self.net_info.outputs)))

        #print(net_output.shape)

        for i in tf.range(0, horizon):
            Q_current = Q[..., i]
            Q_current = (tf.reshape(Q_current, [-1, 1]))
            if i == 0:
                    net_input = (tf.reshape(tf.concat([Q_current, self.net_initial_input_without_Q_TF], axis=1), [-1, 1, len(self.net_info.inputs)])).numpy()
            else:
                    net_input = tf.reshape(tf.concat([Q_current, tf.convert_to_tensor(net_output, np.float32)], axis=1), [-1, 1, len(self.net_info.inputs)]).numpy()

            #print(net_input.shape)  # For GUI is (2000,1,6)

            #net_output = self.net(net_input)

            for batch in range(self.batch_size):
                net_output[batch,:] = self.evaluate_net(net_input[batch,:,:])

            #print(net_output.shape)

            #net_output = tf.convert_to_tensor(net_output, np.float32)
            #tf.print(net_output)

            #net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])
            net_output = np.reshape(net_output, (-1, len(self.net_info.outputs)))
            #tf.print(net_output)
            #print(net_output.shape)

            #net_outputs = net_outputs.write(i, net_output)
            #net_outputs[batch,i,:] = net_output.numpy()
            net_outputs[:, i, :] = net_output
            #tf.print(net_outputs)
            # tf.print(net_inout.read(i+1))

        # print(net_inout)
        #net_outputs = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])
        #print(net_outputs.shape)

        return net_outputs

    #@tf.function
    def evaluate_net_f(self, net_input):
        # print('retracing evaluate_net_f')
        #net_output = self.net(net_input)
        #c = net_input

        #net_output = self.net.step(c=c)
        #self.net.reset()
        #a = net_input[:,:,0]
        #s = net_input[:,:,1:]
        a = net_input[:, 0]
        s = net_input[:, 1:]
        net_output = self.net.step(a=a,s=s)

        #net_out_aux = net_output
        #net_output[2] = net_out_aux[1]
        #net_output[1] = net_out_aux[2]
        return net_output

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        if self._horizon is None:
            # print('I used initialization setter!')
            self._horizon = value
        else:
            # print('I used normal setter!')
            self._horizon = value
            # self.output_array = np.zeros([self.horizon + 1, self.batch_size, len(STATE_VARIABLES) + 1],
            #                              dtype=np.float32)
            #
            # Q_type = tf.TensorSpec((self.horizon,), tf.float32)
            #
            # initial_input_type = tf.TensorSpec((len(self.net_info.inputs) - 1,), tf.float32)
            #
            # net_input_type = tf.TensorSpec((self.batch_size, len(self.net_info.inputs)), tf.float32)
            #
            # # Retracing tensorflow functions
            # try:
            #     self.evaluate_net = self.evaluate_net_f.get_concrete_function(net_input=net_input_type)
            # except:
            #     self.evaluate_net = self.evaluate_net_f
            #
            # try:
            #     self.iterate_net = self.iterate_net_f.get_concrete_function(Q=Q_type,
            #                                                                 initial_input=initial_input_type)
            #     print(self.iterate_net)
            # except:
            #     self.iterate_net = self.iterate_net_f