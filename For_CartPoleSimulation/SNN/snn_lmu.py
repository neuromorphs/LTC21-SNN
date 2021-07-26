import nengo
import numpy as np
import scipy.linalg
from scipy.special import legendre
import yaml
import os

#-----------------------------------------------------------------------------------------------------------------------

class NetInfo():
    def __init__(self):
        config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'),
                       Loader=yaml.FullLoader)

        #self.ctrl_inputs = config['training_default']['control_inputs']         # For any reason, this is reading the full vector of [ctrl_in,state_in]
        self.ctrl_inputs = ['Q']                                                 # I force it, could not find a way to only read 'Q'
        self.state_inputs = config['training_default']['state_inputs']

        self.inputs = config['training_default']['control_inputs']
        self.inputs.extend(self.state_inputs)

        self.outputs = config['training_default']['outputs']

        self.net_type = 'SNN'

        # This part is forced to read from previous non-SNN network (should be changed when integrated properly to SI_Toolkit)
        self.path_to_normalization_info = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0/NI_2021-06-29_12-02-03.csv'
        #self.parent_net_name = 'Network trained from scratch'
        self.parent_net_name = 'GRU-6IN-32H1-32H2-5OUT-0'
        #self.path_to_net = None
        self.path_to_net = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0'

#-----------------------------------------------------------------------------------------------------------------------

def minmax(invalues, bound):
    '''Scale the input to [-1, 1]'''
    out =  2 * (invalues + bound) / (2*bound) - 1
    return out

#-----------------------------------------------------------------------------------------------------------------------

def scale_datasets(data, scales):
    '''Scale inputs in a list of datasets to -1, 1'''

    # Scale all datasets to [-1, 1] based on the maximum value found above
    bounds = []
    for var, bound in scales.items():
        bounds.append(bound)
    for ii in range(len(bounds)):
        data[:,ii] = minmax(data[:,ii], bounds[ii])

    return data

#-----------------------------------------------------------------------------------------------------------------------

class DiscreteDelay(nengo.synapses.Synapse):
    def __init__(self, delay, size_in=1):
        self.delay = delay
        super().__init__(default_size_in=size_in, default_size_out=size_in)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        steps = int(self.delay/dt)
        if steps == 0:
            def step_delay(t, x):
                return x
            return step_delay
        assert steps > 0

        state = np.zeros((steps, shape_in[0]))
        state_index = np.array([0])

        def step_delay(t, x, state=state, state_index=state_index):
            result = state[state_index]
            state[state_index] = x
            state_index[:] = (state_index + 1) % state.shape[0]
            return result

        return step_delay

# -----------------------------------------------------------------------------------------------------------------------

class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q  # number of internal state dimensions per input
        self.theta = theta  # size of time window (in seconds)
        self.size_in = size_in  # number of inputs

        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.) ** i * (2 * i + 1)
            for j in range(q):
                A[i, j] = (2 * i + 1) * (-1 if i < j else (-1.) ** (i - j + 1))
        self.A = A / theta
        self.B = B / theta

        super().__init__(default_size_in=size_in, default_size_out=q * size_in)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A * dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad - np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()

        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2 * r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T

# -----------------------------------------------------------------------------------------------------------------------

class Predictor(object):
    def __init__(self, action_init, state_init, weights=None, seed=42, n=100, samp_freq=50,
               t_delay=0.02, learning_rate=5e-5, dt=0.001):
        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.in_sz = (len(self.s) + len(self.a))

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz))

        self.z_pred = np.zeros(weights.shape[0])

        with self.model:
            a = nengo.Node(lambda t: self.a)
            s = nengo.Node(lambda t: self.s)

            #def set_z_pred(t, x):
                #self.z_pred[:] = x

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=4)
            nengo.Connection(s, z)

            z_pred = nengo.Node(None, size_in=4)

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * 5, dimensions=5,
                                     neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(a, ens[0])
            nengo.Connection(s, ens[1:])

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                pre_synapse=DiscreteDelay(t_delay)
                                                                # delay the activity value when updating weights
                                                                ))

            self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
            self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        self.s[:] = s
        self.sim.run(self.dt)
        return self.z_pred

    def return_internal_states(self, key):
        return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])

    def set_internal_states(self, internal_states, key):
        self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states

    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------

class Predictor_LMU(object):
    def __init__(self, action_init, state_init, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
    t_delay=0.02, learning_rate=0, radius=1.5, dt = 0.001):

        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.in_sz = (len(self.s) + len(self.a))

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz))

        self.z_pred = np.zeros(weights.shape[0])

        with self.model:
            a = nengo.Node(lambda t: self.a)
            s = nengo.Node(lambda t: self.s)

            def set_z_pred(t, x):
                self.z_pred[:] = x

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=len(self.s))
            nengo.Connection(s, z)

            z_pred = nengo.Node(set_z_pred, size_in=len(self.s))

            ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=self.in_sz))

            nengo.Connection(a, ldn[0])
            nengo.Connection(s, ldn[1:])

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * self.in_sz, dimensions=self.in_sz * lmu_q,
                                     neuron_type=nengo.LIF(), seed=seed)

            # How do I connect each lmu to one dimension of ens?
            nengo.Connection(ldn, ens)

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                 pre_synapse=DiscreteDelay(t_delay)
                                                                 # delay the activity value when updating weights
                                                                 ))

            self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
            self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        self.s[:] = s
        self.sim.run(self.dt)
        return self.z_pred

    def return_internal_states(self, key):
        return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])

    def set_internal_states(self, internal_states, key):
        self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states


    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------

class Predictor_LMU2(object):
    def __init__(self, action_init, state_init, scales, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
    t_delay=0.02, learning_rate=0, radius=1.5, dt = 0.001):

        self.model = nengo.Network()
        self.model.config[nengo.Connection].synapse = None

        self.a = np.array(action_init)
        self.s = np.array(state_init)

        self.dt = dt
        self.in_sz = (len(self.s)+len(self.a))

        self.weights = weights
        if self.weights is None:
            self.weights = np.zeros((len(self.s), n * self.in_sz * (1 + lmu_q)))

        self.z_pred = np.zeros(weights.shape[0])

        self.scales = scales

        with self.model:
            a = nengo.Node(lambda t: self.a)
            s = nengo.Node(lambda t: self.s)

            def set_z_pred(t, x):
                self.z_pred[:] = x

            z = nengo.Node(None, size_in=len(s))
            nengo.Connection(s, z)

            z_pred = nengo.Node(set_z_pred, size_in=weights.shape[0])

            ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=self.in_sz))

            nengo.Connection(a, ldn[0])
            nengo.Connection(s, ldn[1:])

            # make the hidden layer
            ens = nengo.Ensemble(n_neurons=n * self.in_sz * (1 + lmu_q), dimensions=self.in_sz * (1 + lmu_q),
                                 neuron_type=nengo.LIF(), seed=seed, radius=radius)

            # How do I connect each lmu to one dimension of ens?
            nengo.Connection(a, ens[:1])
            nengo.Connection(s, ens[1:self.in_sz])
            nengo.Connection(ldn, ens[self.in_sz:])

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=self.weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                                 pre_synapse=DiscreteDelay(t_delay)
                                                                 # delay the activity value when updating weights
                                                                 ))

        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)
        self.ens = ens

    def step(self, a, s):
        self.a[:] = a
        #self.s[:] = scale_datasets(s,self.scales)
        self.s[:] = s
        self.sim.run(self.dt)
        return self.z_pred

    def return_internal_states(self, key):
        return np.asarray(self.sim.signals[self.sim.model.sig[self.ens.neurons][key]])

    def set_internal_states(self, internal_states, key):
        self.sim.signals[self.sim.model.sig[self.ens.neurons][key]] = internal_states


    def reset(self):
        self.sim.reset()

# -----------------------------------------------------------------------------------------------------------------------