import numpy as np
import nengo
import yaml
import os

#-----------------------------------------------------------------------------------------------------------------------

class NetInfo():
    def __init__(self):
        config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'),
                       Loader=yaml.FullLoader)

        self.inputs = config['training_default']['control_inputs']
        self.inputs.extend(config['training_default']['state_inputs'])

        self.outputs = config['training_default']['outputs']

        self.net_type = 'SNN'

        # This part is forced to read from previous non-SNN network (should be changed when integrated properly to SI_Toolkit)
        self.path_to_normalization_info = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0/NI_2021-06-29_12-02-03.csv'
        #self.parent_net_name = 'Network trained from scratch'
        self.parent_net_name = 'GRU-6IN-32H1-32H2-5OUT-0'
        #self.path_to_net = None
        self.path_to_net = './SI_Toolkit_ApplicationSpecificFiles/Experiments/Pretrained-RNN-1/Models/GRU-6IN-32H1-32H2-5OUT-0'

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

#-----------------------------------------------------------------------------------------------------------------------

#freq = 5
#learning_rate = 5e-5
#t_delay = 0.05

class Predictor(object):
    def __init__(self, weights, seed, n_neurons, c_init):
        self.model = nengo.Network()
        self.c = np.array(c_init)
        self.z_pred = np.zeros(weights.shape[0])
        with self.model:
            c = nengo.Node(lambda t: self.c)

            def set_z_pred(t, x):
                self.z_pred[:] = x

            z_pred = nengo.Node(set_z_pred, size_in=weights.shape[0])

            ens = nengo.Ensemble(n_neurons=n_neurons, dimensions=len(self.c),
                                 neuron_type=nengo.LIFRate(), seed=seed)
            nengo.Connection(c, ens, synapse=None)

            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights, synapse=None)


        self.sim = nengo.Simulator(self.model, dt=0.001, progress_bar=False)
        self.ens = ens

    def step(self, c):
        self.c[:] = c
        self.sim.run(0.001)
        return self.z_pred

    def reset(self):
        self.sim.reset()

#-----------------------------------------------------------------------------------------------------------------------