import nengo
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import legendre
import pickle


# This code is completely taken from Terry Steward:
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

# This code is completely taken from Terry Steward:
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


class DataParser:
    """This node will parse data from a pandas dataframe object as input.
    """
    def __init__(self, data_df, sample_freq, vars=[]):
        self.data_df = data_df
        self.sample_freq = sample_freq
        self.vars = vars

    def parse_data(self, t):
        r = [self.data_df[x].iloc[int(t * self.sample_freq)] for x in self.vars]
        return r

    def update_data(self, data_df):
        self.data_df = data_df


class SwitchNode:
    """This node will return the first half of its input for t <= t_switch and
    the second half of its input for t > t_switch.
    By default, the output will be clipped between -1 and 1.
    """
    def __init__(self, t_switch=4.0, stim_size=4, clip=True):
        self.t_switch = t_switch
        self.stim_size = stim_size
        self.clip=clip

    def step(self, t, x):
        if t <= self.t_switch:
            state = x[:self.stim_size]
        else:
            state = x[self.stim_size:]

        if self.clip:
            state = np.clip(state, -1, 1)

        return state


class ErrorSwitchNode:
    """This node will compute the error between a prediction and ground truth.
    The output of the node will be 0 for t < t_init and t > t_switch.
    """
    def __init__(self, t_init=0.1, t_switch=4.0, stim_size=4, error="normal"):
        self.t_init = t_init
        self.t_switch = t_switch
        self.stim_size = stim_size
        self.error = error
        if error.lower() == "normal":
            self.error_func = self.normal
        elif error.lower() == "mse":
            self.error_func = self.mse
        else:
            raise ValueError('This error type is not supported. Please use "mse" or "normal".')

    def mse(self, a, b):
        return np.sign(a - b) * (a - b) ** 2

    def normal(self, a, b):
        return a - b

    def step(self, t, x):
        if self.t_init <= t <= self.t_switch:
            return self.error_func(x[:self.stim_size], x[self.stim_size:])
        return np.zeros(self.stim_size)


class PredictiveModelAutoregressiveLMU:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 lmu_theta=0.1, lmu_q=1, radius=1.5, dt=0.001,
                 t_delay=0.02, learning_rate=5e-5, action_vars=["Q"],
                 state_vars=["angle_sin", "angle_cos", "angleD", "position", "positionD"],
                 action_df=None, state_df=None, weights=None, scales={},
                 predict_delta=True, error="normal", t_switch=4.0, t_init=0.1,
                 model_name="AutoregressiveLMU", *args, **kwargs):

        self.seed = seed
        self.neurons_per_dim = neurons_per_dim
        self.sample_freq = sample_freq
        self.t_delay = t_delay
        self.learning_rate = learning_rate
        self.action_vars = action_vars
        self.action_dim = len(action_vars)
        self.state_vars = state_vars
        self.state_dim = len(state_vars)
        self.radius = radius
        self.dt = dt
        self.lmu_q = lmu_q
        self.lmu_theta = lmu_theta
        self.weights = weights
        self.scales = scales
        self.predict_delta = predict_delta
        self.t_switch = t_switch
        self.t_init = t_init
        self.error = error
        self.model_name = model_name


        if action_df is None:
            self.action_df = pd.DataFrame(
                np.zeros((1, len(action_vars) + 1)),
                columns=["time"] + action_vars,
            )

        if state_df is None:
            self.state_df = pd.DataFrame(
                np.zeros((1, len(state_vars) + 1)),
                columns=["time"] + state_vars,
            )

        self.action_parser = DataParser(
            data_df=self.action_df,
            sample_freq=self.sample_freq,
            vars=self.action_vars
        )

        self.state_parser = DataParser(
            data_df=self.state_df,
            sample_freq=self.sample_freq,
            vars=self.state_vars
        )

        # this function streams the state signal from file to node
        def action_stim_func(t):
            return self.action_parser.parse_data(t)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return self.state_parser.parse_data(t)

        self.action_stim_func = action_stim_func
        self.state_stim_func = state_stim_func

        self.model, self.recordings = self.make_model()
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def set_inputs(self, action_df, state_df):

        for v in self.action_vars + ["time"]:
            assert v in action_df.columns
        for v in self.state_vars:
            assert v in state_df.columns

        self.action_df = action_df
        self.state_df = state_df

        self.action_parser.update_data(self.action_df)
        self.state_parser.update_data(self.state_df)

    def reset_sim(self):

        self.sim.reset(seed=self.seed)

    def set_weights(self, weights):

        weights = np.array(weights)
        assert weights.shape == (
            self.state_dim,
            self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q)
        )

        self.weights = weights
        self.sim.signals[self.sim.model.sig[self.learned_connection]["weights"]] = self.weights

    def get_weights(self):

        weights = self.sim.signals[self.sim.model.sig[self.learned_connection]["weights"]]
        return np.array(weights)

    def process_files(self):

        t_max = self.action_df["time"].max()  # number of seconds to run
        self.sim.run(t_max)

        return self.recordings

    def get_state_dict(self):

        state_dict = {
            "seed": self.seed,
            "neurons_per_dim": self.neurons_per_dim,
            "sample_freq": self.sample_freq,
            "t_delay": self.t_delay,
            "learning_rate": self.learning_rate,
            "action_vars": self.action_vars,
            "state_vars": self.state_vars,
            "radius": self.radius,
            "dt": self.dt,
            "lmu_q": self.lmu_q,
            "lmu_theta": self.lmu_theta,
            "weights": self.get_weights(),
            "scales": self.scales,
            "predict_delta": self.predict_delta,
            "error": self.error,
            "t_switch": self.t_switch,
            "t_init": self.t_init,
            "model_name": self.model_name
        }

        return state_dict

    def save_state_dict(self, path="model_state.pkl"):

        state_dict = self.get_state_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def make_model(self):

        if self.weights is None:
            self.weights = np.zeros((
                self.state_dim,
                self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))
            )

        model = nengo.Network()
        with model:
            # set the default synapse to None (normal default is 0.005s)
            model.config[nengo.Connection].synapse = None

            # initialize input nodes
            true_action = nengo.Node(self.action_stim_func)
            true_state = nengo.Node(self.state_stim_func)

            # create a node that first outputs the true state and then switches to predicted state
            switch = SwitchNode(t_switch=self.t_switch, stim_size=self.state_dim)
            believed_state = nengo.Node(switch.step, size_in=self.state_dim*2, size_out=self.state_dim)
            nengo.Connection(true_state, believed_state[:self.state_dim])

            # make a node for the predicted future state
            predicted_future_state = nengo.Node(None, size_in=self.state_dim)

            # make LMU unit
            ldn = nengo.Node(LDN(theta=self.lmu_theta, q=self.lmu_q, size_in=self.state_dim + self.action_dim))
            nengo.Connection(true_action, ldn[:self.action_dim])
            nengo.Connection(believed_state, ldn[self.action_dim:])

            # make the hidden layer
            ens = nengo.Ensemble(
                n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim)*(1+self.lmu_q),
                dimensions=(self.state_dim + self.action_dim)*(1+self.lmu_q),
                neuron_type=nengo.LIFRate(),
                seed=self.seed
            )
            nengo.Connection(true_action, ens[:self.action_dim])
            nengo.Connection(believed_state, ens[self.action_dim:self.action_dim+self.state_dim])
            nengo.Connection(ldn, ens[self.action_dim+self.state_dim:])

            # if wanted, have the model predict the difference from the last state instead
            if self.predict_delta:
                print("predicting only the difference between the current and next state")
                nengo.Connection(believed_state, predicted_future_state)

            # make the output weights we can learn
            self.learned_connection = nengo.Connection(
                ens.neurons,
                predicted_future_state,
                transform=self.weights,  # change this if you have pre-recorded weights to use
                seed=self.seed,
                learning_rule_type=nengo.PES(
                    learning_rate=self.learning_rate,
                    pre_synapse=DiscreteDelay(self.t_delay)  # delay the activity value when updating weights
                )
            )

            # this is what the network predicted the current state to be in the past
            predicted_current_state = nengo.Node(None, size_in=self.state_dim)
            nengo.Connection(predicted_future_state, predicted_current_state, synapse=DiscreteDelay(self.t_delay))
            nengo.Connection(predicted_current_state, believed_state[self.state_dim:])

            # compute the error by subtracting the current measurement from a delayed version of the prediction
            error_node = ErrorSwitchNode(
                t_init=self.t_init,
                t_switch=self.t_switch,
                stim_size=self.state_dim,
                error=self.error
            )
            prediction_error = nengo.Node(error_node.step, size_in=self.state_dim*2, size_out=self.state_dim)
            nengo.Connection(predicted_current_state, prediction_error[:self.state_dim])
            nengo.Connection(true_state, prediction_error[self.state_dim:])

            # apply the error to the learning rule
            nengo.Connection(prediction_error, self.learned_connection.learning_rule)

            recordings = {
                "states" : nengo.Probe(true_state),
                "actions": nengo.Probe(true_action),
                "delay": self.t_delay,
                "predicted_current_states": nengo.Probe(predicted_current_state),
                "predicted_future_states": nengo.Probe(predicted_future_state),
                "prediction_errors": nengo.Probe(prediction_error),
            }

        return model, recordings