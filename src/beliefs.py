import numpy as np
import sensor_models

def kullback_leibler_divergence(P, Q):
    return (P * np.log(P / Q)).sum()

class StatelessBinarySensor(sensor_models.BinarySensor):
    def full_likelihood(self, state):
        return self.pzgx

    def obs_likelihood(self, obs):
        return self.pzgx[obs]

class DiscreteBinarySensor(sensor_models.BinaryLogisticObs):
    def build_full_likelihood(self, observation_states, true_states):
        # This should build a dictionary {state: [n_true_states x n_observations]}
        self.full_state_likelihood = {}
        for x_o in observation_states:
            s_likelihood = np.zeros((self.get_n_returns(), len(true_states)))
            for z in range(self.get_n_returns()):
                s_likelihood[z] = [self.likelihood(x_o, z, x_t) for x_t in true_states]
            self.full_state_likelihood[x_o] = s_likelihood

    def full_likelihood(self, obs_state):
        return self.full_state_likelihood[obs_state]

    def obs_likelihood(self, obs):
        return self.full_state_likelihood[obs[0]][obs[1]]


class DiscreteBelief(object):
    def __init__(self, sensor_model, state_list, prior=None):
        # A sensor model must have:
        #   - get_n_returns() function that tells you how many returns the sensor can output
        #   - full_likelihood(true_state) the probability of all observations from a given state
        #   - obs_likelihood(obs) where obs is a tuple with (obs_state, z)
        self.observations = []
        self.sensor = sensor_model
        self.reset_state_list(state_list, prior)
        self.shared_pX = self.pX.copy()
        self.last_shared = 0

    def reset(self, state_list=None, prior=None):
        self.reset_observations()
        if state_list is not None:
            self.reset_state_list(state_list, prior)
        self.shared_pX = self.pX.copy()
        self.last_shared = 0

    def reset_observations(self):
        self.observations = []
        self.pX = self.prior.copy()

    def reset_state_list(self, state_list, prior=None):
        temp_obs = self.observations
        self.states = state_list
        self._n_states = len(self.states)
        if prior is None:
            prior = np.ones(self.get_n_states())*1.0/self.get_n_states()
        assert len(prior) == self.get_n_states()
        self.prior = prior
        self.pX = self.prior.copy()
        self.add_observations(temp_obs)

    def get_n_states(self):
        return self._n_states

    def add_observation(self, obs):
        # An observation consists of the state the observation was taken in and the sensor return (integer)
        self.observations.append(obs)
        pzgX, pzandX, pz = self.observation_likelihood(obs)
        self.pX = pzandX/pz

    def add_observations(self, obs_list):
        self.observations.extend(obs_list)
        pZgX, pZandX, pZ = self.joint_observation_likelihood(obs_list)
        self.pX = pZandX / pZ

    def observation_likelihood(self, obs):
        pzgX = self.sensor.obs_likelihood(obs)
        pzandX = pzgX * self.pX
        pz = pzandX.sum()
        return pzgX, pzandX, pz

    def joint_observation_likelihood(self, obs_list):
        pZgX = 1.0
        for obs in obs_list:
            pZgX *= self.sensor.obs_likelihood(obs)
        pZandX = pZgX*self.pX
        pZ = pZandX.sum()
        return pZgX, pZandX, pZ

    def get_observations(self):
        return self.observations

    def E_Dkl(self, trajectory, depth, current_depth = 0, pJgX = 1.0):
        if current_depth >= depth:
            pJandX = pJgX*self.pX
            return (pJandX*(np.log(pJgX) - np.log(pJandX.sum()))).sum()

        E_d = 0.0
        for z in range(self.sensor.get_n_returns()):
            n_pJgX = pJgX*self.sensor.obs_likelihood((trajectory[depth], z))
            E_d += self.E_Dkl(depth, current_depth+1, n_pJgX)
        return E_d

    def set_last_shared(self):
        self.shared_pX = self.pX.copy()
        self.last_shared = len(self.observations)

    def get_unshared_observations(self):
        return self.observations[self.last_shared:]


class Explorer(object):
    # This class defines an exploring vehicle that makes observations and maintains a belief over a target location

    def __init__(self, motion_model, sensor_model, state_list, prior=None, start_state=0, vehicle_color='gold', *args, **kwargs):
        self.motion_model = motion_model
        self.sensor = sensor_model
        self.start_state = start_state
        self.set_current_state(start_state)
        self.full_path = np.array([self.get_current_pose()])
        self.vehicle_color = vehicle_color
        self.belief = DiscreteBelief(sensor_model, state_list, prior)
        self._plots = False
        self._init_extras(*args, **kwargs)

    def _init_extras(self, *args, **kwargs):
        pass

    def reset(self, new_start_state=None, state_list=None, prior=None, *args, **kwargs):
        if new_start_state is not None:
            self.start_state = new_start_state
        self.set_current_state(self.start_state)
        self.full_path = np.array([self.get_current_pose()])
        self._reset_extras(*args, **kwargs)
        self.reset_observations()

    def _reset_extras(self, *args, **kwargs):
        pass

    def set_start_state(self, state):
        self.start_state = state

    def get_start_state(self):
        return self.start_state

    def set_current_state(self, state):
        self.state = state

    def get_current_state(self):
        return self.state

    def get_current_pose(self):
        return self.get_pose(self.get_current_state())

    def get_pose(self, state):
        return state

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def generate_observations(self, x, c=None, set_obs=False):
        # Generate observations at an array of locations
        if c is None:
            c = self.world.get_target_location()
        obs = self.sensor.generate_observations(x, c)
        if set_obs:
            self.add_observations(obs)
        return obs

    def kld_select_obs(self, depth):
        self.leaf_states = self.motion_model.get_leaf_states(self.get_current_state(), depth)
        self.next_states = self.motion_model.get_leaf_states(self.get_current_state(), 1)

        self.leaf_values = np.array(self.kld_tree(depth))
        path_max = np.unravel_index(np.argmax(self.leaf_values),
                                    self.motion_model.get_paths_number() * np.ones(depth, dtype='int'))
        amax = path_max[0]

        self.prune_likelihood_tree(amax, depth)
        self.full_path = np.append(self.full_path, self.motion_model.get_trajectory(self.get_current_state(), amax),
                                   axis=0)

        self.set_current_state(self.next_states[amax])

        cobs = self.generate_observations([self.get_current_pose()])
        self.add_observations(cobs)

        return amax

    def prune_likelihood_tree(self, selected_option, depth):
        self.likelihood_tree.children[selected_option].add_children(depth)
        self.likelihood_tree = self.likelihood_tree.children[selected_option]

    def setup_plot(self, h_ax, tree_depth=None, obs_symbols=['r^', 'go'], ms_start=8, ms_target=10, ms_scatter=20,
                   ms_obs=6.5):
        self._plots = True
        im_extent = [0.0, self.world.get_size()[0], 0.0, self.world.get_size()[1]]
        h_ax.clear()
        self.h_ax = h_ax
        self.h_artists = {}
        self.h_artists['pc'] = self.h_ax.imshow(np.zeros(self.world.get_size()), extent=im_extent, origin='lower',
                                                vmin=0, animated=True)
        self.h_artists['cpos'], = self.h_ax.plot([], [], 'o', color=self.vehicle_color, fillstyle='full', ms=ms_start,
                                                 mew=0)
        target_pos = self.world.get_target_location()
        self.h_artists['target'], = self.h_ax.plot(target_pos[0], target_pos[1], 'wx', mew=2, ms=ms_target)
        start_pose = self.get_pose(self.get_start_state())
        self.h_artists['start'], = self.h_ax.plot(start_pose[0], start_pose[1], '^', color='orange', ms=ms_start,
                                                  fillstyle='full')
        self.h_artists['obsF'], = self.h_ax.plot([], [], obs_symbols[0], mew=0.5, mec='w', ms=ms_obs)
        self.h_artists['obsT'], = self.h_ax.plot([], [], obs_symbols[1], mew=0.5, mec='w', ms=ms_obs)
        self.h_artists['path'], = self.h_ax.plot([], [], 'w-', lw=2)
        if tree_depth is not None:
            self.h_artists['tree'] = self.setup_tree_plot(tree_depth, ms_scatter)
        else:
            self.h_artists['tree'], = self.h_ax.plot([], [])

        if self.unshared:
            self.h_artists['shared_obsF'], = self.h_ax.plot([], [], '^', color='darksalmon', mec='w', mew=0,
                                                            ms=ms_obs - 1.5)
            self.h_artists['shared_obsT'], = self.h_ax.plot([], [], 'o', color='darkseagreen', mec='w', mew=0,
                                                            ms=ms_obs - 1.5)
        self.h_ax.set_xlim(-.5, self.world.get_size()[0] - 0.5)
        self.h_ax.set_ylim(-.5, self.world.get_size()[1] - 0.5)

    def update_plot(self):
        cpos = self.get_current_pose()
        self.h_artists['cpos'].set_data(cpos[0], cpos[1])

        if self.belief.update_pc_map:
            pc = self.belief.persistent_centre_probability_map()
            pc = pc / pc.sum()
            self.h_artists['pc'].set_data(pc.transpose())
            self.h_artists['pc'].set_clim([0, pc.max()])

        obsT = [xx for xx, zz in self.belief.get_observations() if zz == 1]
        obsF = [xx for xx, zz in self.belief.get_observations() if zz == 0]
        self.update_obs(self.h_artists['obsT'], obsT)
        self.update_obs(self.h_artists['obsF'], obsF)

        self.h_artists['path'].set_data(self.full_path[:, 0], self.full_path[:, 1])

        self.update_tree_plot()
        # return self.h_artists.values()

    def setup_tree_plot(self, tree_depth, ms_scatter):
        leaf_states = self.motion_model.get_leaf_states(self.get_start_state(), depth=tree_depth)
        return self.h_ax.scatter(leaf_states[:, 0], leaf_states[:, 1], ms_scatter)

    def update_tree_plot(self):
        try:
            self.h_artists['tree'].set_offsets(self.leaf_states[:, 0:2])
            self.h_artists['tree'].set_array(self.leaf_values - self.leaf_values.min())
        except (KeyError, AttributeError):
            pass

    def get_artists(self):
        # This is because stupid animate doesn't respect plot order, so I can't just return h_artsists.values()
        if self.unshared:
            return (self.h_artists['pc'], self.h_artists['cpos'], self.h_artists['target'],
                    self.h_artists['start'], self.h_artists['obsT'], self.h_artists['obsF'],
                    self.h_artists['path'], self.h_artists['tree'],
                    self.h_artists['shared_obsT'], self.h_artists['shared_obsF'])
        else:
            return (self.h_artists['pc'], self.h_artists['cpos'], self.h_artists['target'],
                    self.h_artists['start'], self.h_artists['obsT'], self.h_artists['obsF'],
                    self.h_artists['path'], self.h_artists['tree'])

    def update_obs(self, h, obs):
        if obs != []:
            h.set_data([o[0] for o in obs], [o[1] for o in obs])
            # h.set_data(*zip(*obs))

    def add_observations(self, obs, *args, **kwargs):
        self.belief.add_observations(obs, *args, **kwargs)

    def get_observations(self):
        return self.belief.get_observations()

    def reset_observations(self):
        self.belief.reset_observations()
