#!/usr/bin/python3

import ipdb
import time
import helpers
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from bcolors import bcolors
from selection import roullet_wheel as rw

#  np.full(shape, fill_value, dtype=None, order='C') > Return a new array of given shape and type, filled with fill_value.
#  np.random.rand(self.num_ships, self.dimensions)
#  np.random.uniform(low=0.0, high=1.0, size=(2,2))

class Pirates():
    def __init__(self, func, fmax=(), fmin=(), hr=0.2, ms=3, max_r=1, num_ships=5, dimensions=2, max_iter=10, max_wind=1, c={},
                 top_ships=10, dynamic_sails=False, iteration_plots=False, animate=False, log=False, quiet=False,
                 sailing_radius=0.3, plundering_radius=0.1):

        self.num_ships = num_ships              #  Number of particles
        self.num_top_ships = top_ships          #  Number of top ships
        self.quiet = quiet                      #  Should algorithm run quietly or not
        self.log_flag = log                     #  Enable logging
        self.log_num = 0                        #  Logging numbering
        self.animate = animate                  #  Enable animation
        self.max_iter = max_iter                #  Max iteration
        self.func_obj = func                    #  Handler to all function properties (name, fopt, xopt, bounds)
        # Check if func_obj is a function or an object with a func method
        if callable(self.func_obj) and not hasattr(self.func_obj, 'func'):
            self.cost_func = self.func_obj      # If it's a function, use it directly
        else:
            self.cost_func = self.func_obj.func  # If it's an object, use its func method
        self.fmin = fmin                        #  Problem min range (0, 0) for each dimension
        self.fmax = fmax                        #  Problem max range (10, 100) for each dimension
        self.dimensions = dimensions            #  Problem dimensions
        
        # Default values for constant weights if not provided
        default_c = {
            'leader': 0.5,
            'private_map': 0.5,
            'map': 0.5,
            'top_ships': 0.5
        }
        self.c = {**default_c, **c}            #  Constant weights with defaults
        
        self.iteration_plots = iteration_plots  #  Plot curves in specefic iterations

        # Sailing parameters
        self.sailing_radius = sailing_radius
        self.plundering_radius = plundering_radius
        self.initial_sailing_radius = sailing_radius
        self.initial_plundering_radius = plundering_radius
        self.sailing_angles = True  # Enable sailing angles

        self.knot = None
        self.max_wind = max_wind
        self.dynamic_sails = dynamic_sails

        self.leader_index = None
        self.hr = 1 - hr

        self.r = None
        self.max_r = max_r

        #  map size
        self.ms = ms
        self.map = None #  np.full((ms, dimensions), 0.0)

        self.problem = 'min' #  or max

        # Improve percision
        # BUG: on longdouble - ackely function returns nagetive value
        self.dtype = np.longdouble
        self.dtype = np.float64

        #  Diagrams
        self.bsf_position = None
        self.bsf_list = []
        self.avg = []
        self.wind = None
        self.trajectory = []
        self.v_history = []
        self.w_history = []
        self.f_history = []
        self.a_history = []

        self.random_init()

        self.iter = 0
        self.log('Initialized...')

        self.start()

    def log(self, *string, **kwargs):

        #  Color mode [warrning, okblue, error]
        mode = kwargs.get('c', 'ENDC')

        #  log_type i info, w warning, e error
        if self.log_flag and not self.quiet:
            print('[' + str(self.log_num) + ']', '(' + str(self.iter) + ')', getattr(bcolors, mode), *string, bcolors.ENDC)
            print()
            self.log_num += 1

    def random_init(self):

        #  Randomly initialize particles (Ships)

        #  Configure numpy printing percision
        np.set_printoptions(precision=20)

        self.ships = np.full((self.num_ships, self.dimensions), 0.0, dtype=self.dtype)

        self.private_map = self.ships.copy()
        self.private_map_costs = None

        self.ships = np.random.uniform(self.fmin, self.fmax, size=(self.num_ships, self.dimensions)).astype(self.dtype)

        self.v = np.random.rand(self.num_ships, self.dimensions)

        #  self.costs = np.full(self.num_ships, 0)
        self.costs = np.full(self.num_ships, 0.0, dtype=self.dtype)

        #  Randomly initialize ships sails states
        self.sails_stats = np.array([np.pi/2])
        self.sails = np.random.choice(self.sails_stats, self.num_ships, replace=True)

    def update_sails(self):
        """
        Update the sailing parameters based on the current iteration
        """
        # Update sailing radius gradually
        self.sailing_radius = self.initial_sailing_radius * (1 - (self.iter / self.max_iter))
        self.plundering_radius = self.initial_plundering_radius * (1 - (self.iter / self.max_iter))
        
        # Update sailing angles range
        if self.sailing_angles:
            half = (np.pi/4) - (self.iter / self.max_iter * (np.pi/4))
            self.sailing_angle_range = (-half, half)

    def cal_costs(self):
        """
        Calculate costs for all ships and return best cost and metrics
        
        Returns:
        --------
        tuple (float, dict) or None
            Best cost and its corresponding metrics, or None if cost function doesn't return metrics
        """
        best_error = float('inf')
        best_metrics = None
        
        for i in range(self.num_ships):
            result = self.cost_func(self.ships[i])
            
            if isinstance(result, tuple) and len(result) == 2:
                error, metrics = result
                self.costs[i] = error  # Only store the error value
                
                # Store best metrics
                if error < best_error:
                    best_error = error
                    best_metrics = metrics
            else:
                # Cost function only returned a single value
                self.costs[i] = result
        
        if best_metrics is not None:
            return best_error, best_metrics
        
        # For older versions that don't return metrics, return None
        return None

    def update_leader(self):
        leader_index = np.argmin(self.costs)

        if self.leader_index is None:
            self.leader_index = leader_index
            self.log('Leader initialized! cost:', self.costs[self.leader_index], self.leader_index, c='BLUE')
            return

        # Only update and log when leader is some new ship!
        if leader_index == self.leader_index:
            return

        # Skip updating leader when new leader's cost is same as old one
        if self.costs[leader_index] == self.costs[self.leader_index]:
            self.log('Skip updating leader', c='YELLOW2')
        else:
            old_leader_index = self.leader_index
            self.leader_index = leader_index
            self.log('Leader changed! New cost:', self.costs[leader_index], 'leader id', leader_index, c='GREEN')
            self.log('Old leader:', old_leader_index, 'cost:', self.costs[old_leader_index], c='RED')
            if self.costs[leader_index] == 0:
                with open('./zero', 'r') as f:
                    self.log(f.read(), c='RED2')

        # Make sure leader's sails is set to 0
        self.sails[self.leader_index] = 0
        self.v[self.leader_index] = 0

    def update_map(self):
        #  Maximization
        if self.problem == 'max':
            #  Reverse so it's decending [::-1] (Maximizing needs higher to lower)
            #  Skip the first one cause it the leader
            best_indices = self.costs.argsort()[::-1][1:self.ms+1]
        #  Minimization
        else:
            #  we skip the first indice (leader) and select 1 one more
            best_indices = self.costs.argsort()[1:self.ms+1]

        #  Initialize map
        if self.map is None:
            self.map = self.ships[best_indices]
            self.map_costs = self.costs[best_indices]
        else:
            # !- WORKS ONLY FOR MINIMIZATION -!
            for bi in best_indices:
                for i in range(self.ms):
                    if self.costs[bi] < self.map_costs[i]:
                        self.map[i] = self.ships[bi]
                        self.map_costs[i] = self.costs[bi]
                        break

    def update_private_map(self):
        if self.private_map_costs is None:
            self.private_map_costs = self.costs.copy()
            return

        for i in range(self.num_ships):
            if self.costs[i] < self.private_map_costs[i]:
                self.private_map_costs[i] = self.costs[i]
                self.private_map[i] = self.ships[i]

    def generate_tale(self):

        map_fitness = np.max(self.map_costs) - self.map_costs
        s = np.sum(map_fitness)

        #  Cost is zero or all the same
        if s == 0:
            return False

        selection_probs = map_fitness.astype(np.float64) / s

        tale = np.full((self.dimensions), 0.0, dtype=self.dtype)

        for axis in range(self.dimensions):
            tale[axis] = np.random.choice(self.map[:,axis], p=selection_probs)

        result = self.cost_func(tale)
        
        # اگر نتیجه یک tuple است، فقط مقدار خطا را استخراج می‌کنیم
        if isinstance(result, tuple) and len(result) == 2:
            tale_cost, _ = result
        else:
            tale_cost = result

        if tale_cost < self.costs[self.leader_index]:
            self.log('Tale exchanged with the leader!', self.costs[self.leader_index], '--->', tale_cost, c='YELLOW')
            self.ships[self.leader_index] = tale
            self.costs[self.leader_index] = tale_cost
            self.sails[self.leader_index] = 0
            return

        for i in range(self.ms):
            #  We already have a position in map with same cost
            #  thus ignore this one
            if tale_cost == self.map_costs[i]:
                break

            if tale_cost < self.map_costs[i]:
                self.log('A better location found by tales, cost:', self.map_costs[i], '-->', tale_cost, c='BLUE')
                self.map[i] = tale
                self.map_costs[i] = tale_cost
                break

    def update_top_ships(self):
        # Get a sorted list of ships
        sorted_indices = self.costs.argsort()

        # Remove leader
        # -------------------------
        # We can use sorted_indices[1:num_top_ships+1] to exludele leader. However in some
        # cases when there are multiple ships with same fitness as leader it
        # would include leader in top pirates list too. So it's better to
        # remove the leader from indices then select some ships.
        sorted_indices = np.delete(sorted_indices, np.argwhere(sorted_indices == self.leader_index))

        # Recheck for leader
        if self.leader_index in sorted_indices:
            print('Leader is in top pirates list')
            print(self.leader_index)
            print(sorted_indices)
            raise ValueError

        # Set indinces and tp list
        self.top_ships_indices = sorted_indices[:self.num_top_ships]
        self.top_ships = self.ships[self.top_ships_indices]


    def update_vtable(self):
        #  Leader pisitions (Values)
        leader = self.ships[self.leader_index]

        #  Euclidean distance
        norms = np.array([np.linalg.norm(self.ships[i] - leader) for i in range(self.num_ships)])
        self.vtable = norms < self.r
        self.vtable[self.leader_index] = False

    def update_wind(self):
        maxw = self.max_wind - (self.iter / self.max_iter * self.max_wind)
        minw = maxw - 0.02
        minw = 0 if minw < 0 else minw
        self.wind = np.random.uniform(minw, maxw)

        # To plot wind change during time
        self.w_history.append(self.wind)

    def cool_down(self):
        # Cooldown the wind and velocity when max_wind is high
        self.log('Cooling down...', c='VIOLET')
        self.max_wind = 1
        self.v = np.random.rand(self.num_ships, self.dimensions)
        self.v[self.leader_index] = 0
        self.sails[self.leader_index] = 0

    def update_velocity_li(self):
        leader = self.ships[self.leader_index]

        #  Calculate acceleration
        a = self.cal_acceleration()

        if np.any(self.v > 1e40):
            self.cool_down()

        # First step to update velocity, v(t) * a(t+1) + (Md: next lines)
        self.v *= a

        # People who know where leader is
        kleader = np.where(self.vtable)[0]
        if self.leader_index in kleader:
            index = np.argwhere(kleader==self.leader_index)
            kleader = np.delete(kleader, index)
            print('Leader itself is in kleader, removed!')

        # Top n ships
        top_ships = self.top_ships_indices

        #  A mask corespond to regular pirates
        #  Exluding top ships, and the one who see leader
        regular_pirates= np.ones(self.num_ships, dtype=bool)
        regular_pirates[kleader] = False
        regular_pirates[top_ships] = False

        # Random Treasure map used by top ships
        map_id = rw.select(costs=self.map_costs, num=len(top_ships), replace=True)

        # Random Top Ship used by regular Pirates
        ts_id = rw.select(costs=self.costs[self.top_ships_indices], num=np.count_nonzero(regular_pirates), replace=True)

        #  The one who know where leader is
        self.v[kleader] += self.c['leader'] * np.random.rand() * (leader - self.ships[kleader])
        self.v[kleader] += self.c['private_map'] * np.random.rand() * (self.private_map[kleader] - self.ships[kleader])

        #  The top ships
        self.v[top_ships] += self.c['map'] * np.random.rand() * (self.map[map_id] - self.ships[top_ships])
        self.v[top_ships] += self.c['private_map'] * np.random.rand() * (self.private_map[top_ships] - self.ships[top_ships])

        #  The regular pirates
        self.v[regular_pirates] += self.c['top_ships'] * np.random.rand() * (self.top_ships[ts_id] - self.ships[regular_pirates])
        self.v[regular_pirates] += self.c['private_map'] * np.random.rand() * (self.private_map[regular_pirates] - self.ships[regular_pirates])

        self.v[self.leader_index] = 0
        self.v_history.append(np.mean(self.v))


    def update_velocity(self):
        leader = self.ships[self.leader_index]

        #  Calculate acceleration
        a = self.cal_acceleration()

        for i in range(self.num_ships):
            if any(self.v[i] > 1e40):
                self.cool_down()

            # Randomly select a top ship id
            ts_id = rw.select(costs=self.costs[self.top_ships_indices], num=1)

            # Randomly select a map id
            map_id = rw.select(costs=self.map_costs, num=1)

            if i == self.leader_index and self.iter > 1:
                if any(self.v[i] != 0):
                    print('While updating velocity at iteration', self.iter, 'leader has velocity!')
                    raise False

            self.v[i] *= a[i]

            if self.vtable[i]:
                self.v[i] += self.c['leader'] * np.random.rand() * (leader - self.ships[i])
                self.v[i] += self.c['private_map'] * np.random.rand() * (self.private_map[i] - self.ships[i])
            elif i in self.top_ships_indices:
                self.v[i] += self.c['map'] * np.random.rand() * (self.map[map_id] - self.ships[i])
                self.v[i] += self.c['private_map'] * np.random.rand() * (self.private_map[i] - self.ships[i])
            else:
                self.v[i] += self.c['top_ships'] * np.random.rand() * (self.top_ships[ts_id] - self.ships[i])
                self.v[i] += self.c['private_map'] * np.random.rand() * (self.private_map[i] - self.ships[i])
                #  self.v[i] += 2.0 * np.random.rand() * (np.random.rand(self.dimensions) - self.ships[i])

        self.v[self.leader_index] = 0
        self.v_history.append(np.mean(self.v))

    def get_best_worst(self):
        # Lowest cost to highest cost (best to worst)
        sorted_indexes = np.argsort(self.costs)

        # Best and worst ships
        best = sorted_indexes[0] # index 0 is leader which should have the self.v of zero
        worst = sorted_indexes[-1]

        # Returns id not actual ship values
        return best, worst

    def cal_acceleration(self):
        #  Get besst and wrost ships
        best_id, worst_id = self.get_best_worst()

        if self.costs[best_id] == self.costs[worst_id]:
            m = np.array([1] * self.num_ships)
        else:
            # range between 1 and 2
            m = ( (self.costs - self.costs[worst_id]) / (self.costs[best_id] - self.costs[worst_id]) ) + 1

        #  Now we are using 1 and 0.7 constants instead of sing(sails)
        #  --> sin(pi/2) and sin(pi/4)
        f = (self.wind ** 2) * self.sails.reshape((self.num_ships, 1)) * 1
        #  f = (self.wind ** 2) * np.sin(self.sails.reshape((self.num_ships, 1))) * 1

        a = f / m.reshape((self.num_ships,1))

        # To keep track of acceleration and f history during time
        self.f_history.append(np.mean(f))
        self.a_history.append(np.mean(a))

        return a

    def update_trajectory(self):
        self.trajectory.append(self.ships[1][1])

    def update_positions(self):
        bk = self.ships.copy()
        self.ships += self.v
        self.scale_ships()

        # ships without movements = swm
        swm = []
        for i in range(self.num_ships):
            if i == self.leader_index:
                continue
            if np.array_equal(bk[i], self.ships[i]):
                swm.append(i)

        if len(swm) > 0:
            self.log('-'*40, c='GREY')
            self.log('Ships are not moving around:', swm, c='GREY')
            self.log('Velocities:', self.v[swm], c='GREY')
            self.log('Sails:', self.sails[swm], c='GREY')
            self.log('-'*40, c='GREY')

    def scale_ships(self):
        for d in range(self.dimensions):
            w_max = np.where(self.ships[:, d] > self.fmax[d])
            #  Scale using highest value
            if w_max[0].size > 0:
                #  Index of max value in self.ships - argmax returns the index
                #  of w_max
                index = w_max[0][np.argmax(self.ships[w_max, d])]
                value = self.ships[index, d]
                scale = value / self.fmax[d]
                self.ships[w_max, d] /= scale

            w_min = np.where(self.ships[:, d] < self.fmin[d])
            if w_min[0].size > 0:
                index = w_min[0][np.argmin(self.ships[w_min, d])]
                value = self.ships[index, d]
                # Fix division by zero
                if self.fmin[d] == 0:
                    self.ships[w_min, d] *= 0
                else:
                    scale = value / self.fmin[d]
                    self.ships[w_min, d] /= scale

    def battle(self):
        # Don't re-spawn at final iterations
        sink = False if self.iter > (self.max_iter / 3) else True

        for i in range(self.num_top_ships):

            select_from = self.top_ships_indices

            if self.leader_index in select_from:
                print('Leader is in battle list')
                raise ValueError

            #  Roullet wheel
            s1, s2 = rw.select(costs=self.costs[select_from], num=2)
            ship_1 = select_from[s1]
            ship_2 = select_from[s2]

            fitness = rw.fitness

            if np.sum(fitness) != 0:
                alfa = fitness[s1] / (fitness[s1] + fitness[s2])
                beta = fitness[s2] / (fitness[s1] + fitness[s2])
            else:
                # They have same fitness but might not be on same position
                alfa = 0.5
                beta = 0.5

            new_ship = alfa * self.ships[ship_1] + beta * self.ships[ship_2]
            result = self.cost_func(new_ship)
            
            # اگر نتیجه یک tuple است، فقط مقدار خطا را استخراج می‌کنیم
            if isinstance(result, tuple) and len(result) == 2:
                new_ship_cost, _ = result
            else:
                new_ship_cost = result

            if new_ship_cost < self.costs[ship_1] and new_ship_cost < self.costs[ship_2]:
                #  CAPTURE --
                #  new ship is better than of both old ships
                if self.costs[ship_1] < self.costs[ship_2]:
                    self.log('ship1 captures ship2', self.costs[ship_1], '-->', new_ship_cost, c='YELLOW2')
                    #  self.ships[ship_2] = self.ships[ship_1]
                    self.ships[ship_1] = new_ship
                    #  Re-spawn ship 2
                    if sink:
                        self.ships[ship_2] = np.random.rand(self.dimensions)
                # When ship2 is better or cost of ship1 and ship2 are same
                else:
                    self.log('ship2 captures ship1', self.costs[ship_2], '-->', new_ship_cost, c='YELLOW2')
                    #  self.ships[ship_1] = self.ships[ship_2]
                    self.ships[ship_2] = new_ship
                    #  Re-spawn ship 1
                    if sink:
                        self.ships[ship_1] = np.random.rand(self.dimensions)
            elif new_ship_cost > self.costs[ship_1] and new_ship_cost > self.costs[ship_2]:
                #  New ship is not better than of any of ships engaging in battle
                #  -- SINK --
                #  But don't sink anything at final iterations
                if sink == True:
                    #  self.log('SINK...', c='YELLOW2')
                    if self.costs[ship_1] < self.costs[ship_2]:
                        #  self.log('ship1 sinks ship2')
                        #  Re-spawn ship 2
                        self.ships[ship_2] = np.random.rand(self.dimensions)
                    elif self.costs[ship_2] < self.costs[ship_1]:
                        #  self.log('ship2 sinks ship1')
                        #  Re-spawn ship 1
                        self.ships[ship_1] = np.random.rand(self.dimensions)
                    #  else:
                        # same cost for both ships in battle
                        # draw
                        #  pass

    def hurricane(self):
        # Don't run hurricane after T/landa of algorithm , landa=3
        if self.iter >  (self.max_iter / 3):
            return False

        prob_matrix = np.random.uniform(0, 1, size=(self.num_ships))

        #  Skip leader
        prob_matrix[self.leader_index] = 0
        indices = np.argwhere(prob_matrix > self.hr)

        if self.leader_index in indices:
            print('Leader is in hurricane list')
            raise ValueError

        if any(indices):
            self.log('Hurricane! iteration:', self.iter)

        random_positions = np.random.uniform(self.fmax, self.fmin, size=(len(indices), self.dimensions))

        #  np.put(self.ships, indices, random_positions)

        for i in range(len(indices)):
            self.ships[indices[i]] = random_positions[i]

        self.scale_ships()

    def update_bsf(self):
        #  Refactored
        self.bsf_position = self.ships[self.leader_index]
        self.bsf_list.append(self.costs[self.leader_index])
        # If bsf costs list is not empty
        if len(self.bsf_list) >= 2:
            # در حالت کمینه‌سازی، هزینه کمتر بهتر است
            if self.problem == 'min':
                # اگر هزینه جدید بیشتر از قبلی باشد، اشتباه است
                if self.bsf_list[-2] < self.costs[self.leader_index]:
                    print('BSF being updated wrong...')
                    print('Old:', self.bsf_list[-2], '-->', 'New:', self.costs[self.leader_index])
                    print('iteration', self.iter, 'leader index', self.leader_index, 'leder velo', self.v[self.leader_index])
                    # به جای ایجاد خطا، مقدار قبلی را نگه می‌داریم
                    self.bsf_list[-1] = self.bsf_list[-2]
            # در حالت بیشینه‌سازی، هزینه بیشتر بهتر است
            elif self.problem == 'max':
                # اگر هزینه جدید کمتر از قبلی باشد، اشتباه است
                if self.bsf_list[-2] > self.costs[self.leader_index]:
                    print('BSF being updated wrong...')
                    print('Old:', self.bsf_list[-2], '-->', 'New:', self.costs[self.leader_index])
                    print('iteration', self.iter, 'leader index', self.leader_index, 'leder velo', self.v[self.leader_index])
                    # به جای ایجاد خطا، مقدار قبلی را نگه می‌داریم
                    self.bsf_list[-1] = self.bsf_list[-2]

    def update_avg(self):
        self.avg.append(np.average(self.costs))

    def show_positions(self):
        if self.dimensions >= 2:
            # Ignore more than of 2 dimension
            plt.figure(2)
            x, y, *_ = self.ships.T
            #  x_b, y_b, *_ = self.bsf_position.T
            x_l, y_l, *_ = self.ships[self.leader_index].T
            x_t, y_t, *_ = self.top_ships.T
            x_m, y_m, *_ = self.map.T

            # Find particles that have not converged
            w = np.where(self.costs > self.costs[self.leader_index] + 100)[0]
            x_uc, y_uc, *_  = self.ships[w].T

            xlim = (self.fmax[0] - self.fmin[0])
            xlim_min = x_l - xlim
            xlim_max = x_l + xlim
            ylim_min = y_l - xlim
            ylim_max = y_l + xlim

            plt.xlim(xlim_min, xlim_max)
            plt.ylim(ylim_min, ylim_max)

            plt.scatter(x_l, y_l, c='deeppink', s=250, label='Leader', marker='*')
            plt.scatter(x_m, y_m, c='deepskyblue', s=150, label='Map', marker='P')
            plt.scatter(x, y, c='lightgreen', s=30, label='Ships', marker='o')
            plt.scatter(x_t, y_t, c='black', s=10, label='Top Ships', marker='.')

            if self.iter == self.max_iter:
                plt.scatter(x_uc, y_uc, c='dimgray', s=20, label='Un-Conv', marker='x')

            plt.legend()
            plt.show()

    def show_results(self):
        if self.problem == 'min':
            label = 'Cost :'
        else:
            label = 'Fitness :'

        print('10 last AVG: \n', self.avg[-9:], '\n')
        print('10 last BSF changes: \n', np.unique(self.bsf_list)[:10][::-1], '\n')
        print('BSF Changed:', len(np.unique(self.bsf_list)), 'times.\n')
        print()
        print(label, self.cost_func(self.bsf_position))
        print('BSF Positions: ', self.bsf_position)

    def plot_results_curves(self):
        data = [self.bsf_list, self.avg, self.trajectory, self.v_history, self.w_history, self.a_history, self.f_history]
        helpers.plot_results_curves(data, block=False)

    def animate_ships(self):
        if not self.animate: #or self.dimensions != 2:
            return False

        plt.ion()
        self.animate_fig, ax = plt.subplots()

        #  Just a placeholder
        arr = np.array([])

        x, y = arr, arr

        #  leader
        self.sc_leader = ax.scatter(x, y, c='deeppink', s=350, label='Leader', marker='*')

        #  map
        self.sc_map = ax.scatter(x, y, s=150, c='deepskyblue', marker='P', label='Map')

        #  ships
        self.sc_ships = ax.scatter(x, y, c='lightgreen', s=10, label='Ships', marker='.')

        #  Top Ships
        self.sc_top_ships = ax.scatter(x, y, c='orange', s=10, label='Top Ships', marker='o')

        #  r
        self.r_circle = plt.Circle((0, 0), 0.0, color='mediumorchid', fill=False)
        plt.gca().add_patch(self.r_circle)

        self.ax = ax

        self.animate_fig.legend()

        #  plt.draw()
        plt.show()

    def update_animation(self):
        if not self.animate:
            return

        x, y, *_ = self.ships.T
        self.sc_ships.set_offsets(np.c_[x, y])

        #  Top ships
        x_t, y_t, *_ = self.top_ships.T
        self.sc_top_ships.set_offsets(np.c_[x_t, y_t])

        x_l, y_l, *_ = self.ships[self.leader_index]
        self.sc_leader.set_offsets(np.c_[x_l, y_l])

        self.r_circle.center = x_l, y_l
        self.r_circle.set_radius(self.r)

        x_m, y_m, *_ = self.map.T
        self.sc_map.set_offsets(np.c_[x_m, y_m])

        #  Reposition plot based on leader place in such a way
        #  that leader stays at center
        xlim = (self.fmax[0] - self.fmin[0])
        xlim_min = x_l - xlim
        xlim_max = x_l + xlim
        ylim_min = y_l - xlim
        ylim_max = y_l + xlim

        self.ax.set_xlim(xlim_min, xlim_max)
        self.ax.set_ylim(ylim_min, ylim_max)

        self.animate_fig.canvas.draw_idle()
        plt.pause(0.1)


    def update_r(self):
        #  Dynamicly and linearly increase r
        if not self.max_r:
            self.r = 0
            return
        self.r = self.iter / self.max_iter * self.max_r

    def run_iteration(self):
        """
        اجرای یک تکرار از الگوریتم و برگرداندن بهترین هزینه و پارامترها
        
        Returns:
        --------
        tuple (float, numpy.ndarray, dict)
            هزینه بهترین موقعیت، آرایه پارامترهای آن و معیارهای ارزیابی
        """
        self.update_r()
        
        # به‌روزرسانی شماره تکرار
        self.iter += 1
        
        # محاسبه هزینه‌ها
        error_and_metrics = self.cal_costs()
        self.update_avg()
        
        self.update_private_map()
        
        # یافتن رهبر
        self.update_leader()
        self.update_bsf()
        
        # به‌روزرسانی جدول دید
        self.update_vtable()
        
        # به‌روزرسانی بادبان‌ها
        self.update_sails()
        
        # ایجاد نقشه و داستان
        self.update_map()
        self.generate_tale()
        
        self.update_top_ships()
        
        # به‌روزرسانی باد
        self.update_wind()
        
        # به‌روزرسانی سرعت و موقعیت‌ها
        self.update_velocity()
        self.update_positions()
        
        self.battle()
        self.update_leader()
        
        # اجرای طوفان در صورت نیاز
        self.hurricane()
        
        self.update_trajectory()
        
        # به‌روزرسانی انیمیشن
        self.update_animation()
        
        # برگرداندن بهترین هزینه، پارامترها و معیارها
        if error_and_metrics and isinstance(error_and_metrics, tuple) and len(error_and_metrics) == 2:
            error, metrics = error_and_metrics
            return error, self.ships[self.leader_index], metrics
        else:
            # اگر تابع هزینه metrics را برنگرداند، یک دیکشنری خالی برمی‌گردانیم
            default_metrics = {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
            return self.costs[self.leader_index], self.ships[self.leader_index], default_metrics

    def start(self):

        self.animate_ships()

        for i in range(1, self.max_iter+1):

            #  Randomly initialized
            #  Or new positions from last iter

            self.update_r()

            #  make iterations available to all functions
            self.iter = i

            #  1. Cal costs
            self.cal_costs()
            self.update_avg()

            self.update_private_map()

            #  2. Find the leader
            self.update_leader()
            self.update_bsf()

            #  4. Generate visibility table
            self.update_vtable()

            #  5. Update sails
            self.update_sails()

            #  6. Create map and tale
            self.update_map()
            self.generate_tale()

            self.update_top_ships()

            # Show plots in defined iterations if there is
            # no animated and quiet is set to false
            if self.iteration_plots and not self.animate and not self.quiet:
                if i in [1, 5, 50, 100, 200, 400, 500]:
                    self.show_positions()

            #  7. Wind
            self.update_wind()

            #  8. Velocity and positions
            self.update_velocity()
            self.update_positions()

            self.battle()
            self.update_leader()

            #  9. If any hurricane
            self.hurricane()

            self.update_trajectory()

            #  A reference to the real function or pass when animate=False
            self.update_animation()


        #  Recalculate after last iteration and position changes
        self.cal_costs()
        self.update_leader()
        self.update_bsf()

        #  Close animation plot
        #  if self.animate and self.dimensions == 2:
        if self.animate:
            print('End of algorithm...\n\n')
            plt.waitforbuttonpress(0)
            plt.close()
            #  Fix not showing plots below
            matplotlib.interactive(False)

        if not self.quiet:
            self.show_results()
            self.plot_results_curves()
            self.show_positions()

    def search(self):
        """
        Run the optimization algorithm and return best results
        
        Returns:
        --------
        tuple
            (best_position, best_cost, best_metrics)
        """
        # Run the optimization algorithm
        self.start()
        
        # Get results from cal_costs
        result = self.cal_costs()
        
        if result is not None:
            best_cost, best_metrics = result
        else:
            best_cost = self.costs[self.leader_index]
            best_metrics = {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Return best position, cost, and metrics
        return self.ships[self.leader_index], best_cost, best_metrics
