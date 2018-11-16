import numpy as np
import logging as log
import random as rand

dlog = log.getLogger('default')
dlog.setLevel(log.INFO)

class NeuronState:
    ReadyForStimulus = 'READY_FOR_STIMULUS'
    Depolarisation = 'DEPOLARISATION'
    Repolarisation = 'REPOLARISATION'

class SpikingNeuron:
    num_of_sn = 0
    def __init__(self, name='', resting_charge=0, charge_decay=0.2, threshold=2, frames_to_depolarise=1,
                 frames_to_repolarise=1):
        self.resting_charge = resting_charge
        self.charge_decay = charge_decay
        self.threshold = threshold
        self.frames_to_depolarize = frames_to_depolarise
        self.frames_to_repolarize = frames_to_repolarise

        self.id = f'{SpikingNeuron.num_of_sn}_{name}'
        SpikingNeuron.num_of_sn += 1

        self.value = 0

        self._counter = 0
        self.current_charge = self.resting_charge
        self._state = NeuronState.ReadyForStimulus
        self._axon_connection_neurons = []
        self._dendrite_connection_neurons = []

    @property
    def num_axon_connections(self):
        return len(self._axon_connection_neurons)

    @property
    def num_dendride_connections(self):
        return len(self._dendrite_connection_neurons)

    def get_neuron_description(self):
        return f'{self.id}_{self._state}'

    def add_axon_to(self, neuron):
        self._axon_connection_neurons.append(neuron)
        neuron._dendrite_connection_neurons.append(neuron)

    def add_dendride_to(self, neuron):
        neuron.add_axon_to(self)
        self._dendrite_connection_neurons.append(neuron)

    def stimulate(self, charge=1):
        self.current_charge += charge

    def fire(self, charge=1):
        dlog.info(f'{self.get_neuron_description()} neuron fired')
        for n in self._axon_connection_neurons:
            n.stimulate(charge)

    def update(self):
        dlog.info(f'current charge is {self.current_charge} at {self.get_neuron_description()}')
        if self._state == NeuronState.ReadyForStimulus:
            if self.current_charge >= self.threshold:
                self._state = NeuronState.Depolarisation
            self.current_charge = max(self.current_charge - self.charge_decay, self.resting_charge)
        if self._state == NeuronState.Depolarisation:
            self._counter += 1
            if self._counter > self.frames_to_depolarize:
                self._counter = 0
                self.fire()
                self.value = 1
                self._state = NeuronState.Repolarisation
        if self._state == NeuronState.Repolarisation:
            self._counter += 1
            if self._counter > self.frames_to_repolarize:
                self._counter = 0
                self.value = 0
                self.current_charge = self.resting_charge
                self._state = NeuronState.ReadyForStimulus


class SpikingNeuronNetwork:
    def __init__(self, numer_of_neurons, average_connection_per_neuron, num_input_neurons, num_output_neurons):
        self.hidden_neurons = []
        self.input_neurons = []
        self.output_neurons = []
        for _ in range(numer_of_neurons):
            self.hidden_neurons.append(SpikingNeuron())
        for _ in range(int(numer_of_neurons * average_connection_per_neuron)):
            self.add_random_connection(self.hidden_neurons)

        for _ in range(num_input_neurons):
            neuron = SpikingNeuron()
            neuron.add_axon_to(self.get_random_neuron(self.hidden_neurons))
            self.input_neurons.append(neuron)
        for _ in range(num_output_neurons):
            neuron = SpikingNeuron()
            neuron.add_dendride_to(self.get_random_neuron(self.hidden_neurons))
            self.output_neurons.append(neuron)


    def get_random_neuron_idx(self, list):
        return rand.randint(0, len(list) - 1)

    def get_random_neuron(self, neuron_list):
        return neuron_list[self.get_random_neuron_idx(neuron_list)]

    def add_random_connection(self, source_list, target_list=None):
        if not target_list:
            target_list = source_list
        idx_source, idx_target = 0, 0
        while idx_source == idx_target and source_list == target_list:
            idx_source = self.get_random_neuron_idx(source_list)
            idx_target = self.get_random_neuron_idx(target_list)
        self.hidden_neurons[idx_source].add_axon_to(self.hidden_neurons[idx_target])

    def mutate(self, connections=(-1, 1), weight_adjust=(-0.1, 0.1), neurons=(-1, 1)):
        pass

    def clean_up_network(self):
        self.hidden_neurons[:] = [x for x in self.hidden_neurons if len(x._axon_connection_neurons) > 0]

    def update(self):
        for n in self.hidden_neurons:
            n.update()

    def get_output(self):
        return [x.value for x in self.output_neurons]

    def push_input(self, input_lst):
        if len(input_lst) != len(self.input_neurons):
            raise Exception(f'Received {len(input_lst)} inputs, '
                            f'but network has {len(self.input_neurons)} input neurons.')
        for neuron, inp in zip(self.input_neurons, input_lst):
            neuron.current_charge += inp


def run():
    network = SpikingNeuronNetwork(100, 5, 10, 10)
    ri = lambda: rand.randint(0,10)

    while True:
        inputs = [ri() for _ in range(10)]
        network.push_input(inputs)
        network.update()
        print(f'inputs {inputs}')
        print(network.get_output())


if __name__ == '__main__':
    run()



