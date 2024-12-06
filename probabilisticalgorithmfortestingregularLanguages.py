import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# Define the dynamic automaton with SCC integration
class Automaton:
    def __init__(self, alphabet, max_states):
        self.states = set()
        self.alphabet = alphabet
        self.transitions = defaultdict(dict)
        self.start_state = None
        self.accept_states = set()
        self.max_states = max_states
        self.sccs = []  # Strongly Connected Components

    def add_transition(self, from_state, symbol, to_state):
        """Add a transition dynamically."""
        self.states.add(from_state)
        self.states.add(to_state)
        self.transitions[from_state][symbol] = to_state

    def set_start_state(self, state):
        """Set the start state."""
        self.start_state = state
        self.states.add(state)

    def add_accept_state(self, state):
        """Add an accepting state."""
        self.accept_states.add(state)
        self.states.add(state)

    def transition(self, state, symbol):
        """Transition function."""
        return self.transitions.get(state, {}).get(symbol, None)

    def dynamically_adjust_transitions(self, recursive_inputs):
        """Dynamically adjust transitions to align with all recursive inputs."""
        state_counter = len(self.states)  # Start counter for new states
        for input_word in recursive_inputs:
            current_state = self.start_state
            for symbol in input_word:
                if symbol not in self.transitions[current_state]:
                    if len(self.states) >= self.max_states:
                        raise ValueError("State limit exceeded.")
                    next_state = state_counter
                    state_counter += 1
                    self.add_transition(current_state, symbol, next_state)
                current_state = self.transition(current_state, symbol)
            self.add_accept_state(current_state)

    def decompose_scc(self):
        """Decompose the automaton into strongly connected components."""
        index, lowlink = {}, {}
        stack, on_stack = [], set()
        self.sccs = []
        current_index = 0

        def strong_connect(v):
            nonlocal current_index
            index[v] = lowlink[v] = current_index
            current_index += 1
            stack.append(v)
            on_stack.add(v)

            for symbol in self.alphabet:
                next_state = self.transition(v, symbol)
                if next_state:
                    if next_state not in index:
                        strong_connect(next_state)
                        lowlink[v] = min(lowlink[v], lowlink[next_state])
                    elif next_state in on_stack:
                        lowlink[v] = min(lowlink[v], index[next_state])

            if lowlink[v] == index[v]:
                scc = set()
                while stack:
                    node = stack.pop()
                    on_stack.remove(node)
                    scc.add(node)
                    if node == v:
                        break
                self.sccs.append(scc)

        for state in self.states:
            if state not in index:
                strong_connect(state)

# Recursive Input Generator
def generate_recursive_inputs(base_word="010", iterations=1000):
    """Efficient recursive input generation using a generator."""
    word = base_word
    for _ in range(iterations):
        newword = "1" + word + "0"
        word=word+newword
        yield word

# Feasibility Check for Runs
def is_feasible_run(automaton, run, start_state):
    """Check if a run is feasible for the automaton."""
    state = start_state
    for symbol in run:
        state = automaton.transition(state, symbol)
        if state is None:
            return False
    return state in automaton.accept_states

# Generate Random Runs
def sample_runs(word, run_length, num_samples):
    """Sample random runs of a given length from the word."""
    n = len(word)
    return [word[random.randint(0, n - run_length):][:run_length] for _ in range(num_samples)]

# Probabilistic Test for Regular Languages
def probabilistic_test(word, automaton, epsilon):
    """Optimized probabilistic test."""
    n = len(word)
    k, m = len(automaton.states), len(automaton.alphabet)

    # Efficient parameter calculation
    S = max(1, int(129 * k * m * math.log(1 / epsilon) / epsilon))
    interval_size = n // S
    intervals = [(i * interval_size, min((i + 1) * interval_size, n)) for i in range(S)]

    for i in range(1, int(math.log2(8 * k * m / epsilon)) + 1):
        run_length = 2 ** (i + 1)
        num_samples = max(1, int((2 ** (8 - i)) * k ** 2 * m / (epsilon ** 2)))

        sampled_runs = sample_runs(word, run_length, num_samples)

        for run in sampled_runs:
            start_state = automaton.start_state
            if not any(
                interval_start <= word.index(run[0]) < interval_end
                and is_feasible_run(automaton, run, start_state)
                for interval_start, interval_end in intervals
            ):
                return False
    return True

# Main Execution with Visualization
if __name__ == "__main__":
    max_states = 10000
    automaton = Automaton(["0", "1"], max_states)

    # Generate and dynamically adjust automaton for inputs
    base_word = "010"
    recursive_inputs = generate_recursive_inputs(base_word, iterations=500)

    try:
        automaton.set_start_state("q0")
        automaton.dynamically_adjust_transitions(recursive_inputs)
    except ValueError as e:
        print(f"Error during transition adjustment: {e}")

    # Decompose into SCCs
    automaton.decompose_scc()
    print(f"Decomposed into {len(automaton.sccs)} SCCs.")

    # Test probabilistic membership for generated inputs
    epsilon = 0.1
    results = {"YES": 0, "NO": 0}
    total_inputs = 500

    for test_word in generate_recursive_inputs(base_word, iterations=total_inputs):
        try:
            result = probabilistic_test(test_word, automaton, epsilon)
            if result:
                results["YES"] += 1
            else:
                results["NO"] += 1
        except ValueError as e:
            print(f"Error during testing: {e}")
            break

    # Plot probabilities
    plt.figure(figsize=(8, 5))

# Compute probabilities
    total_tests = sum(results.values())
    probabilities = {key: value / total_tests for key, value in results.items()}

# Extract labels and probabilities for plotting
    labels = list(probabilities.keys())
    prob_values = list(probabilities.values())

# Plot the probabilities
    plt.bar(labels, prob_values, color=["green", "red"], alpha=0.7)
    plt.title(f"Probabilistic Test Probabilities for {total_inputs} Recursive Inputs")
    plt.xlabel("Result")
    plt.ylabel("Probability")
    plt.xticks(labels)
    plt.ylim(0, 1)  # Ensure y-axis is in probability range [0, 1]
    plt.tight_layout()
    plt.show()


    print(f"Total Inputs Tested: {total_inputs}")
    print(f"YES: {results['YES']}, NO: {results['NO']}")
