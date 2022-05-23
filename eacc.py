"""
Entanglement-assisted classical communication see-saw for the channel
inequality `2[p(0,0) + p(1,1) + p(2,2)] + p(3,0) + p(3,1) + p(3,2) <= 4`.

Usage:
>>> import eacc
>>> prob, states, meas = eacc.driver()
>>> eacc.check_result(prob, states, meas)
"""

from itertools import chain
import picos as pc
import numpy as np

# TODO: Change to programatically set the entanglement dimension.
N_X = 3  # Nof. preparations.
N_C = 2  # Dim. of preparations.
D_B = 2  # Dim. of Bob's system (i.e., his initial resource)
N_B = 4  # Nof. Bob's outputs.
N_Y = 1  # Nof. Bob's measurements.
N_ROUNDS = 30
N_STARTS = 15 
SOLVER = "mosek"
ATOL = 10e-10

#-----------------------------
# Utilitary functions:
#-----------------------------


def flat(llst):
    return list(chain.from_iterable(llst))

def nested_map(f, llst):
    return [list(map(f, lst)) for lst in llst]

def dag(matrix):
    return matrix.conj().T

def outer(vec1, vec2=None):
    """Outer product (with complex conjugation) between `vec1` and `vec2`

    If `vec2` is not supplied, return outer product of `vec1` with itself

    Args:
        vec1: ndarray either with shape (n,) or (n,1)
        vec2: ndarray either with shape (n,) or (n,1)

    Returns:
        ndarray: outer product (with complex conjugation) between `vec1` and
            `vec2`, or between `vec1` and itself it `vec2` not given.
"""

    if vec1.ndim == 1:
        vec1 = vec1[:,None]
    if vec2:
        if vec2.ndim == 1:
            vec2 = vec2[:,None]
    else:
        vec2 = vec1
    return vec1 @ dag(vec2)

def randnz(shape, norm=1/np.sqrt(2)):
    """Normally distributed complex number matrix (Ginibre ensemble)."""

    real = np.random.normal(0, 1, shape)
    imag = 1j * np.random.normal(0, 1, shape)
    return (real + imag) * norm

def random_unitary_haar(dim=2):
    """Random unitary matrix according to Haar measure.

    Ref.: https://arxiv.org/abs/math-ph/0609050v2
    """

    q, r = np.linalg.qr(randnz((dim, dim)))
    m = np.diagonal(r)
    m = m / np.abs(m)
    return np.multiply(q, m, q)

def random_pure_state(dim=2, density=True):
    """Generates a random pure quantum state of dimension `dim` in Haar measure.

    Takes first column of a Haar-random unitary operator.

    Args:
        dim: dimension of the state vectors (2 for qubits, 3 for qutrits etc.)
        density: if `True`, returns a density matrix instead of state vector.

    Returns:
        ndarray: a `dim`-length state vector if `density == False`, else a
            `dim x dim` density operator.
    """

    st = random_unitary_haar(dim)[:,0]
    if density:
        st = outer(st)
    return st

#-----------------------------
# Implementation:
#-----------------------------

def objective_function(states, meas):
    """ 2[p(0,0) + p(1,1) + p(2,2)] + p(3,0) + p(3,1) + p(3,2) <= 4 """
    terms = []
    for i in range(N_X):
        terms.append(2 * sum([pc.trace(states[i][c] * meas[c][i]) for c in range(N_C)]))
        terms.append(sum([pc.trace(states[i][c] * meas[c][3]) for c in range(N_C)]))
    return sum(terms)

def optimize_states(meas, verb=1):
    prob = pc.Problem()
    states = [[pc.HermitianVariable(f"st({i},{j})", D_B) for i in range(N_C)] for j in range(N_X)]
    [prob.add_constraint(sum(states[x]) == sum(states[x + 1])) for x in range(N_X - 1)]
    prob.add_list_of_constraints([st >> 0 for st in flat(states)])
    prob.add_list_of_constraints([pc.trace(sum(st)) == 1 for st in states])
    prob.set_objective("max", objective_function(states, meas))
    return prob.solve(solver=SOLVER, verbose=verb), states
    
def optimize_meas(states, verb=1):
    prob = pc.Problem()
    meas = [[pc.HermitianVariable(f"meas({i},{j})", D_B) for i in range(N_B)] for j in range(N_C)]
    prob.add_list_of_constraints([m >> 0 for m in flat(meas)])
    prob.add_list_of_constraints([sum(m) == np.eye(D_B) for m in meas])
    prob.set_objective("max", objective_function(states, meas))
    return prob.solve(solver=SOLVER, verbose=verb), meas

def random_subnormalized_states():
    # Did uniform normalization because it is easier.
    return [[random_pure_state(dim=D_B) / N_C for i in range(N_C)] for j in range(N_X)]

def see_saw(N_ROUNDS=N_ROUNDS, states=None, verb=0):
    if not states: states = random_subnormalized_states()
    for _ in range(N_ROUNDS):
        # I convert to np.array so that Picos doesn't complain about nonconstants.
        prob, meas = optimize_meas(nested_map(np.array, states), verb)
        prob, states = optimize_states(nested_map(np.array, meas), verb)
        print(f"   {prob.value}")
    return prob, states, meas

def driver(N_STARTS=N_STARTS):
    print(f"Starting parameter 1/{N_STARTS}.")
    prob, states, meas = see_saw()
    for _ in range(N_STARTS - 1):
        print(f"Starting parameter {_ + 2}/{N_STARTS}.")
        new_prob, new_states, new_meas = see_saw()
        if new_prob.value > prob.value:
            prob, states, meas = new_prob, new_states, new_meas
    print(f"Optimal objective value: {prob.value}")
    return prob, states, meas

#-----------------------------
# Result verification:
#-----------------------------

def is_herm(matrix):
    """Check whether `matrix` is Hermitian."""

    return np.allclose(matrix, matrix.conj().T)


def is_psd(matrix):
    """Test `matrix` for positive semi-definiteness."""

    if is_herm(matrix):
        return np.all(np.linalg.eigvalsh(matrix) >= - ATOL)
    else:
        return False

def is_meas(meas):
    """Check whether `meas` is a well-defined quantum measurement.

    Args:
        meas (list): list of ndarrays representing the measurement's effects.

    Returns:
        bool: returns `True` iff `meas` is composed of effects which are:
            - Square matrices with the same dimension,
            - Positive semi-definite, and
            - Sum to the identity operator.
    """

    dims = meas[0].shape

    try:
        square = len(dims) == 2 and dims[0] == dims[1]
        same_dim = np.all(np.asarray([eff.shape for eff in meas]) == dims)
        psd = np.all([is_psd(ef) for ef in meas])
        complete = np.allclose(sum(meas), np.eye(dims[0]))
    except (ValueError, np.linalg.LinAlgError):
        return False

    return square and same_dim and psd and complete

def is_dm(matrix):
    """Check whether `matrix` is a well-defined density matrix.

    Args:
        matrix (ndarray): density operator to test.

    Returns:
        bool: returns `True` iff `matrix` has unit-trace, is positive semi-definite
            and is hermitian.
    """

    try:
        trace_one = np.isclose(np.trace(matrix), 1)
        psd = is_psd(matrix)
        herm = is_herm(matrix)
    except (ValueError, np.linalg.LinAlgError):
        return False

    return trace_one and psd and herm

def check_result(states, meas):
    are_states = np.alltrue([is_dm(np.array(sum(st))) for st in states])
    are_measurements = np.alltrue([is_meas(m) for m in nested_map(np.array, meas)])
    return are_states & are_measurements