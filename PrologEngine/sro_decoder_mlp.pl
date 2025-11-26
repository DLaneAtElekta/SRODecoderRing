/**
 * SRO Decoder Ring MLP - SWI-Prolog Implementation
 *
 * Implements neural network inference using relational predicates:
 *   - mat_vec(M, V_in, V_out): Matrix-vector multiplication
 *   - vec_add(A, B, C): Vector addition
 *   - activation(Type, V_in, V_out): Activation functions
 *
 * Architecture: 1 -> 6 -> 12 -> 24 -> 6 -> 1 (energy output)
 *
 * Usage:
 *   ?- consult('sro_decoder_mlp.pl').
 *   ?- mlp([0.5], Energy).
 *   ?- mlp_with_trace([0.5], Energy, Trace).
 */

:- module(sro_decoder_mlp, [
    mat_vec/3,
    vec_add/3,
    activation/3,
    dense_layer/5,
    mlp/2,
    mlp_with_trace/3,
    load_weights/1,
    export_prolog_weights/2
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

% =============================================================================
% Core Predicates
% =============================================================================

/**
 * mat_vec(+Matrix, +Vector, -Result)
 *
 * Matrix-vector multiplication predicate.
 * Matrix is a list of rows, Vector is a list of values.
 * Result[i] = sum(Matrix[i][j] * Vector[j]) for all j
 *
 * Example:
 *   ?- mat_vec([[1,2],[3,4]], [1,1], R).
 *   R = [3, 7].
 */
mat_vec(Matrix, Vector, Result) :-
    maplist(dot_product(Vector), Matrix, Result).

/**
 * dot_product(+V1, +V2, -Product)
 * Compute dot product of two vectors.
 */
dot_product(V1, V2, Product) :-
    maplist(multiply, V1, V2, Products),
    sum_list(Products, Product).

multiply(X, Y, Z) :- Z is X * Y.

/**
 * vec_add(+A, +B, -C)
 *
 * Element-wise vector addition predicate.
 * C[i] = A[i] + B[i]
 *
 * Example:
 *   ?- vec_add([1,2,3], [0.5,0.5,0.5], R).
 *   R = [1.5, 2.5, 3.5].
 */
vec_add(A, B, C) :-
    maplist(add, A, B, C).

add(X, Y, Z) :- Z is X + Y.

/**
 * activation(+Type, +V_in, -V_out)
 *
 * Apply activation function element-wise.
 * Supported types: linear, relu, swish, sigmoid, softmax
 *
 * Example:
 *   ?- activation(relu, [-1, 0, 1], R).
 *   R = [0, 0, 1].
 */
activation(linear, V, V).

activation(relu, V_in, V_out) :-
    maplist(relu_fn, V_in, V_out).

activation(swish, V_in, V_out) :-
    maplist(swish_fn, V_in, V_out).

activation(sigmoid, V_in, V_out) :-
    maplist(sigmoid_fn, V_in, V_out).

activation(softmax, V_in, V_out) :-
    softmax_fn(V_in, V_out).

% Activation function implementations
relu_fn(X, Y) :- X >= 0 -> Y = X ; Y = 0.

sigmoid_fn(X, Y) :- Y is 1 / (1 + exp(-X)).

swish_fn(X, Y) :-
    Sig is 1 / (1 + exp(-X)),
    Y is X * Sig.

softmax_fn(V_in, V_out) :-
    max_list(V_in, Max),
    maplist({Max}/[X, Y]>>(Y is exp(X - Max)), V_in, Exps),
    sum_list(Exps, Sum),
    maplist({Sum}/[E, S]>>(S is E / Sum), Exps, V_out).

% =============================================================================
% Layer Predicates
% =============================================================================

/**
 * dense_layer(+Weights, +Bias, +Activation, +V_in, -V_out)
 *
 * Dense (fully connected) layer predicate.
 * Computes: V_out = activation(V_in @ Weights + Bias)
 *
 * This is the composition:
 *   dense_layer(W, B, Act, V_in, V_out) :-
 *       mat_vec(W, V_in, Z),
 *       vec_add(Z, B, A),
 *       activation(Act, A, V_out).
 */
dense_layer(Weights, Bias, Activation, V_in, V_out) :-
    mat_vec(Weights, V_in, Z),
    vec_add(Z, Bias, A),
    activation(Activation, A, V_out).

/**
 * dense_layer_traced(+LayerId, +Weights, +Bias, +Activation, +V_in, -V_out, -Trace)
 *
 * Dense layer with trace of intermediate computations.
 */
dense_layer_traced(LayerId, Weights, Bias, Activation, V_in, V_out, Trace) :-
    mat_vec(Weights, V_in, Z),
    vec_add(Z, Bias, A),
    activation(Activation, A, V_out),
    Trace = trace(LayerId, V_in, Z, A, V_out).

% =============================================================================
% Weight Storage (Dynamic Predicates)
% =============================================================================

:- dynamic weight/3.    % weight(LayerId, RowIndex, Row)
:- dynamic bias/2.      % bias(LayerId, BiasVector)
:- dynamic layer_config/3.  % layer_config(LayerId, InFeatures, OutFeatures)
:- dynamic layer_activation/2.  % layer_activation(LayerId, ActivationType)

/**
 * get_weight_matrix(+LayerId, -Matrix)
 * Retrieve the weight matrix for a layer.
 */
get_weight_matrix(LayerId, Matrix) :-
    findall(Row, weight(LayerId, _, Row), Matrix).

/**
 * get_bias_vector(+LayerId, -Bias)
 * Retrieve the bias vector for a layer.
 */
get_bias_vector(LayerId, Bias) :-
    bias(LayerId, Bias).

/**
 * get_layer_activation(+LayerId, -Activation)
 * Get activation type for a layer.
 */
get_layer_activation(LayerId, Activation) :-
    layer_activation(LayerId, Activation).

% =============================================================================
% MLP Architecture
% =============================================================================

/**
 * mlp(+Input, -Output)
 *
 * Full MLP forward pass.
 * Architecture: 1 -> 6 -> 12 -> 24 -> 6 -> 1
 *
 * Example:
 *   ?- mlp([0.5], Energy).
 */
mlp(Input, Output) :-
    layer(0, Input, H1),
    layer(1, H1, H2),
    layer(2, H2, H3),
    layer(3, H3, H4),
    layer(4, H4, Output).

/**
 * layer(+LayerId, +Input, -Output)
 *
 * Compute single layer given its ID.
 */
layer(LayerId, Input, Output) :-
    get_weight_matrix(LayerId, Weights),
    get_bias_vector(LayerId, Bias),
    get_layer_activation(LayerId, Activation),
    dense_layer(Weights, Bias, Activation, Input, Output).

/**
 * mlp_with_trace(+Input, -Output, -Trace)
 *
 * MLP forward pass with trace of all intermediate activations.
 * Useful for debugging and interpretability.
 */
mlp_with_trace(Input, Output, Trace) :-
    layer_traced(0, Input, H1, T1),
    layer_traced(1, H1, H2, T2),
    layer_traced(2, H2, H3, T3),
    layer_traced(3, H3, H4, T4),
    layer_traced(4, H4, Output, T5),
    Trace = [T1, T2, T3, T4, T5].

layer_traced(LayerId, Input, Output, Trace) :-
    get_weight_matrix(LayerId, Weights),
    get_bias_vector(LayerId, Bias),
    get_layer_activation(LayerId, Activation),
    dense_layer_traced(LayerId, Weights, Bias, Activation, Input, Output, Trace).

% =============================================================================
% Weight Loading
% =============================================================================

/**
 * load_weights(+Filename)
 *
 * Load weights from a Prolog facts file.
 * File format:
 *   weight(LayerId, RowIndex, [w1, w2, ...]).
 *   bias(LayerId, [b1, b2, ...]).
 *   layer_activation(LayerId, swish).
 */
load_weights(Filename) :-
    clear_weights,
    consult(Filename).

/**
 * clear_weights/0
 * Remove all loaded weights.
 */
clear_weights :-
    retractall(weight(_, _, _)),
    retractall(bias(_, _)),
    retractall(layer_config(_, _, _)),
    retractall(layer_activation(_, _)).

% =============================================================================
% Default Weights (Random Initialization for Testing)
% =============================================================================

/**
 * init_random_weights/0
 * Initialize with small random weights for testing.
 */
init_random_weights :-
    clear_weights,
    init_layer(0, 1, 6, swish),
    init_layer(1, 6, 12, swish),
    init_layer(2, 12, 24, swish),
    init_layer(3, 24, 6, swish),
    init_layer(4, 6, 1, linear).

init_layer(LayerId, InFeatures, OutFeatures, Activation) :-
    assertz(layer_config(LayerId, InFeatures, OutFeatures)),
    assertz(layer_activation(LayerId, Activation)),
    Scale is sqrt(2.0 / InFeatures),
    init_weight_rows(LayerId, 0, OutFeatures, InFeatures, Scale),
    init_bias(LayerId, OutFeatures).

init_weight_rows(_, RowIdx, OutFeatures, _, _) :-
    RowIdx >= OutFeatures, !.
init_weight_rows(LayerId, RowIdx, OutFeatures, InFeatures, Scale) :-
    random_vector(InFeatures, Scale, Row),
    assertz(weight(LayerId, RowIdx, Row)),
    NextRow is RowIdx + 1,
    init_weight_rows(LayerId, NextRow, OutFeatures, InFeatures, Scale).

init_bias(LayerId, Size) :-
    length(Bias, Size),
    maplist(=(0.0), Bias),
    assertz(bias(LayerId, Bias)).

random_vector(0, _, []) :- !.
random_vector(N, Scale, [V|Rest]) :-
    N > 0,
    random(R),
    V is (R * 2 - 1) * Scale,
    N1 is N - 1,
    random_vector(N1, Scale, Rest).

% =============================================================================
% Weight Export (for interoperability)
% =============================================================================

/**
 * export_prolog_weights(+Model, +Filename)
 * Export current weights to a Prolog facts file.
 */
export_prolog_weights(_, Filename) :-
    open(Filename, write, Stream),
    write(Stream, '% SRO Decoder Ring MLP Weights\n'),
    write(Stream, '% Generated from Prolog\n\n'),
    export_all_weights(Stream),
    export_all_biases(Stream),
    export_all_activations(Stream),
    close(Stream).

export_all_weights(Stream) :-
    forall(weight(L, R, W),
           format(Stream, 'weight(~w, ~w, ~w).~n', [L, R, W])).

export_all_biases(Stream) :-
    forall(bias(L, B),
           format(Stream, 'bias(~w, ~w).~n', [L, B])).

export_all_activations(Stream) :-
    forall(layer_activation(L, A),
           format(Stream, 'layer_activation(~w, ~w).~n', [L, A])).

% =============================================================================
% Utility Predicates
% =============================================================================

/**
 * print_architecture/0
 * Display the current MLP architecture.
 */
print_architecture :-
    format('SRO Decoder Ring MLP Architecture~n'),
    format('================================~n'),
    forall(layer_config(L, In, Out),
           (layer_activation(L, Act),
            format('Layer ~w: ~w -> ~w (~w)~n', [L, In, Out, Act]))).

/**
 * verify_predicate(+Pred, +Expected)
 * Verify a predicate produces expected output.
 */
verify_predicate(Goal, Expected) :-
    call(Goal) ->
        (Goal = Expected ->
            format('PASS: ~w~n', [Goal])
        ;
            format('FAIL: Expected ~w, got ~w~n', [Expected, Goal]))
    ;
        format('FAIL: Goal ~w failed~n', [Goal]).

% =============================================================================
% Self-Test
% =============================================================================

/**
 * run_tests/0
 * Run basic self-tests.
 */
run_tests :-
    format('~n=== SRO Decoder MLP Tests ===~n~n'),

    % Test mat_vec
    format('Test mat_vec: '),
    (mat_vec([[1,2],[3,4]], [1,1], [3,7]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test vec_add
    format('Test vec_add: '),
    (vec_add([1,2], [0.5,0.5], [1.5,2.5]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test relu
    format('Test relu: '),
    (activation(relu, [-1,0,1], [0,0,1]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test swish at 0
    format('Test swish(0): '),
    (activation(swish, [0], [0.0]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test sigmoid at 0
    format('Test sigmoid(0): '),
    (activation(sigmoid, [0], [0.5]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test dense layer
    format('Test dense_layer: '),
    (dense_layer([[1,0],[0,1]], [0.5,0.5], linear, [1,2], [1.5,2.5]) ->
        format('PASS~n') ; format('FAIL~n')),

    % Test full MLP (with random weights)
    format('~nTest full MLP with random weights:~n'),
    init_random_weights,
    print_architecture,
    format('~nRunning mlp([0.5], Output):~n'),
    (mlp([0.5], Output) ->
        format('Output: ~w~n', [Output])
    ;
        format('MLP failed~n')),

    format('~n=== Tests Complete ===~n').

% Auto-initialize on load (comment out for production)
% :- init_random_weights.
