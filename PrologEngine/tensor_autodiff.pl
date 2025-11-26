/**
 * Tensor Computation Graph with Automatic Differentiation
 *
 * This module implements a computational graph framework in SWI-Prolog
 * that builds execution graphs suitable for automatic differentiation.
 *
 * Core predicates:
 *   - dense(Weights, Bias, In, Out)    : Dense layer operation
 *   - relu(In, Out)                     : ReLU activation
 *   - swish(In, Out)                    : Swish activation
 *   - mse(Expected, Actual, Loss)       : Mean squared error loss
 *   - forward(Graph, Input, Output)     : Execute forward pass
 *   - backward(Graph, Loss, Gradients)  : Compute gradients (autodiff)
 *
 * Usage:
 *   ?- build_graph(mlp, Graph), forward(Graph, [0.5], Out), backward(Graph, Out, Grads).
 */

:- module(tensor_autodiff, [
    % Tensor operations (build graph)
    dense/4,
    bias_add/3,
    relu/2,
    swish/2,
    sigmoid/2,
    softmax/2,
    mse/3,
    mae/3,

    % Graph operations
    build_mlp/3,
    forward/3,
    backward/2,

    % Tensor utilities
    tensor/2,
    tensor_shape/2,
    zeros/2,
    ones/2,
    random_tensor/3,
    from_list/2,

    % Graph utilities
    reset_graph/0,
    print_graph/0,
    export_graph/2,

    % Testing
    run_autodiff_tests/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

% Discontiguous declarations
:- discontiguous grad/4.

% =============================================================================
% Tensor Representation
% =============================================================================

/**
 * tensor(Id, Data)
 *
 * A tensor is represented as tensor(Id, Data) where:
 * - Id: unique identifier for the tensor node in the graph
 * - Data: the actual numerical data (list or nested list)
 *
 * Tensors can also be symbolic placeholders:
 * - input(Name): Input placeholder
 * - param(Name): Trainable parameter
 * - const(Value): Constant value
 */

:- dynamic tensor_store/2.      % tensor_store(Id, Data)
:- dynamic tensor_shape/2.      % tensor_shape(Id, Shape)
:- dynamic tensor_grad/2.       % tensor_grad(Id, Gradient)
:- dynamic graph_node/3.        % graph_node(OutputId, Op, Inputs)
:- dynamic param/2.             % param(Name, TensorId)

% Generate unique tensor ID
:- dynamic tensor_counter/1.
tensor_counter(0).

new_tensor_id(Id) :-
    retract(tensor_counter(N)),
    Id is N + 1,
    assertz(tensor_counter(Id)).

reset_graph :-
    retractall(tensor_store(_, _)),
    retractall(tensor_shape(_, _)),
    retractall(tensor_grad(_, _)),
    retractall(graph_node(_, _, _)),
    retractall(param(_, _)),
    retractall(tensor_counter(_)),
    assertz(tensor_counter(0)).

% =============================================================================
% Tensor Constructors
% =============================================================================

/**
 * tensor(+Shape, -Tensor)
 * Create a new tensor with given shape (uninitialized).
 */
tensor(Shape, tensor(Id, Shape)) :-
    new_tensor_id(Id),
    assertz(tensor_shape(Id, Shape)).

/**
 * zeros(+Shape, -Tensor)
 * Create a tensor filled with zeros.
 */
zeros(Shape, tensor(Id, Data)) :-
    new_tensor_id(Id),
    create_zeros(Shape, Data),
    assertz(tensor_store(Id, Data)),
    assertz(tensor_shape(Id, Shape)).

create_zeros([], 0.0) :- !.
create_zeros([N], List) :- !,
    length(List, N),
    maplist(=(0.0), List).
create_zeros([N|Rest], List) :-
    length(List, N),
    maplist(create_zeros(Rest), List).

/**
 * ones(+Shape, -Tensor)
 * Create a tensor filled with ones.
 */
ones(Shape, tensor(Id, Data)) :-
    new_tensor_id(Id),
    create_ones(Shape, Data),
    assertz(tensor_store(Id, Data)),
    assertz(tensor_shape(Id, Shape)).

create_ones([], 1.0) :- !.
create_ones([N], List) :- !,
    length(List, N),
    maplist(=(1.0), List).
create_ones([N|Rest], List) :-
    length(List, N),
    maplist(create_ones(Rest), List).

/**
 * random_tensor(+Shape, +Scale, -Tensor)
 * Create a tensor with random values scaled by Scale.
 */
random_tensor(Shape, Scale, tensor(Id, Data)) :-
    new_tensor_id(Id),
    create_random(Shape, Scale, Data),
    assertz(tensor_store(Id, Data)),
    assertz(tensor_shape(Id, Shape)).

create_random([], Scale, V) :- !,
    random(R),
    V is (R * 2 - 1) * Scale.
create_random([N], Scale, List) :- !,
    length(List, N),
    maplist({Scale}/[V]>>(random(R), V is (R * 2 - 1) * Scale), List).
create_random([N|Rest], Scale, List) :-
    length(List, N),
    maplist(create_random(Rest, Scale), List).

/**
 * from_list(+Data, -Tensor)
 * Create a tensor from a nested list.
 */
from_list(Data, tensor(Id, Data)) :-
    new_tensor_id(Id),
    infer_shape(Data, Shape),
    assertz(tensor_store(Id, Data)),
    assertz(tensor_shape(Id, Shape)).

infer_shape(X, []) :- number(X), !.
infer_shape([], [0]) :- !.
infer_shape([H|T], [N|Rest]) :-
    length([H|T], N),
    infer_shape(H, Rest).

% =============================================================================
% Graph Node Representation
% =============================================================================

/**
 * Graph nodes represent operations in the computation graph.
 *
 * node(Id, Op, Inputs, Output, GradFn)
 * - Id: unique node identifier
 * - Op: operation name (dense, relu, mse, etc.)
 * - Inputs: list of input tensor IDs
 * - Output: output tensor ID
 * - GradFn: gradient function for backprop
 */

:- dynamic node/5.
:- dynamic node_counter/1.
node_counter(0).

new_node_id(Id) :-
    retract(node_counter(N)),
    Id is N + 1,
    assertz(node_counter(Id)).

add_node(Op, Inputs, Output, GradFn) :-
    new_node_id(NodeId),
    assertz(node(NodeId, Op, Inputs, Output, GradFn)),
    assertz(graph_node(Output, Op, Inputs)).

% =============================================================================
% Core Operations (Build Computation Graph)
% =============================================================================

/**
 * dense(+Weights, +Bias, +Input, -Output)
 *
 * Dense (fully connected) layer: Output = Input @ Weights + Bias
 *
 * Builds a graph node for the operation.
 * Gradient: dL/dW = Input^T @ dL/dOut, dL/dIn = dL/dOut @ W^T
 */
dense(Weights, Bias, Input, Output) :-
    % Get tensor IDs
    tensor_id(Weights, WId),
    tensor_id(Bias, BId),
    tensor_id(Input, InId),

    % Compute forward pass
    get_tensor_data(WId, WData),
    get_tensor_data(BId, BData),
    get_tensor_data(InId, InData),

    mat_vec_compute(WData, InData, MvResult),
    vec_add_compute(MvResult, BData, OutData),

    % Create output tensor
    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    % Record in graph with gradient function
    add_node(dense, [WId, BId, InId], OutId, grad_dense(WId, BId, InId)).

/**
 * bias_add(+Bias, +Input, -Output)
 *
 * Add bias to input: Output = Input + Bias
 */
bias_add(Bias, Input, Output) :-
    tensor_id(Bias, BId),
    tensor_id(Input, InId),

    get_tensor_data(BId, BData),
    get_tensor_data(InId, InData),

    vec_add_compute(InData, BData, OutData),

    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    add_node(bias_add, [BId, InId], OutId, grad_bias_add(BId, InId)).

/**
 * relu(+Input, -Output)
 *
 * ReLU activation: Output = max(0, Input)
 * Gradient: dL/dIn = dL/dOut * (Input > 0 ? 1 : 0)
 */
relu(Input, Output) :-
    tensor_id(Input, InId),
    get_tensor_data(InId, InData),

    map_relu(InData, OutData),

    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    add_node(relu, [InId], OutId, grad_relu(InId)).

map_relu(X, Y) :- number(X), !, (X > 0 -> Y = X ; Y = 0.0).
map_relu(List, Result) :-
    is_list(List),
    maplist(map_relu, List, Result).

/**
 * swish(+Input, -Output)
 *
 * Swish activation: Output = Input * sigmoid(Input)
 * Gradient: dL/dIn = dL/dOut * (swish(x) + sigmoid(x) * (1 - swish(x)))
 */
swish(Input, Output) :-
    tensor_id(Input, InId),
    get_tensor_data(InId, InData),

    map_swish(InData, OutData),

    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    add_node(swish, [InId], OutId, grad_swish(InId)).

map_swish(X, Y) :-
    number(X), !,
    Sig is 1 / (1 + exp(-X)),
    Y is X * Sig.
map_swish(List, Result) :-
    is_list(List),
    maplist(map_swish, List, Result).

/**
 * sigmoid(+Input, -Output)
 *
 * Sigmoid activation: Output = 1 / (1 + exp(-Input))
 * Gradient: dL/dIn = dL/dOut * sigmoid(x) * (1 - sigmoid(x))
 */
sigmoid(Input, Output) :-
    tensor_id(Input, InId),
    get_tensor_data(InId, InData),

    map_sigmoid(InData, OutData),

    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    add_node(sigmoid, [InId], OutId, grad_sigmoid(InId)).

map_sigmoid(X, Y) :- number(X), !, Y is 1 / (1 + exp(-X)).
map_sigmoid(List, Result) :-
    is_list(List),
    maplist(map_sigmoid, List, Result).

/**
 * softmax(+Input, -Output)
 *
 * Softmax activation (for classification)
 */
softmax(Input, Output) :-
    tensor_id(Input, InId),
    get_tensor_data(InId, InData),

    compute_softmax(InData, OutData),

    new_tensor_id(OutId),
    assertz(tensor_store(OutId, OutData)),
    Output = tensor(OutId, OutData),

    add_node(softmax, [InId], OutId, grad_softmax(InId)).

compute_softmax(List, Result) :-
    max_list(List, Max),
    maplist({Max}/[X, E]>>(E is exp(X - Max)), List, Exps),
    sum_list(Exps, Sum),
    maplist({Sum}/[E, S]>>(S is E / Sum), Exps, Result).

/**
 * mse(+Expected, +Actual, -Loss)
 *
 * Mean Squared Error loss: Loss = mean((Expected - Actual)^2)
 * Gradient: dL/dActual = 2 * (Actual - Expected) / N
 */
mse(Expected, Actual, Loss) :-
    tensor_id(Expected, ExpId),
    tensor_id(Actual, ActId),

    get_tensor_data(ExpId, ExpData),
    get_tensor_data(ActId, ActData),

    compute_mse(ExpData, ActData, LossVal),

    new_tensor_id(LossId),
    assertz(tensor_store(LossId, LossVal)),
    Loss = tensor(LossId, LossVal),

    add_node(mse, [ExpId, ActId], LossId, grad_mse(ExpId, ActId)).

compute_mse(Exp, Act, Loss) :-
    flatten(Exp, ExpFlat),
    flatten(Act, ActFlat),
    maplist([E, A, D]>>(D is (E - A)^2), ExpFlat, ActFlat, Diffs),
    sum_list(Diffs, Sum),
    length(Diffs, N),
    Loss is Sum / N.

/**
 * mae(+Expected, +Actual, -Loss)
 *
 * Mean Absolute Error loss: Loss = mean(|Expected - Actual|)
 */
mae(Expected, Actual, Loss) :-
    tensor_id(Expected, ExpId),
    tensor_id(Actual, ActId),

    get_tensor_data(ExpId, ExpData),
    get_tensor_data(ActId, ActData),

    compute_mae(ExpData, ActData, LossVal),

    new_tensor_id(LossId),
    assertz(tensor_store(LossId, LossVal)),
    Loss = tensor(LossId, LossVal),

    add_node(mae, [ExpId, ActId], LossId, grad_mae(ExpId, ActId)).

compute_mae(Exp, Act, Loss) :-
    flatten(Exp, ExpFlat),
    flatten(Act, ActFlat),
    maplist([E, A, D]>>(D is abs(E - A)), ExpFlat, ActFlat, Diffs),
    sum_list(Diffs, Sum),
    length(Diffs, N),
    Loss is Sum / N.

% =============================================================================
% Helper Functions
% =============================================================================

tensor_id(tensor(Id, _), Id) :- !.
tensor_id(Id, Id) :- integer(Id).

get_tensor_data(Id, Data) :-
    tensor_store(Id, Data), !.
get_tensor_data(tensor(_, Data), Data).

mat_vec_compute(Matrix, Vector, Result) :-
    maplist(dot_product_compute(Vector), Matrix, Result).

dot_product_compute(V1, V2, Product) :-
    maplist([X, Y, Z]>>(Z is X * Y), V1, V2, Products),
    sum_list(Products, Product).

vec_add_compute(A, B, C) :-
    maplist([X, Y, Z]>>(Z is X + Y), A, B, C).

% =============================================================================
% Gradient Functions (Backward Pass)
% =============================================================================

/**
 * grad(+Op, +Inputs, +GradOutput, -GradInputs)
 *
 * Compute gradients for each operation type.
 */

% ReLU gradient: pass through where input > 0
grad(relu, [InId], GradOut, [GradIn]) :-
    get_tensor_data(InId, InData),
    compute_relu_grad(InData, GradOut, GradIn).

compute_relu_grad(In, GradOut, GradIn) :-
    number(In), !,
    (In > 0 -> GradIn = GradOut ; GradIn = 0).
compute_relu_grad(InList, GradOutList, GradInList) :-
    maplist(compute_relu_grad, InList, GradOutList, GradInList).

% Swish gradient: swish'(x) = swish(x) + sigmoid(x)(1 - swish(x))
grad(swish, [InId], GradOut, [GradIn]) :-
    get_tensor_data(InId, InData),
    compute_swish_grad(InData, GradOut, GradIn).

compute_swish_grad(In, GradOut, GradIn) :-
    number(In), !,
    Sig is 1 / (1 + exp(-In)),
    Swish is In * Sig,
    Deriv is Swish + Sig * (1 - Swish),
    GradIn is GradOut * Deriv.
compute_swish_grad(InList, GradOutList, GradInList) :-
    maplist(compute_swish_grad, InList, GradOutList, GradInList).

% Sigmoid gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
grad(sigmoid, [InId], GradOut, [GradIn]) :-
    get_tensor_data(InId, InData),
    compute_sigmoid_grad(InData, GradOut, GradIn).

compute_sigmoid_grad(In, GradOut, GradIn) :-
    number(In), !,
    Sig is 1 / (1 + exp(-In)),
    Deriv is Sig * (1 - Sig),
    GradIn is GradOut * Deriv.
compute_sigmoid_grad(InList, GradOutList, GradInList) :-
    maplist(compute_sigmoid_grad, InList, GradOutList, GradInList).

% MSE gradient: 2 * (actual - expected) / N
grad(mse, [ExpId, ActId], GradOut, [GradExp, GradAct]) :-
    get_tensor_data(ExpId, ExpData),
    get_tensor_data(ActId, ActData),
    compute_mse_grad(ExpData, ActData, GradOut, GradExp, GradAct).

compute_mse_grad(Exp, Act, GradOut, GradExp, GradAct) :-
    flatten(Exp, ExpFlat),
    flatten(Act, ActFlat),
    length(ExpFlat, N),
    Scale is 2 * GradOut / N,
    maplist({Scale}/[E, A, GE]>>(GE is Scale * (E - A)), ExpFlat, ActFlat, GradExp),
    maplist({Scale}/[E, A, GA]>>(GA is Scale * (A - E)), ExpFlat, ActFlat, GradAct).

% Dense layer gradient
grad(dense, [WId, _BId, InId], GradOut, [GradW, GradB, GradIn]) :-
    get_tensor_data(WId, WData),
    get_tensor_data(InId, InData),
    compute_dense_grad(WData, InData, GradOut, GradW, GradB, GradIn).

compute_dense_grad(W, In, GradOut, GradW, GradB, GradIn) :-
    % GradB = GradOut
    GradB = GradOut,
    % GradW[i][j] = In[j] * GradOut[i]
    maplist({In}/[GO, Row]>>maplist({GO}/[I, G]>>(G is I * GO), In, Row), GradOut, GradW),
    % GradIn[j] = sum_i(W[i][j] * GradOut[i])
    transpose_matrix(W, WT),
    mat_vec_compute(WT, GradOut, GradIn).

transpose_matrix([], []) :- !.
transpose_matrix([[]|_], []) :- !.
transpose_matrix(Matrix, [Row|Rows]) :-
    maplist(nth0(0), Matrix, Row),
    maplist(select_tail, Matrix, RestMatrix),
    transpose_matrix(RestMatrix, Rows).

select_tail([_|T], T).

% =============================================================================
% Backward Pass (Automatic Differentiation)
% =============================================================================

/**
 * backward(+LossTensor, -Gradients)
 *
 * Perform backward pass through the computation graph.
 * Returns gradients for all parameters.
 */
backward(LossTensor, Gradients) :-
    tensor_id(LossTensor, LossId),
    % Start with gradient of 1 for the loss
    backward_from(LossId, 1.0, Gradients).

backward_from(NodeId, GradOut, Gradients) :-
    (graph_node(NodeId, Op, Inputs) ->
        % Compute gradients for this node
        grad(Op, Inputs, GradOut, GradInputs),
        % Recursively backprop to inputs
        maplist(backward_from_pair, Inputs, GradInputs, GradLists),
        append(GradLists, Gradients)
    ;
        % Leaf node (parameter or input)
        Gradients = [(NodeId, GradOut)]
    ).

backward_from_pair(NodeId, GradOut, Gradients) :-
    backward_from(NodeId, GradOut, Gradients).

% =============================================================================
% Graph Building Utilities
% =============================================================================

/**
 * build_mlp(+LayerSizes, +Activations, -Graph)
 *
 * Build an MLP computation graph.
 * Example: build_mlp([1, 6, 12, 24, 6, 1], [swish, swish, swish, swish, linear], Graph)
 */
build_mlp(Sizes, Activations, graph(Params, InputId, OutputId)) :-
    reset_graph,
    % Create input placeholder
    new_tensor_id(InputId),
    % Build layers
    build_layers(Sizes, Activations, InputId, OutputId, Params).

build_layers([_], [], LastId, LastId, []) :- !.
build_layers([In, Out|Rest], [Act|Acts], PrevId, FinalId, [param(W, B)|Params]) :-
    % Create weight and bias parameters
    Scale is sqrt(2.0 / In),
    random_tensor([Out, In], Scale, W),
    zeros([Out], B),
    % Forward through this layer
    tensor_id(W, WId),
    tensor_id(B, BId),
    new_tensor_id(OutId),
    add_node(dense, [WId, BId, PrevId], OutId, grad_dense(WId, BId, PrevId)),
    % Apply activation
    apply_activation(Act, OutId, ActOutId),
    % Continue
    build_layers([Out|Rest], Acts, ActOutId, FinalId, Params).

apply_activation(linear, InId, InId) :- !.
apply_activation(relu, InId, OutId) :-
    new_tensor_id(OutId),
    add_node(relu, [InId], OutId, grad_relu(InId)).
apply_activation(swish, InId, OutId) :-
    new_tensor_id(OutId),
    add_node(swish, [InId], OutId, grad_swish(InId)).
apply_activation(sigmoid, InId, OutId) :-
    new_tensor_id(OutId),
    add_node(sigmoid, [InId], OutId, grad_sigmoid(InId)).
apply_activation(softmax, InId, OutId) :-
    new_tensor_id(OutId),
    add_node(softmax, [InId], OutId, grad_softmax(InId)).

% =============================================================================
% Forward Pass
% =============================================================================

/**
 * forward(+Graph, +Input, -Output)
 *
 * Execute forward pass through the computation graph.
 */
forward(graph(_, InputId, OutputId), InputData, OutputData) :-
    % Store input data
    assertz(tensor_store(InputId, InputData)),
    % Execute graph nodes in topological order
    execute_graph(InputId, OutputId),
    % Get output
    get_tensor_data(OutputId, OutputData).

execute_graph(InputId, OutputId) :-
    findall(node(N, Op, Ins, Out), node(N, Op, Ins, Out, _), Nodes),
    execute_nodes(Nodes, InputId, OutputId).

execute_nodes([], _, _) :- !.
execute_nodes([node(_, Op, Inputs, OutId)|Rest], InputId, FinalId) :-
    % Check if all inputs are available
    maplist(input_available, Inputs),
    % Execute operation
    execute_op(Op, Inputs, OutId),
    % Continue
    execute_nodes(Rest, InputId, FinalId).

input_available(Id) :- tensor_store(Id, _).

execute_op(dense, [WId, BId, InId], OutId) :-
    get_tensor_data(WId, W),
    get_tensor_data(BId, B),
    get_tensor_data(InId, In),
    mat_vec_compute(W, In, Mv),
    vec_add_compute(Mv, B, Out),
    assertz(tensor_store(OutId, Out)).

execute_op(relu, [InId], OutId) :-
    get_tensor_data(InId, In),
    map_relu(In, Out),
    assertz(tensor_store(OutId, Out)).

execute_op(swish, [InId], OutId) :-
    get_tensor_data(InId, In),
    map_swish(In, Out),
    assertz(tensor_store(OutId, Out)).

execute_op(sigmoid, [InId], OutId) :-
    get_tensor_data(InId, In),
    map_sigmoid(In, Out),
    assertz(tensor_store(OutId, Out)).

% =============================================================================
% Execution Graph Export
% =============================================================================

/**
 * export_graph(+Format, -GraphRepr)
 *
 * Export the computation graph in various formats.
 */
export_graph(dot, DotString) :-
    findall(node(N, Op, Ins, Out), node(N, Op, Ins, Out, _), Nodes),
    export_as_dot(Nodes, DotString).

export_as_dot(Nodes, DotString) :-
    format(string(Header), 'digraph ComputationGraph {~n  rankdir=TB;~n', []),
    maplist(node_to_dot, Nodes, NodeStrings),
    maplist(edges_to_dot, Nodes, EdgeStrings),
    atomic_list_concat([Header|NodeStrings], NodesPart),
    atomic_list_concat(EdgeStrings, EdgesPart),
    format(string(DotString), '~w~w}~n', [NodesPart, EdgesPart]).

node_to_dot(node(N, Op, _, Out), String) :-
    format(string(String), '  ~w [label="~w\\n(~w)"];~n', [Out, Op, N]).

edges_to_dot(node(_, _, Inputs, Out), String) :-
    maplist({Out}/[In, S]>>format(string(S), '  ~w -> ~w;~n', [In, Out]), Inputs, Strings),
    atomic_list_concat(Strings, String).

/**
 * print_graph/0
 * Print the current computation graph.
 */
print_graph :-
    format('Computation Graph:~n'),
    format('=================~n'),
    forall(node(N, Op, Inputs, Output, _),
           format('Node ~w: ~w(~w) -> ~w~n', [N, Op, Inputs, Output])).

% =============================================================================
% Self-Test
% =============================================================================

run_autodiff_tests :-
    format('~n=== Tensor Autodiff Tests ===~n~n'),

    % Test tensor creation
    format('Test tensor creation: '),
    reset_graph,
    from_list([1.0, 2.0, 3.0], T1),
    (T1 = tensor(_, [1.0, 2.0, 3.0]) -> format('PASS~n') ; format('FAIL~n')),

    % Test relu
    format('Test relu operation: '),
    reset_graph,
    from_list([-1.0, 0.0, 1.0, 2.0], Input),
    relu(Input, Output),
    tensor_id(Output, OutId),
    get_tensor_data(OutId, OutData),
    (OutData = [0.0, 0.0, 1.0, 2.0] -> format('PASS~n') ; format('FAIL~n')),

    % Test graph recording
    format('Test graph recording: '),
    (graph_node(OutId, relu, _) -> format('PASS~n') ; format('FAIL~n')),

    % Test MSE loss
    format('Test MSE loss: '),
    reset_graph,
    from_list([1.0, 2.0], Exp),
    from_list([1.5, 2.5], Act),
    mse(Exp, Act, Loss),
    tensor_id(Loss, LossId),
    get_tensor_data(LossId, LossVal),
    (abs(LossVal - 0.25) < 0.001 -> format('PASS~n') ; format('FAIL~n')),

    format('~n=== Tests Complete ===~n').

% Entry point for testing
:- initialization((
    format('Tensor Autodiff module loaded.~n'),
    format('Run run_autodiff_tests for self-tests.~n')
)).
