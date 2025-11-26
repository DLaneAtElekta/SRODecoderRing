namespace FsSRODecoderEngine

/// Predicate-Based MLP Implementation for SRO Decoder Ring
///
/// This module implements neural network operations using relational predicates:
/// - mat_vec: Matrix-vector multiplication predicate
/// - vec_add: Vector addition predicate
/// - activation: Activation function predicate
///
/// The predicate approach enables:
/// 1. Declarative specification of computations
/// 2. Compositional reasoning about network structure
/// 3. Natural integration with logic programming concepts
module PredicateMLP =
    open System

    // =========================================================================
    // Types
    // =========================================================================

    /// Activation function types
    type ActivationType =
        | Linear
        | ReLu
        | Swish
        | Sigmoid
        | Softmax

    /// Result of a predicate evaluation
    type PredicateResult<'T> = {
        Success: bool
        Value: 'T option
        Bindings: Map<string, obj>
    }

    /// Specification for a single layer
    type LayerSpec = {
        InFeatures: int
        OutFeatures: int
        Activation: ActivationType
        UseBias: bool
    }

    /// A layer with its weights
    type Layer = {
        Spec: LayerSpec
        Weights: float[,]  // [in_features, out_features]
        Bias: float[]      // [out_features]
    }

    /// Complete MLP model
    type PredicateMLPModel = {
        Layers: Layer list
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    let private createResult success value bindings =
        { Success = success; Value = value; Bindings = bindings }

    let private successResult value =
        createResult true (Some value) Map.empty

    let private failureResult () =
        createResult false None Map.empty

    // =========================================================================
    // Core Predicates
    // =========================================================================

    /// Matrix-vector multiplication predicate: mat_vec(M, V_in, V_out)
    ///
    /// Computes V_out = V_in @ M for each row in the batch
    ///
    /// Parameters:
    /// - M: Weight matrix [in_features, out_features]
    /// - V_in: Input matrix [batch, in_features]
    ///
    /// Returns: PredicateResult with output [batch, out_features]
    let mat_vec (M: float[,]) (V_in: float[,]) : PredicateResult<float[,]> =
        let batchSize = V_in.GetLength(0)
        let inFeatures = V_in.GetLength(1)
        let outFeatures = M.GetLength(1)

        // Validate dimensions
        if M.GetLength(0) <> inFeatures then
            failureResult ()
        else
            let V_out = Array2D.init batchSize outFeatures (fun b o ->
                seq { 0 .. inFeatures - 1 }
                |> Seq.fold (fun sum i -> sum + V_in.[b, i] * M.[i, o]) 0.0
            )
            createResult true (Some V_out) (Map.ofList [("V_out", box V_out)])

    /// Vector addition predicate: vec_add(A, B, C)
    ///
    /// Computes C = A + B (broadcasts B across batch dimension)
    ///
    /// Parameters:
    /// - A: Matrix [batch, features]
    /// - B: Bias vector [features]
    ///
    /// Returns: PredicateResult with sum [batch, features]
    let vec_add (A: float[,]) (B: float[]) : PredicateResult<float[,]> =
        let batchSize = A.GetLength(0)
        let features = A.GetLength(1)

        if B.Length <> features then
            failureResult ()
        else
            let C = Array2D.init batchSize features (fun b f ->
                A.[b, f] + B.[f]
            )
            createResult true (Some C) (Map.ofList [("C", box C)])

    /// Activation function predicate: activation(V_in, V_out, type)
    ///
    /// Applies element-wise activation function
    ///
    /// Parameters:
    /// - V_in: Input matrix [batch, features]
    /// - actType: Type of activation function
    ///
    /// Returns: PredicateResult with activated values
    let activation (V_in: float[,]) (actType: ActivationType) : PredicateResult<float[,]> =
        let sigmoid x = 1.0 / (1.0 + exp(-x))

        let activationFn =
            match actType with
            | Linear -> id
            | ReLu -> fun x -> max 0.0 x
            | Swish -> fun x -> x * sigmoid x
            | Sigmoid -> sigmoid
            | Softmax -> id  // Handled specially below

        let batchSize = V_in.GetLength(0)
        let features = V_in.GetLength(1)

        let V_out =
            match actType with
            | Softmax ->
                // Softmax: exp(x_i) / sum(exp(x_j)) for numerical stability
                Array2D.init batchSize features (fun b f ->
                    let maxVal = seq { 0 .. features - 1 } |> Seq.map (fun i -> V_in.[b, i]) |> Seq.max
                    let expSum = seq { 0 .. features - 1 } |> Seq.sumBy (fun i -> exp(V_in.[b, i] - maxVal))
                    exp(V_in.[b, f] - maxVal) / expSum
                )
            | _ ->
                Array2D.map activationFn V_in

        createResult true (Some V_out) (Map.ofList [("V_out", box V_out)])

    // =========================================================================
    // Compound Predicates
    // =========================================================================

    /// Dense layer predicate combining mat_vec, vec_add, and activation
    ///
    /// Semantics:
    ///     dense_layer(W, B, V_in, V_out) :-
    ///         mat_vec(W, V_in, Z),
    ///         vec_add(Z, B, A),
    ///         activation(A, V_out).
    let dense_layer (layer: Layer) (V_in: float[,]) : PredicateResult<float[,]> =
        // mat_vec(W, V_in, Z)
        let matResult = mat_vec layer.Weights V_in
        match matResult.Value with
        | None -> failureResult ()
        | Some Z ->
            // vec_add(Z, B, A)
            let addResult = vec_add Z layer.Bias
            match addResult.Value with
            | None -> failureResult ()
            | Some A ->
                // activation(A, V_out)
                let actResult = activation A layer.Spec.Activation
                match actResult.Value with
                | None -> failureResult ()
                | Some V_out ->
                    createResult true (Some V_out) (Map.ofList [
                        ("Z", box Z)
                        ("A", box A)
                        ("V_out", box V_out)
                    ])

    // =========================================================================
    // MLP Forward Pass
    // =========================================================================

    /// Forward pass through the entire MLP using predicate composition
    ///
    /// The computation is equivalent to:
    ///     mlp(X, Y) :-
    ///         layer(0, X, H0),
    ///         layer(1, H0, H1),
    ///         ...
    ///         layer(n, Hn-1, Y).
    let forward (model: PredicateMLPModel) (input: float[,]) : PredicateResult<float[,]> =
        let rec forwardLayers layers current =
            match layers with
            | [] -> successResult current
            | layer :: rest ->
                let result = dense_layer layer current
                match result.Value with
                | None -> failureResult ()
                | Some output -> forwardLayers rest output

        forwardLayers model.Layers input

    /// Forward pass with trace of intermediate activations
    let forwardWithTrace (model: PredicateMLPModel) (input: float[,])
        : PredicateResult<float[,]> * (int * float[,]) list =

        let rec forwardLayers layers current layerIdx trace =
            match layers with
            | [] -> (successResult current, List.rev trace)
            | layer :: rest ->
                let result = dense_layer layer current
                match result.Value with
                | None -> (failureResult (), List.rev trace)
                | Some output ->
                    let newTrace = (layerIdx, output) :: trace
                    forwardLayers rest output (layerIdx + 1) newTrace

        forwardLayers model.Layers input 0 []

    // =========================================================================
    // Model Construction
    // =========================================================================

    /// Create a layer with random initialization
    let createLayer (spec: LayerSpec) : Layer =
        let rng = Random()
        let scale = sqrt(2.0 / float spec.InFeatures)  // He initialization

        let weights = Array2D.init spec.InFeatures spec.OutFeatures (fun _ _ ->
            (rng.NextDouble() * 2.0 - 1.0) * scale
        )
        let bias = Array.zeroCreate spec.OutFeatures

        { Spec = spec; Weights = weights; Bias = bias }

    /// Create the SRO Decoder Ring MLP architecture
    /// Architecture: 1 -> 6 -> 12 -> 24 -> 6 -> 1
    let createSRODecoderMLP () : PredicateMLPModel =
        let specs = [
            { InFeatures = 1; OutFeatures = 6; Activation = Swish; UseBias = true }
            { InFeatures = 6; OutFeatures = 12; Activation = Swish; UseBias = true }
            { InFeatures = 12; OutFeatures = 24; Activation = Swish; UseBias = true }
            { InFeatures = 24; OutFeatures = 6; Activation = Swish; UseBias = true }
            { InFeatures = 6; OutFeatures = 1; Activation = Linear; UseBias = true }
        ]
        { Layers = specs |> List.map createLayer }

    /// Create a classifier MLP for rotation order prediction
    /// Architecture: 12 -> 64 -> 32 -> 32 -> 6 (softmax)
    let createClassifierMLP () : PredicateMLPModel =
        let specs = [
            { InFeatures = 12; OutFeatures = 64; Activation = ReLu; UseBias = true }
            { InFeatures = 64; OutFeatures = 32; Activation = ReLu; UseBias = true }
            { InFeatures = 32; OutFeatures = 32; Activation = ReLu; UseBias = true }
            { InFeatures = 32; OutFeatures = 6; Activation = Softmax; UseBias = true }
        ]
        { Layers = specs |> List.map createLayer }

    // =========================================================================
    // Prolog-Style Query Interface
    // =========================================================================

    /// Query result type
    type QueryResult = {
        Output: float[,]
        Trace: (int * float[,]) list
        Success: bool
    }

    /// Execute a forward query on the model
    let query (model: PredicateMLPModel) (input: float[,]) : QueryResult =
        let result, trace = forwardWithTrace model input
        {
            Output = result.Value |> Option.defaultValue (Array2D.zeroCreate 0 0)
            Trace = trace
            Success = result.Success
        }

    // =========================================================================
    // Prolog Representation Export
    // =========================================================================

    /// Export the MLP structure as Prolog-style rules
    let toPrologRules (model: PredicateMLPModel) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine("% Predicate-based MLP for SRO Decoder Ring") |> ignore
        sb.AppendLine("% Generated from F# PredicateMLP") |> ignore
        sb.AppendLine() |> ignore

        model.Layers |> List.iteri (fun i layer ->
            let actName =
                match layer.Spec.Activation with
                | Linear -> "linear"
                | ReLu -> "relu"
                | Swish -> "swish"
                | Sigmoid -> "sigmoid"
                | Softmax -> "softmax"

            sb.AppendLine(sprintf "%% Layer %d: %d -> %d, %s"
                i layer.Spec.InFeatures layer.Spec.OutFeatures actName) |> ignore
            sb.AppendLine(sprintf "layer(%d, V_in, V_out) :-" i) |> ignore
            sb.AppendLine(sprintf "    mat_vec(w%d, V_in, Z%d)," i i) |> ignore
            sb.AppendLine(sprintf "    vec_add(Z%d, b%d, A%d)," i i i) |> ignore
            sb.AppendLine(sprintf "    %s(A%d, V_out)." actName i) |> ignore
            sb.AppendLine() |> ignore
        )

        // Full MLP predicate
        let nLayers = model.Layers.Length
        let layerCalls =
            [0 .. nLayers - 1]
            |> List.map (fun i -> sprintf "layer(%d, H%d, H%d)" i i (i + 1))
            |> String.concat ", "

        sb.AppendLine(sprintf "mlp(X, Y) :- H0 = X, %s, Y = H%d." layerCalls nLayers) |> ignore

        sb.ToString()

    // =========================================================================
    // Weight Loading/Saving
    // =========================================================================

    /// Load weights into a layer from arrays
    let loadLayerWeights (layer: Layer) (weights: float[,]) (bias: float[]) : Layer =
        { layer with Weights = weights; Bias = bias }

    /// Load model from weight dictionary
    let loadWeights (model: PredicateMLPModel) (weightDict: Map<string, float[,] * float[]>)
        : PredicateMLPModel =
        let newLayers =
            model.Layers
            |> List.mapi (fun i layer ->
                let key = sprintf "layer_%d" i
                match weightDict.TryFind key with
                | Some (w, b) -> loadLayerWeights layer w b
                | None -> layer
            )
        { Layers = newLayers }
