
use ark_ff::PrimeField;
use ark_poly_commit::{PolynomialCommitment, ipa_pc::Commitment};

mod layers;
mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?

/// A layer of the model including its transition function to the next layer.
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub trait Layer<F: PrimeField, PCS: PolynomialCommitment> {    
    /// Types for the prover and verifier keys
    type ProverKey: Clone;
    type VerifierKey: Clone;

    /// A commitment to the layer's transition function. For instance, it is
    /// empty for a ReLU transition; and a commitment to the matrix and bias
    /// for a MatMul transition.
    type Commitment: Clone = ();

    /// A proof of execution of the layer's transition function to a particular
    /// set of node values
    type Proof;

    /// Returns the number of nodes in the layer
    fn num_nodes(&self) -> usize;

    /// Returns the base-two logarithm of the number of nodes in the layer, i.e.
    /// the number of variables of the MLE of the node values
    fn log_num_nodes(&self) -> usize;

    /// Evaluate the layer on the given input natively.
    fn evaluate(&self, input: Vec<F>) -> Vec<F>;

    /// Evaluate the layer on the given input natively.
    // TODO: is it okay to trim all keys from the same original PCS key?
    // (e.g. to trim the same key to for the matrix and for the bias in the
    // case of MatMul)
    fn setup(&self, params: PCS::UniversalParams) -> (ProverKey, VerifierKey);

    /// Evaluate the layer on the given input natively.
    fn commit(&self) -> Commitment;

    /// Prove that the layer was executed correctly on the given input.
    fn prove(com: Commitment, input: Vec<F>) -> Proof;

    /// Check that the layer transition was executed correctly.
    fn check(com: Commitment, proof: Proof) -> bool;
}

impl<F: PrimeField, PCS: PolynomialCommitment> Layer<F, PCS> {
    fn log_num_nodes(&self) -> usize {
        self.num_nodes().log2()
    }

    fn commit(&self) -> Commitment {
    }
}

struct Model {
    layers: Vec<Layer>,
}

impl Model {
    fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers,
        }
    }
}


