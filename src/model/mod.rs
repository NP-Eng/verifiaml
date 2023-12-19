
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::Polynomial;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

mod layers;
mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A layer of the model including its transition function to the next layer.
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub trait Layer<F, P, S, PCS>
where
    F: PrimeField,
    P: Polynomial<F>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, P, S>,
{
// pub trait Layer<F: PrimeField, PCS: PolynomialCommitment<F, P: Polynomial<F>, S: CryptographicSponge>> {    
    /// Types for the prover and verifier keys
    type ProverKey: Clone;
    type VerifierKey: Clone;

    /// A commitment associated to the layer's transition function. For
    /// instance, it is empty for a ReLU transition; and a commitment to the
    /// matrix and bias for a MatMul transition.
    type Commitment;

    /// A proof of execution of the layer's transition function to a particular
    /// set of node values
    type Proof;

    /// Returns the number of nodes in the layer
    fn num_nodes(&self) -> usize;

    /// Returns the base-two logarithm of the number of nodes in the layer, i.e.
    /// the number of variables of the MLE of the node values
    fn log_num_nodes(&self) -> usize {
        log2(self.num_nodes()) as usize
    }

    /// Evaluate the layer on the given input natively.
    fn evaluate(&self, input: Vec<F>) -> Vec<F>;

    /// Evaluate the layer on the given input natively.
    // TODO: is it okay to trim all keys from the same original PCS key?
    // (e.g. to trim the same key to for the matrix and for the bias in the
    // case of MatMul)
    fn setup(&self, params: PCS::UniversalParams) -> (Self::ProverKey, Self::VerifierKey);

    /// Evaluate the layer on the given input natively.
    fn commit(&self) -> Self::Commitment;

    /// Prove that the layer was executed correctly on the given input.
    fn prove(com: Self::Commitment, input: Vec<F>) -> Self::Proof;

    /// Check that the layer transition was executed correctly.
    fn check(com: Self::Commitment, proof: Self::Proof) -> bool;
}

// TODO: for now, we require all layers to use the same PCS; this might change
// in the future
struct Model<F, P, S, PCS>
where
    F: PrimeField,
    P: Polynomial<F>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, P, S>,
{
    layers: Vec<Box<dyn Layer<F, P, S, PCS>>>,
}

impl<F, P, S, PCS> Model<F, P, S, PCS>
where
    F: PrimeField,
    P: Polynomial<F>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, P, S>,
{
    fn new(layers: Vec<dyn Layer>) -> Self {
        Self {
            layers,
        }
    }
}


