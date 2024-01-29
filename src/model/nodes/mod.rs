use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::{
    model::{
        nodes::{fc::FCNode, relu::ReLUNode},
        CryptographicSponge, Poly,
    },
    quantization::QSmallType,
};

pub(crate) mod fc;
pub(crate) mod relu;

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A layer of the model including its transition function to the next layer.
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub trait NodeOps<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// A commitment associated to the layer's transition function. For
    /// instance, it is empty for a ReLU transition; and a commitment to the
    /// matrix and bias for a MatMul transition.
    type Commitment;

    // /// A proof of execution of the layer's transition function to a particular
    // /// set of node values
    type Proof;

    /// Returns the number of nodes in the layer
    fn num_nodes(&self) -> usize;

    /// Returns the base-two logarithm of the number of nodes in the layer, i.e.
    /// the number of variables of the MLE of the node values
    fn log_num_nodes(&self) -> usize {
        log2(self.num_nodes()) as usize
    }

    /// Evaluate the layer on the given input natively.
    fn evaluate(&self, input: Vec<QSmallType>) -> Vec<QSmallType>;

    // TODO: is it okay to trim all keys from the same original PCS key?
    // (e.g. to trim the same key to for the matrix and for the bias in the
    // case of MatMul)
    // fn setup(&self, params: PCS::UniversalParams) -> (, Self::VerifierKey);

    /// Evaluate the layer on the given input natively.
    fn commit(&self) -> PCS::Commitment;

    /// Prove that the layer was executed correctly on the given input.
    fn prove(com: Self::Commitment, input: Vec<F>) -> Self::Proof;

    /// Check that the layer transition was executed correctly.
    fn check(com: Self::Commitment, proof: Self::Proof) -> bool;
}

pub(crate) enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    MatMul(FCNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
}

impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn num_nodes(&self) -> usize {
        1 << self.log_num_nodes()
    }

    pub(crate) fn log_num_nodes(&self) -> usize {
        match self {
            Node::MatMul(n) => n.log_num_nodes(),
            Node::ReLU(r) => r.log_num_nodes(),
        }
    }

    pub(crate) fn evaluate(&self, input: Vec<QSmallType>) -> Vec<QSmallType> {
        match self {
            Node::MatMul(n) => n.evaluate(input),
            Node::ReLU(r) => r.evaluate(input),
        }
    }

    pub(crate) fn commit(&self) -> PCS::Commitment {
        match self {
            Node::MatMul(n) => n.commit(),
            Node::ReLU(r) => r.commit(),
        }
    }

    pub(crate) fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    pub(crate) fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }
}
