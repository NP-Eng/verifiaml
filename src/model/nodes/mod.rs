use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::model::{
    nodes::{fc::FCNode, relu::ReLUNode},
    CryptographicSponge, Poly,
};

use self::reshape::ReshapeNode;

use super::qarray::QArray;

pub(crate) mod fc;
pub(crate) mod relu;
pub(crate) mod reshape;

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

    /// A proof of execution of the layer's transition function to a particular
    /// set of node values
    type Proof;

    /// The log2 of the number of output units of the node
    fn num_units(&self) -> usize;

    /// Returns the base-two logarithm of the number of nodes in the layer, i.e.
    /// the number of variables of the MLE of the node values
    fn log_num_units(&self) -> usize {
        log2(self.num_units()) as usize
    }

    /// Evaluate the node natively
    fn evaluate(&self, input: QArray) -> QArray;

    // TODO: is it okay to trim all keys from the same original PCS key?
    // (e.g. to trim the same key to for the matrix and for the bias in the
    // case of MatMul)
    // fn setup(&self, params: PCS::UniversalParams) -> (, Self::VerifierKey);

    /// Commit to the layer parameters
    fn commit(&self) -> PCS::Commitment;

    /// Produce a node output proof
    fn prove(com: Self::Commitment, input: Vec<F>) -> Self::Proof;

    /// Verify a node output proof
    fn check(com: Self::Commitment, proof: Self::Proof) -> bool;
}

pub(crate) enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
    Reshape(ReshapeNode<F, S, PCS>),
}

impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// The number of output units of the node
    pub(crate) fn num_units(&self) -> usize {
        1 << self.log_num_units()
    }

    /// The log2 of the number of output units of the node
    pub(crate) fn log_num_units(&self) -> usize {
        match self {
            Node::FC(n) => n.log_num_units(),
            Node::ReLU(r) => r.log_num_units(),
            Node::Reshape(r) => r.log_num_units(),
        }
    }

    /// Evaluate the node natively
    pub(crate) fn evaluate(&self, input: QArray) -> QArray {
        match self {
            Node::FC(n) => n.evaluate(input),
            Node::ReLU(r) => r.evaluate(input),
            Node::Reshape(r) => r.evaluate(input),
        }
    }

    /// Commit to the layer parameters
    pub(crate) fn commit(&self) -> PCS::Commitment {
        match self {
            Node::FC(n) => n.commit(),
            Node::ReLU(r) => r.commit(),
            Node::Reshape(r) => r.commit(),
        }
    }

    /// Produce a node output proof
    pub(crate) fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    /// Verify a node output proof
    pub(crate) fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }
}
