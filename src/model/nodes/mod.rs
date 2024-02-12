use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;

use crate::{
    model::{
        nodes::{fc::FCNode, relu::ReLUNode},
        CryptographicSponge, Poly,
    },
    quantization::QSmallType,
};

use self::{
    fc::{FCNodeCommitment, FCNodeCommitmentState, FCNodeProof},
    loose_fc::{LooseFCNode, LooseFCNodeCommitment, LooseFCNodeCommitmentState, LooseFCNodeProof},
    relu::{ReLUNodeCommitment, ReLUNodeCommitmentState, ReLUNodeProof},
    reshape::{ReshapeNode, ReshapeNodeCommitment, ReshapeNodeCommitmentState, ReshapeNodeProof},
};

use super::qarray::QArray;

pub(crate) mod fc;
pub(crate) mod loose_fc;
pub(crate) mod relu;
pub(crate) mod reshape;

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub(crate) trait NodeOps<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// A commitment associated to the node's transition function. For
    /// instance, it is empty for a ReLU transition; and a commitment to the
    /// matrix and bias for a MatMul transition.
    type NodeCommitment;

    /// The state (for instance, randomness) associated to the NodeCommitment
    type NodeCommitmentState;

    /// A proof of execution of the node's transition function to a particular
    /// set of node values
    type Proof;

    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize>;

    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    // TODO we could apply next_power_of_two to self.shape() elementwise, but
    // I expect this to be less efficient since each implementor will likely
    // internally store padded_shape_log
    fn padded_shape_log(&self) -> Vec<usize>;

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log()
            .into_iter()
            .map(|x| 1 << x)
            .collect()
    }

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    /// Returns the maximum number of variables of the MLEs committed to as part of
    /// this nodes's commitment.
    fn com_num_vars(&self) -> usize;

    /// Evaluate the node natively (without padding)
    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType>;

    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType>;

    /// Commit to the node parameters
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (Self::NodeCommitment, Self::NodeCommitmentState);

    /// Produce a node output proof
    fn prove(
        node_com: Self::NodeCommitment,
        input: QArray<QSmallType>,
        input_com: PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: PCS::Commitment,
    ) -> Self::Proof;

    /// Verify a node output proof
    fn verify(node_com: Self::NodeCommitment, proof: Self::Proof) -> bool;
}

pub(crate) enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNode<F, S, PCS>),
    LooseFC(LooseFCNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
    Reshape(ReshapeNode<F, S, PCS>),
}

pub(crate) enum NodeProof {
    FCProof(FCNodeProof),
    LooseFCProof(LooseFCNodeProof),
    ReLUProof(ReLUNodeProof),
    ReshapeProof(ReshapeNodeProof),
}

pub(crate) enum NodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FCCommitment(FCNodeCommitment<F, S, PCS>),
    LooseFCCommitment(LooseFCNodeCommitment<F, S, PCS>),
    ReLUCommitment(ReLUNodeCommitment),
    ReshapeCommitment(ReshapeNodeCommitment),
}

pub(crate) enum NodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FCCommitmentState(FCNodeCommitmentState<F, S, PCS>),
    LooseFCCommitmentState(LooseFCNodeCommitmentState<F, S, PCS>),
    ReLUCommitmentState(ReLUNodeCommitmentState),
    ReshapeCommitmentState(ReshapeNodeCommitmentState),
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Return the type of the node
    // This cannot be cleanly achieved by deriving Debug
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Node::FC(_) => "FC",
            Node::LooseFC(_) => "LooseFC",
            Node::ReLU(_) => "ReLU",
            Node::Reshape(_) => "Reshape",
        }
    }

    /// Returns the shape of the node's output tensor
    pub(crate) fn shape(&self) -> Vec<usize> {
        match self {
            Node::FC(fc) => fc.shape(),
            Node::LooseFC(fc) => fc.shape(),
            Node::ReLU(r) => r.shape(),
            Node::Reshape(r) => r.shape(),
        }
    }

    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    pub(crate) fn padded_shape_log(&self) -> Vec<usize> {
        match self {
            Node::FC(fc) => fc.padded_shape_log(),
            Node::LooseFC(fc) => fc.padded_shape_log(),
            Node::ReLU(r) => r.padded_shape_log(),
            Node::Reshape(r) => r.padded_shape_log(),
        }
    }

    /// Returns the element-wise padded node's output shape
    pub(crate) fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log()
            .into_iter()
            .map(|x| 1 << x)
            .collect()
    }

    /// The number of output units of the node
    pub(crate) fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// The number of output units of the padded node
    pub(crate) fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    pub(crate) fn com_num_vars(&self) -> usize {
        match self {
            Node::FC(fc) => fc.com_num_vars(),
            Node::LooseFC(fc) => fc.com_num_vars(),
            Node::ReLU(r) => r.com_num_vars(),
            Node::Reshape(r) => r.com_num_vars(),
        }
    }

    /// Evaluate the node natively (without padding)
    pub(crate) fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        match self {
            Node::FC(fc) => fc.evaluate(input),
            Node::LooseFC(fc) => fc.evaluate(input),
            Node::ReLU(r) => r.evaluate(input),
            Node::Reshape(r) => r.evaluate(input),
        }
    }

    /// Evaluate the padded node natively
    pub(crate) fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        match self {
            Node::FC(fc) => fc.padded_evaluate(input),
            Node::LooseFC(fc) => fc.padded_evaluate(input),
            Node::ReLU(r) => r.padded_evaluate(input),
            Node::Reshape(r) => r.padded_evaluate(input),
        }
    }

    /// Commit to the node parameters
    pub(crate) fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        // TODO this is very ugly, should start thinking about using trait objects
        match self {
            Node::FC(fc) => {
                let (com, state) = fc.commit(ck, rng);
                (
                    NodeCommitment::FCCommitment(com),
                    NodeCommitmentState::FCCommitmentState(state),
                )
            }
            Node::LooseFC(fc) => {
                let (com, state) = fc.commit(ck, rng);
                (
                    NodeCommitment::LooseFCCommitment(com),
                    NodeCommitmentState::LooseFCCommitmentState(state),
                )
            }
            Node::ReLU(r) => {
                let (com, state) = r.commit(ck, rng);
                (
                    NodeCommitment::ReLUCommitment(com),
                    NodeCommitmentState::ReLUCommitmentState(state),
                )
            }
            Node::Reshape(r) => {
                let (com, state) = r.commit(ck, rng);
                (
                    NodeCommitment::ReshapeCommitment(com),
                    NodeCommitmentState::ReshapeCommitmentState(state),
                )
            }
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
